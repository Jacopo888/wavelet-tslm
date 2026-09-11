#!/usr/bin/env python3
"""One-zoom episodes for the va1 WaveletTSModel.

The prompt/window helpers are deliberately NumPy-only and CPU-testable.  Model
and dataset imports stay behind the evaluation CLI so importing this module does
not load Transformers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_ZOOM_SAMPLES = 1024
ACTION_INSTRUCTION = (
    'Optional action: before answering, you may output exactly '
    'zoom:{"channel":"<name>","start":s,"end":e}. Use one listed channel '
    'and absolute sample coordinates [start,end). Otherwise output the final answer now.'
)
FINAL_INSTRUCTION = "Now output the final answer in absolute coordinates."
ACTION_HISTORY_PREFIX = "Assistant action: "
_ACTION_RE = re.compile(r"^zoom:(\{.*\})$", re.DOTALL)


@dataclass(frozen=True)
class ZoomAction:
    channel: str
    start: int
    end: int
    length: int

    @property
    def window_len(self) -> int:
        return self.end - self.start

    def text(self) -> str:
        payload = {"channel": self.channel, "start": self.start, "end": self.end}
        return "zoom:" + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ZoomDecision:
    kind: str  # action | final | invalid
    action: ZoomAction | None


@dataclass
class ZoomEpisode:
    final_text: str
    final_ids: torch.Tensor
    final_prompt: dict[str, torch.Tensor]
    action_text: str | None = None
    action_ids: torch.Tensor | None = None
    action_prompt: dict[str, torch.Tensor] | None = None
    action: ZoomAction | None = None
    zoom_invalid: bool = False

    @property
    def zoomed(self) -> bool:
        return self.action is not None


def series_length(sample: Mapping[str, Any]) -> int:
    explicit = int(sample.get("length", 0) or 0)
    channels = sample.get("time_series", ())
    inferred = len(channels[0]) if channels else 0
    length = explicit or inferred
    if length < 1:
        raise ValueError("sample has no positive series length")
    if channels and any(len(channel) != length for channel in channels):
        raise ValueError("all channels must have the declared absolute length")
    return length


def channel_names(sample: Mapping[str, Any]) -> list[str]:
    names = sample.get("channel_names") or []
    if not names:
        names = [str(text).removeprefix("The following is channel ").rstrip(":")
                 for text in sample.get("time_series_text", ())]
    if len(names) != len(sample.get("time_series", ())):
        raise ValueError("channel names/data count mismatch")
    return [str(name) for name in names]


def clamp_zoom_window(start: int, end: int, length: int,
                      min_samples: int = MIN_ZOOM_SAMPLES) -> tuple[int, int] | None:
    """Clamp a non-empty intersecting half-open interval and expand to min length."""
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in (start, end, length, min_samples)):
        return None
    if length <= 0 or min_samples <= 0 or start >= end or end <= 0 or start >= length:
        return None
    start, end = max(0, start), min(length, end)
    target = min(min_samples, length)
    if end - start < target:
        missing = target - (end - start)
        start -= missing // 2
        end += missing - missing // 2
        if start < 0:
            end = min(length, end - start)
            start = 0
        if end > length:
            start = max(0, start - (end - length))
            end = length
    if not (0 <= start < end <= length) or end - start < target:
        return None
    return start, end


def parse_zoom_decision(text: str, names: Sequence[str], length: int,
                        min_samples: int = MIN_ZOOM_SAMPLES) -> ZoomDecision:
    """Strict action parser; non-action text is a final answer, not an error."""
    stripped = (text or "").strip()
    match = _ACTION_RE.fullmatch(stripped)
    if match is None:
        return ZoomDecision("invalid" if stripped.lower().startswith("zoom") else "final", None)
    try:
        payload = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return ZoomDecision("invalid", None)
    if not isinstance(payload, dict) or set(payload) != {"channel", "start", "end"}:
        return ZoomDecision("invalid", None)
    channel, start, end = payload["channel"], payload["start"], payload["end"]
    if not isinstance(channel, str) or channel not in names:
        return ZoomDecision("invalid", None)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)):
        return ZoomDecision("invalid", None)
    window = clamp_zoom_window(start, end, length, min_samples)
    if window is None:
        return ZoomDecision("invalid", None)
    return ZoomDecision("action", ZoomAction(channel, *window, length))


def parse_zoom_action(text: str, names: Sequence[str], length: int,
                      min_samples: int = MIN_ZOOM_SAMPLES) -> ZoomAction | None:
    """Convenience API returning only a valid normalized action."""
    return parse_zoom_decision(text, names, length, min_samples).action


def zoom_header(action: ZoomAction) -> str:
    return (
        f"Zoom on channel {action.channel}, samples "
        f"[{action.start}, {action.end}) of {action.length}:"
    )


def with_action_instruction(sample: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(sample)
    post = str(sample.get("post_prompt", "")).rstrip()
    result["post_prompt"] = f"{post}\n{ACTION_INSTRUCTION}" if post else ACTION_INSTRUCTION
    return result


def build_zoom_block(sample: Mapping[str, Any], action: ZoomAction) -> tuple[str, np.ndarray]:
    names = channel_names(sample)
    if action.length != series_length(sample) or action.channel not in names:
        raise ValueError("action does not belong to this sample")
    index = names.index(action.channel)
    values = np.asarray(sample["time_series"][index], dtype=np.float32)[action.start:action.end]
    if values.size != action.window_len:
        raise RuntimeError("zoom slice length mismatch")
    return zoom_header(action), values.copy()


def make_zoom_prompt(sample: Mapping[str, Any], action: ZoomAction) -> Any:
    """Create the public TextTimeSeriesPrompt representing the zoom block."""
    try:
        from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt
    except ImportError:
        local_checkout = REPO_ROOT / "timerlm" / "sft" / "tslm"
        if (local_checkout / "src").is_dir() and str(local_checkout) not in sys.path:
            sys.path.insert(0, str(local_checkout))
        from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt
    header, values = build_zoom_block(sample, action)
    return TextTimeSeriesPrompt(header, values)


def build_turn2_sample(sample: Mapping[str, Any], action: ZoomAction,
                       *, answer: str | None = None) -> dict[str, Any]:
    """Append the zoom after originals and action history after the question."""
    result = with_action_instruction(sample)
    block = make_zoom_prompt(sample, action)
    result["time_series"] = list(sample["time_series"]) + [block.get_time_series()]
    result["time_series_text"] = list(sample["time_series_text"]) + [block.get_text()]
    post = result["post_prompt"].rstrip()
    result["post_prompt"] = (
        f"{post}\n{ACTION_HISTORY_PREFIX}{action.text()}\n{FINAL_INSTRUCTION}"
    )
    if answer is not None:
        result["answer"] = answer
    return result


def supervised_segments(sample: Mapping[str, Any], action: ZoomAction,
                        eos_token: str) -> tuple[dict[str, Any], dict[str, Any]]:
    first = with_action_instruction(sample)
    first["answer"] = action.text() + eos_token
    gold = str(sample["answer"])
    if eos_token and not gold.endswith(eos_token):
        gold += eos_token
    second = build_turn2_sample(sample, action, answer=gold)
    return first, second


def two_segment_token_mask(prompt1_tokens: int, action_tokens: int,
                           prompt2_tokens: int, answer_tokens: int) -> torch.Tensor:
    """Mask emitted tokens in a conceptual concatenation of the two forwards."""
    from l3.grpo import response_token_mask
    return torch.cat((
        response_token_mask(prompt1_tokens, action_tokens),
        response_token_mask(prompt2_tokens, answer_tokens),
    ))


def _gold_payload(sample: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = str(sample.get("answer", "")).strip()
    category, sep, payload = raw.partition(":")
    if not sep:
        return str(sample.get("category", "")), {}
    # Ignore tokenizer EOS/prose following the first balanced object.
    start, depth, end, in_string, escaped = payload.find("{"), 0, -1, False, False
    if start < 0:
        return category, {}
    for offset, char in enumerate(payload[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = offset + 1
                break
    if end < 0:
        return category, {}
    try:
        value = json.loads(payload[start:end])
    except (json.JSONDecodeError, ValueError):
        value = {}
    return category, value if isinstance(value, dict) else {}


def gold_events(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    category, payload = _gold_payload(sample)
    names = channel_names(sample)
    if category == "localize_all_channels":
        raw = payload.get("anomalies", [])
        return [dict(event) for event in raw if isinstance(event, dict)
                and event.get("channel") in names
                and isinstance(event.get("start"), int) and isinstance(event.get("end"), int)
                and not isinstance(event.get("start"), bool) and not isinstance(event.get("end"), bool)
                and event["start"] < event["end"]]
    if category == "localize" and not bool(payload.get("present", False)):
        return []
    if category == "classify_with_evidence" and payload.get("kind") == "No anomaly":
        return []
    if category in {"localize", "classify_with_evidence"}:
        start, end = payload.get("start"), payload.get("end")
        if (isinstance(start, int) and not isinstance(start, bool)
                and isinstance(end, int) and not isinstance(end, bool) and start < end):
            return [{"channel": names[0], "start": start, "end": end}]
    return []


def _window_around(start: int, end: int, length: int, margin: float) -> tuple[int, int]:
    width = max(1, end - start)
    extra = int(math.ceil(width * margin))
    window = clamp_zoom_window(start - extra, end + extra, length)
    if window is None:
        raise ValueError("gold window does not intersect the sample")
    return window


def oracle_zoom(sample: Mapping[str, Any], margin: float = 0.25,
                *, rng: random.Random | None = None,
                random_multichannel: bool = False) -> ZoomAction | None:
    events = gold_events(sample)
    if not events:
        return None
    names = channel_names(sample)
    ordered = sorted(events, key=lambda e: (names.index(str(e["channel"])), e["start"], e["end"]))
    if random_multichannel and len(ordered) > 1:
        event = (rng or random).choice(ordered)
    else:
        event = min(ordered, key=lambda e: (-(e["end"] - e["start"]),
                                             names.index(str(e["channel"])), e["start"]))
    start, end = _window_around(int(event["start"]), int(event["end"]),
                                series_length(sample), margin)
    return ZoomAction(str(event["channel"]), start, end, series_length(sample))


def heuristic_zoom(sample: Mapping[str, Any], buckets: int = 64) -> ZoomAction:
    """First deterministic multilevel Haar-energy candidate across all channels."""
    try:
        from l2.overlay.timerlm.environments.timeseries_qa.wavelet_pyramid import WaveletPyramid
    except ImportError:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from l2.overlay.timerlm.environments.timeseries_qa.wavelet_pyramid import WaveletPyramid

    length = series_length(sample)
    names = channel_names(sample)
    bucket_count = min(max(1, buckets), length)
    edges = np.linspace(0, length, bucket_count + 1, dtype=int)
    best: tuple[float, int, int] | None = None
    for channel_index, raw in enumerate(sample["time_series"]):
        pyramid = WaveletPyramid(np.asarray(raw, dtype=np.float64))
        scores = np.zeros(bucket_count, dtype=np.float64)
        for level, detail in pyramid.details.items():
            energy = np.square(np.asarray(detail, dtype=np.float64))
            positive = energy[energy > 0]
            scale = float(np.median(positive)) if positive.size else 1.0
            scale = max(scale, np.finfo(np.float64).eps)
            centers = np.arange(energy.size) * (2 ** level) + (2 ** level) // 2
            ids = np.searchsorted(edges, centers, side="right") - 1
            valid = (ids >= 0) & (ids < bucket_count) & (centers < length)
            np.add.at(scores, ids[valid], energy[valid] / scale)
        bucket = int(np.argmax(scores))
        candidate = (float(scores[bucket]), -channel_index, -bucket)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    channel_index, bucket = -best[1], -best[2]
    target = min(length, max(MIN_ZOOM_SAMPLES, math.ceil(length / 8)))
    center = (int(edges[bucket]) + int(edges[bucket + 1])) // 2
    window = clamp_zoom_window(center - target // 2, center + (target - target // 2), length)
    if window is None:
        raise RuntimeError("heuristic produced an invalid window")
    return ZoomAction(names[channel_index], *window, length)


def zoom_iou_with_gold(sample: Mapping[str, Any], action: ZoomAction | None) -> float:
    if action is None:
        return 0.0
    values = []
    for event in gold_events(sample):
        if str(event["channel"]) != action.channel:
            continue
        intersection = max(0, min(action.end, int(event["end"])) -
                           max(action.start, int(event["start"])))
        union = action.window_len + int(event["end"]) - int(event["start"]) - intersection
        values.append(intersection / union if union else 0.0)
    return max(values, default=0.0)


def choose_supervised_action(sample: Mapping[str, Any], rng: random.Random,
                             heuristic_fraction: float = 0.30) -> tuple[ZoomAction, str]:
    if rng.random() < heuristic_fraction:
        return heuristic_zoom(sample), "heuristic"
    margin = rng.uniform(0.10, 0.50)
    action = oracle_zoom(sample, margin, rng=rng, random_multichannel=True)
    return (action, "gold_jitter") if action is not None else (heuristic_zoom(sample), "heuristic")


def _model_inputs(model: Any, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    values = model.prepare_batch([sample], training=False)
    return {key: value for key, value in values.items() if key != "labels"}


def prepare_zoom_batch(model: Any, sample: Mapping[str, Any], action: ZoomAction,
                       *, training: bool, answer: str | None = None) -> dict[str, torch.Tensor]:
    """Prepare text jointly but encode originals/window at their own lengths.

    ChatTS ``prepare_batch`` pads all blocks to the longest channel.  Calling the
    encoder twice here is what makes the zoom a true full-resolution reread rather
    than a short prefix followed by zeros up to L.
    """
    turn2 = build_turn2_sample(sample, action, answer=answer)
    prepared = model.prepare_batch([turn2], training=training)
    original = model.prepare_batch([sample], training=False)
    _, values = build_zoom_block(sample, action)
    dtype = next(model.llm_model.parameters()).dtype
    zoom_tensor = torch.as_tensor(values, dtype=dtype, device=original["time_series"].device).unsqueeze(0)
    from l3.grpo import _autocast_for
    with torch.no_grad(), _autocast_for(model):
        original_features = model._encode_and_project(original["time_series"])
        zoom_features = model._encode_and_project(zoom_tensor)
    prepared["_l3_ts_features"] = torch.cat((original_features, zoom_features), dim=0).detach()
    tokenizer = getattr(model, "text_tokenizer", None)
    if tokenizer is not None:
        pad_id = tokenizer(model.TS_PAD_TOKEN, add_special_tokens=False)["input_ids"][0]
        positions = int((prepared["input_ids"] == pad_id).sum().item())
        if positions != int(prepared["_l3_ts_features"].shape[0]):
            raise RuntimeError(
                f"zoom feature/placeholder mismatch: {prepared['_l3_ts_features'].shape[0]} != {positions}"
            )
    if not training:
        prepared.pop("labels", None)
    return prepared


def zoom_training_loss(model: Any, prepared: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Memory-efficient answer loss using separately encoded zoom features."""
    labels = prepared.get("labels")
    if labels is None or labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("zoom training loss requires one labeled sample")
    features = prepared.get("_l3_ts_features")
    if features is None:
        raise ValueError("zoom training batch is missing separate block features")
    input_ids = prepared["input_ids"]
    shift_labels = labels[..., 1:].contiguous()
    keep = torch.nonzero(shift_labels[0] != -100, as_tuple=False).flatten()
    if keep.numel() == 0:
        raise ValueError("zoom training answer segment is empty")
    inputs_embeds = model._merge_features(features, input_ids)
    outputs = model.llm_model(
        inputs_embeds=inputs_embeds,
        attention_mask=prepared["attention_mask"],
        use_cache=False,
        logits_to_keep=keep,
    )
    logits = outputs.logits
    if logits.shape[1] != keep.numel():
        logits = logits.index_select(1, keep)
    targets = shift_labels.index_select(1, keep)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def _generate(model: Any, prompt: Mapping[str, torch.Tensor], *, sample: bool,
              max_new_tokens: int, temperature: float = 0.7,
              top_p: float = 0.95) -> torch.Tensor:
    from l3.grpo import _autocast_for, _trim_completion
    kwargs: dict[str, Any] = {"do_sample": sample, "max_new_tokens": max_new_tokens}
    if sample:
        kwargs.update(temperature=temperature, top_p=top_p)
    features = prompt.get("_l3_ts_features")
    with _autocast_for(model):
        if features is None:
            ids = model.generate(**prompt, **kwargs)
        else:
            input_ids = prompt["input_ids"]
            inputs_embeds = model._merge_features(features, input_ids)
            generated = model.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prompt.get("attention_mask"),
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                **kwargs,
            )
            ids = generated[:, input_ids.shape[1]:] if generated.shape[1] > input_ids.shape[1] else generated
    return _trim_completion(ids, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id)


def run_zoom_episode(model: Any, sample: Mapping[str, Any], *, sample_generation: bool,
                     max_action_tokens: int = 500, max_answer_tokens: int = 500,
                     temperature: float = 0.7, top_p: float = 0.95,
                     forced_action: ZoomAction | None = None) -> ZoomEpisode:
    """Run policy action+answer, or a forced-action answer for Z0."""
    if forced_action is None:
        first_sample = with_action_instruction(sample)
        first_prompt = _model_inputs(model, first_sample)
        model.eval()
        first_ids = _generate(model, first_prompt, sample=sample_generation,
                              max_new_tokens=max_action_tokens,
                              temperature=temperature, top_p=top_p)
        first_text = model.tokenizer.batch_decode(first_ids, skip_special_tokens=True)[0].strip()
        decision = parse_zoom_decision(first_text, channel_names(sample), series_length(sample))
        if decision.kind != "action":
            return ZoomEpisode(first_text, first_ids, first_prompt,
                               zoom_invalid=decision.kind == "invalid")
        action, action_ids, action_prompt, action_text = (
            decision.action, first_ids, first_prompt, first_text
        )
    else:
        action, action_ids, action_prompt, action_text = forced_action, None, None, forced_action.text()
    assert action is not None
    second_prompt = prepare_zoom_batch(model, sample, action, training=False)
    model.eval()
    final_ids = _generate(model, second_prompt, sample=sample_generation,
                          max_new_tokens=max_answer_tokens,
                          temperature=temperature, top_p=top_p)
    final_text = model.tokenizer.batch_decode(final_ids, skip_special_tokens=True)[0].strip()
    return ZoomEpisode(final_text, final_ids, second_prompt, action_text, action_ids,
                       action_prompt, action, False)


def _load_eval_runtime(config_path: str, checkpoint: str, split: str) -> tuple[Any, Any, dict[str, Any]]:
    from l3.grpo import _runtime_imports, configure_lora_only, load_l3_config
    resolved = load_l3_config(config_path)
    ExperimentConfig, configure_training_mode, load_checkpoint, load_model, _ = _runtime_imports(resolved)
    from src.datasets.anomaly_xl.qa_dataset import AnomalyXLDataset
    experiment = ExperimentConfig.from_dict({k: v for k, v in resolved.items() if not k.startswith("_")})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(resolved.get("seed", 42)))
    model = load_model(experiment, device)
    configure_training_mode(model, experiment)
    configure_lora_only(model)
    load_checkpoint(str(Path(checkpoint).expanduser()), model)
    kwargs = dict(experiment.dataset.extra_kwargs)
    kwargs["EOS_TOKEN"] = model.get_eos_token()
    dataset = AnomalyXLDataset(split="validation" if split == "val" else split, **kwargs)
    return model, dataset, resolved


def evaluate_zoom(args: argparse.Namespace) -> None:
    from l3.grpo import _atomic_json_dump, _write_jsonl, reward_completion, sha256_file
    out = Path(args.out_json).expanduser().resolve()
    jsonl = Path(args.out_jsonl).expanduser().resolve()
    if out.exists() or jsonl.exists():
        raise FileExistsError("zoom evaluation outputs already exist")
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    model, dataset, resolved = _load_eval_runtime(args.config, args.checkpoint, args.split)
    limit = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    metric_sums: dict[str, dict[str, float]] = {}
    zoom_lengths: dict[str, list[int]] = {}
    zoomed = invalid = 0
    iou_sum = 0.0
    with jsonl.open("x", encoding="utf-8") as handle:
        for index in range(limit):
            sample = dataset[index]
            mode = args.mode
            if mode == "no_zoom":
                prompt = _model_inputs(model, sample)
                model.eval()
                ids = _generate(model, prompt, sample=False, max_new_tokens=args.max_answer_tokens)
                text = model.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
                episode = ZoomEpisode(text, ids, prompt)
            else:
                forced = None
                if mode == "oracle":
                    forced = oracle_zoom(sample) or heuristic_zoom(sample)
                elif mode == "heuristic":
                    forced = heuristic_zoom(sample)
                episode = run_zoom_episode(
                    model, sample, sample_generation=False,
                    max_action_tokens=args.max_action_tokens,
                    max_answer_tokens=args.max_answer_tokens,
                    forced_action=forced,
                )
            result = dataset.evaluate_answer(episode.final_text, sample)
            category = str(sample["category"])
            primary = float(result[dataset.primary_metric])
            counts[category] = counts.get(category, 0) + 1
            sums[category] = sums.get(category, 0.0) + primary
            bucket = metric_sums.setdefault(category, {})
            for key, value in result.items():
                if key != "category" and isinstance(value, (int, float)) and not isinstance(value, bool):
                    bucket[key] = bucket.get(key, 0.0) + float(value)
            if episode.zoomed:
                zoomed += 1
                zoom_lengths.setdefault(category, []).append(episode.action.window_len)
                iou_sum += zoom_iou_with_gold(sample, episode.action)
            invalid += int(episode.zoom_invalid)
            _write_jsonl(handle, {
                "index": index, "id": str(sample.get("id", "")), "category": category,
                "ground_truth": dataset.get_ground_truth(sample), "prediction": episode.final_text,
                "primary": primary, **{k: v for k, v in result.items() if k != "primary"},
                "zoom": None if episode.action is None else {
                    "channel": episode.action.channel, "start": episode.action.start,
                    "end": episode.action.end, "window_len": episode.action.window_len,
                    "iou_with_gold": zoom_iou_with_gold(sample, episode.action),
                }, "zoom_invalid": episode.zoom_invalid,
            })
    total_primary = sum(sums.values()) / max(1, limit)
    summary = {
        "status": "DONE", "split": args.split, "mode": args.mode,
        "samples_evaluated": limit, "primary": total_primary,
        "per_category": {cat: {"count": counts[cat],
                                **{key: value / counts[cat]
                                   for key, value in metric_sums[cat].items()},
                                "primary": sums[cat] / counts[cat],
                                "zoom_window_len": (sum(zoom_lengths.get(cat, [])) /
                                                    len(zoom_lengths[cat]))
                                if zoom_lengths.get(cat) else 0.0}
                         for cat in sorted(counts)},
        "zoom_rate": zoomed / max(1, limit), "zoom_invalid": invalid,
        "zoom_window_len": (sum(map(sum, zoom_lengths.values())) /
                            max(1, sum(map(len, zoom_lengths.values())))),
        "zoom_iou_with_gold": iou_sum / max(1, zoomed),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": sha256_file(Path(args.checkpoint).expanduser()),
        "eval_jsonl": str(jsonl), "config": resolved,
    }
    _atomic_json_dump(out, summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("val", "test"), required=True)
    evaluate.add_argument("--mode", choices=("no_zoom", "oracle", "heuristic", "policy"), required=True)
    evaluate.add_argument("--out-json", required=True)
    evaluate.add_argument("--out-jsonl", required=True)
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--max-action-tokens", type=int, default=500)
    evaluate.add_argument("--max-answer-tokens", type=int, default=500)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "eval":
        evaluate_zoom(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
