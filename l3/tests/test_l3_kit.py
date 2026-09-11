from __future__ import annotations

import json
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "timerlm" / "sft" / "tslm"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM))

from l3.grpo import (  # noqa: E402
    LoraReference,
    MAX_ABS_LOG_RATIO,
    REQUIRED_CATEGORIES,
    RewardResult,
    RolloutAccounting,
    _cumulative_from_rows,
    _save_lora_checkpoint,
    deterministic_prompt_order,
    prompt_indices_from_cursor,
    seal_eval_result,
    select_balanced_pool,
    stratified_candidate_indices,
    truncate_jsonl_after_step,
    k3_kl,
    latest_valid_checkpoint,
    load_l3_config,
    max_abs_log_ratio,
    normalize_advantages,
    official_score_any,
    ppo_ratio,
    response_token_mask,
    reward_completion,
    grpo_step,
    grpo_longrun_step,
)
from l3.paired_eval_l3 import paired_report, markdown_table  # noqa: E402


CONFIG = ROOT / "l3" / "configs" / "l3_smoke.yaml"
LONG_CONFIG = ROOT / "l3" / "configs" / "l3_longrun.yaml"


class OfficialFakeDataset:
    category_key = "category"
    primary_metric = "primary"
    EOS_TOKEN = "</s>"

    def extract_answer(self, prediction, sample):
        return (prediction or "").strip()

    def get_ground_truth(self, sample):
        return sample["answer"].replace(self.EOS_TOKEN, "").strip()

    def evaluate_answer(self, prediction, sample):
        result = official_score_any(prediction, self.get_ground_truth(sample))
        result["correct"] = bool(result["primary"] >= 1.0 - 1e-9)
        return result


def test_config_resolves_fixed_smoke_contract():
    config = load_l3_config(CONFIG)
    rl = config["rl"]
    assert config["backbone"]["model_id"] == "Qwen/Qwen3.5-4B"
    assert config["model"]["architecture"] == "wavelet"
    assert rl["G"] == 4
    assert rl["steps"] == 20
    assert len(rl["train_prompt_indices"]) == 20
    assert rl["memory_gate_index"] == 3804
    assert config["_l3"]["base_config_path"].endswith("wavelet_precise_va1.yaml")


def test_longrun_config_is_distinct_and_fixed():
    config = load_l3_config(LONG_CONFIG)
    rl = config["rl"]
    assert rl["contract"] == "longrun"
    assert (rl["G"], rl["prompts_per_step"], rl["steps"]) == (8, 2, 400)
    assert rl["weight_decay"] == 0.0 and rl["checkpoint_every"] == 25
    assert rl["memory_gate_indices"] == [3804, 3930]
    broken = json.loads(json.dumps(config))
    broken["rl"]["G"] = 4
    from l3.grpo import validate_l3_config
    with pytest.raises(ValueError, match="longrun"):
        validate_l3_config(broken)


def test_reward_uses_official_parser_and_primary():
    dataset = OfficialFakeDataset()
    sample = {
        "category": "localize",
        "answer": 'localize:{"_L":100,"present":true,"start":10,"end":20}</s>',
    }
    good = reward_completion(
        dataset,
        'localize:{"_L":100,"present":true,"start":10,"end":20}',
        sample,
    )
    malformed = reward_completion(dataset, "not json", sample, format_penalty=0.2)
    wrong_category = reward_completion(
        dataset,
        'measure_magnitude:{"_L":100,"present":true,"start":10,"end":20}',
        sample,
        format_penalty=0.2,
    )
    assert good.parsable and good.primary == 1.0 and good.optimization == 1.0
    assert not malformed.parsable and malformed.primary == 0.0
    assert malformed.format_adjustment == -0.2 and malformed.optimization == -0.2
    assert not wrong_category.parsable and wrong_category.primary == 0.0


def test_advantages_normalized_and_zero_variance_counted():
    advantages, zero = normalize_advantages(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert not zero
    assert torch.isclose(advantages.mean(), torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(advantages.square().mean(), torch.tensor(1.0), atol=1e-6)
    flat, zero = normalize_advantages(torch.tensor([0.4, 0.4, 0.4, 0.4]))
    assert zero
    assert torch.equal(flat, torch.zeros(4))


class TinyLoraModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor([9.0]))
        self.lora_a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.block = torch.nn.Module()
        self.block.lora_b = torch.nn.Parameter(torch.tensor([3.0]))


def test_reference_swap_uses_initial_lora_and_round_trips_exactly():
    model = TinyLoraModel()
    reference = LoraReference(model)
    with torch.no_grad():
        model.lora_a.add_(10)
        model.block.lora_b.mul_(7)
    current = {name: value.detach().clone() for name, value in model.named_parameters()}
    with reference.applied(model):
        assert torch.equal(model.lora_a, torch.tensor([1.0, 2.0]))
        assert torch.equal(model.block.lora_b, torch.tensor([3.0]))
        assert torch.equal(model.base, torch.tensor([9.0]))
    for name, value in model.named_parameters():
        assert torch.equal(value, current[name])


def test_k3_matches_hand_calculation_and_is_token_level():
    policy = torch.log(torch.tensor([0.5, 0.25]))
    reference = torch.log(torch.tensor([0.25, 0.5]))
    delta = reference - policy
    expected = torch.exp(delta) - delta - 1.0
    actual = k3_kl(policy, reference)
    assert torch.allclose(actual, expected)
    assert torch.all(actual >= 0)


def test_k3_is_stable_finite_and_nonnegative_near_zero():
    policy = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
    reference = policy + torch.tensor([1e-7, -1e-7, 0.0])
    actual = k3_kl(policy, reference)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert torch.all(actual >= 0)
    assert torch.all(actual < 1e-12)


def test_zero_policy_and_kl_gradient_with_zero_decay_is_a_fixed_point():
    config = load_l3_config(LONG_CONFIG)
    parameter = torch.nn.Parameter(torch.tensor([1.25, -0.5]))
    optimizer = torch.optim.AdamW(
        [parameter], lr=config["rl"]["lr"], weight_decay=config["rl"]["weight_decay"]
    )
    before = parameter.detach().clone()
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert torch.equal(parameter, before)


def test_response_mask_marks_only_answer_tokens():
    mask = response_token_mask(prompt_tokens=3, answer_tokens=2)
    assert mask.tolist() == [False, False, False, True, True]
    with pytest.raises(ValueError):
        response_token_mask(3, 0)


def test_ratio_exact_flag_is_diagnostic_but_log_ratio_has_tolerant_gate():
    log_probs = torch.tensor([-0.1, -2.5, -0.7])
    ratio = ppo_ratio(log_probs, log_probs.detach().clone())
    assert torch.equal(ratio, torch.ones_like(ratio))
    perturbed = log_probs + torch.tensor([0.0, MAX_ABS_LOG_RATIO / 2, 0.0])
    perturbed_ratio = ppo_ratio(perturbed, log_probs)
    assert not torch.equal(perturbed_ratio, torch.ones_like(perturbed_ratio))
    assert max_abs_log_ratio(perturbed, log_probs) <= MAX_ABS_LOG_RATIO


class TinyTokenizer:
    eos_token_id = 7
    pad_token_id = 0

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ['localize:{"_L":100,"present":true,"start":10,"end":20}']


class TinyLlm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_gain = torch.nn.Parameter(torch.tensor(0.1))
        self.training_modes = []

    def forward(self, *, inputs_embeds, attention_mask, use_cache, logits_to_keep):
        self.training_modes.append(self.training)
        vocab = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
        kernel_delta = torch.zeros_like(vocab)
        if torch.is_grad_enabled():
            kernel_delta[..., 3] = MAX_ABS_LOG_RATIO / 2
        logits = (self.lora_gain * vocab + kernel_delta).expand(
            1, inputs_embeds.shape[1], 8
        )
        return SimpleNamespace(logits=logits)


class TinyGrpoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.llm_model = TinyLlm()
        self.tokenizer = TinyTokenizer()
        self.generate_modes = []

    def prepare_batch(self, samples, training=False):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "time_series": torch.zeros((1, 1)),
        }

    def _encode_and_project(self, time_series):
        return torch.zeros((1, 1))

    def _merge_features(self, ts_features, full_ids):
        return full_ids.float().unsqueeze(-1)

    def generate(self, **kwargs):
        self.generate_modes.append(self.training)
        return torch.tensor([[3]])


def test_grpo_generates_in_eval_then_teacher_forces_in_train_mode():
    model = TinyGrpoModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    dataset = OfficialFakeDataset()
    sample = {
        "_l3_train_index": 46,
        "id": "tiny",
        "category": "localize",
        "length": 100,
        "n_channels": 1,
        "answer": 'localize:{"_L":100,"present":true,"start":10,"end":20}</s>',
    }
    rl = {
        "G": 1,
        "max_new_tokens": 1,
        "temperature": 0.7,
        "top_p": 0.95,
        "format_penalty": 0.0,
        "advantage_eps": 1e-8,
        "clip_eps": 0.2,
        "kl_coef": 0.01,
        "max_grad_norm": 1.0,
    }

    row = grpo_step(model, optimizer, LoraReference(model), dataset, sample, rl, 1)

    assert model.generate_modes == [False]
    assert model.llm_model.training_modes == [True, True, True]
    assert not row["ratio_all_exact_one"]
    assert row["max_abs_log_ratio"] <= MAX_ABS_LOG_RATIO


def test_longrun_step_processes_two_groups_sequentially_and_steps_once():
    class CountingSGD(torch.optim.SGD):
        def __init__(self, parameters):
            super().__init__(parameters, lr=1e-5, weight_decay=0.0)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    model = TinyGrpoModel()
    optimizer = CountingSGD(model.parameters())
    dataset = OfficialFakeDataset()
    samples = [{
        "_l3_train_index": index, "id": f"tiny-{index}", "category": "localize",
        "length": 100, "n_channels": 1,
        "answer": 'localize:{"_L":100,"present":true,"start":10,"end":20}</s>',
    } for index in (46, 47)]
    rl = {
        "G": 2, "prompts_per_step": 2, "max_new_tokens": 1,
        "temperature": 0.7, "top_p": 0.95, "format_penalty": 0.0,
        "advantage_eps": 1e-8, "clip_eps": 0.2, "kl_coef": 0.01,
        "max_grad_norm": 1.0,
    }
    rollout_log = io.StringIO()
    row = grpo_longrun_step(
        model, optimizer, LoraReference(model), dataset, samples, rl, 1, rollout_log
    )
    rollout_rows = [json.loads(line) for line in rollout_log.getvalue().splitlines()]
    assert optimizer.step_calls == 1
    assert model.generate_modes == [False] * 4
    assert row["rollouts"] == 4 and len(row["prompts"]) == 2
    assert row["informative_groups"] == 0 and row["policy_informative_steps"] == 0
    assert all({"step", "prompt", "train_index", "category", "reward_primary",
                "reward_optimization", "parsable", "response_tokens", "text_sha256"} <= set(r)
               for r in rollout_rows)
    assert [r["prompt"] for r in rollout_rows] == [1, 1, 2, 2]
    assert math.isfinite(row["policy_grad_norm"]) and math.isfinite(row["kl_grad_norm"])


def test_positive_rollout_accounting_detects_loss():
    tracker = RolloutAccounting(expected=2)
    tracker.record(RewardResult(1.0, 1.0, 0.0, True, "{}"), 3)
    with pytest.raises(RuntimeError, match="lost rollout"):
        tracker.finalize()
    tracker.record(RewardResult(0.0, 0.0, 0.0, False, "bad"), 2)
    counters = tracker.finalize()
    assert counters["rollouts"] == 2
    assert counters["parsable"] == 1
    assert counters["unparsable"] == 1
    assert counters["tokens_answer"] == 5
    assert math.isclose(counters["reward_mean"], 0.5)


def test_pool_stratification_balance_and_seed_are_deterministic():
    metadata = [
        {"index": i, "category": category, "length": 100 * (1 + i % 2)}
        for i, category in enumerate(sorted(REQUIRED_CATEGORIES) * 4)
    ]
    first = stratified_candidate_indices(metadata, 20260911, len(metadata))
    second = stratified_candidate_indices(metadata, 20260911, len(metadata))
    assert first == second and set(first) == set(range(len(metadata)))
    accepted = [
        {**row, "id": str(row["index"]), "n_channels": 1, "reward_std": 0.1}
        for row in metadata
    ]
    selected = select_balanced_pool(accepted, 2)
    assert len(selected) == 2 * len(REQUIRED_CATEGORIES)
    assert all(sum(row["category"] == cat for row in selected) == 2 for cat in REQUIRED_CATEGORIES)


def test_resume_cursor_truncation_has_no_skips_or_duplicate_rollouts(tmp_path):
    order = deterministic_prompt_order(list(range(10)), 20260911)
    first, cursor = prompt_indices_from_cursor(order, 0, 4)
    resumed, final_cursor = prompt_indices_from_cursor(order, cursor, 2)
    uninterrupted, _ = prompt_indices_from_cursor(order, 0, 6)
    assert first + resumed == uninterrupted and final_cursor == 6

    steps = tmp_path / "steps.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    with steps.open("w", encoding="utf-8") as step_handle, rollouts.open("w", encoding="utf-8") as rollout_handle:
        for step in range(1, 4):
            row = {"step": step, "rollouts": 16, "parsable": 16, "unparsable": 0,
                   "informative_groups": 2, "policy_informative_steps": 1}
            step_handle.write(json.dumps(row) + "\n")
            for rollout in range(16):
                rollout_handle.write(json.dumps({"step": step, "rollout": rollout}) + "\n")
    truncate_jsonl_after_step(steps, 2)
    truncate_jsonl_after_step(rollouts, 2)
    step_rows = [json.loads(line) for line in steps.read_text().splitlines()]
    rollout_rows = [json.loads(line) for line in rollouts.read_text().splitlines()]
    assert [row["step"] for row in step_rows] == [1, 2]
    assert len(rollout_rows) == 32 and all(row["step"] <= 2 for row in rollout_rows)
    assert _cumulative_from_rows(step_rows) == {
        "steps": 2, "rollouts": 32, "parsable": 32, "unparsable": 0,
        "informative_groups": 4, "policy_informative_steps": 2,
        "zero_variance_groups": 0, "reward_sum": 0.0, "reward_square_sum": 0.0,
        "tokens_prompt": 0, "tokens_answer": 0, "seconds": 0.0,
    }


def test_checkpoint_contains_optimizer_cursor_rng_and_is_atomic(tmp_path):
    model = TinyLoraModel()
    optimizer = torch.optim.AdamW([model.lora_a, model.block.lora_b], lr=1e-5, weight_decay=0)
    path = tmp_path / "checkpoint.pt"
    from l3.grpo import capture_rng_state
    cumulative = {"steps": 2, "rollouts": 32}
    _save_lora_checkpoint(
        path, model, ["block.lora_b", "lora_a"], {}, 2,
        optimizer=optimizer, cursor=4, rng_state=capture_rng_state(), cumulative=cumulative,
    )
    payload = torch.load(path, weights_only=False)
    assert set(payload["model_state"]) == {"block.lora_b", "lora_a"}
    assert payload["optimizer_state"] is not None
    assert payload["l3"]["pool_cursor"] == 4
    assert payload["l3"]["cumulative"] == cumulative
    assert not list(tmp_path.glob("*.tmp.*"))


def test_resume_chooses_latest_checkpoint_with_valid_pair_of_hashes(tmp_path):
    import hashlib
    records = []
    for step in (0, 25):
        lora = tmp_path / f"step_{step}_lora.pt"
        composed = tmp_path / f"step_{step}_composed.pt"
        torch.save({
            "model_state": {}, "optimizer_state": {},
            "l3": {"steps": step, "pool_cursor": step * 2, "rng_state": {"ok": True}},
        }, lora)
        composed.write_bytes(f"composed-{step}".encode())
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({
            "step": step, "pool_cursor": step * 2,
            "lora_file": lora.name, "lora_sha256": sha(lora),
            "composed_file": composed.name, "composed_sha256": sha(composed),
        })
    record, _ = latest_valid_checkpoint(tmp_path, {"checkpoints": records})
    assert record["step"] == 25
    records[-1]["composed_sha256"] = "0" * 64
    record, _ = latest_valid_checkpoint(tmp_path, {"checkpoints": records})
    assert record["step"] == 0


def test_eval_sha_chain_matches_manifest_before_and_after(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "step_000025_composed.pt"
    checkpoint.write_bytes(b"immutable checkpoint")
    import hashlib
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = tmp_path / "RUN_MANIFEST.json"
    manifest.write_text(json.dumps({"checkpoints": [{
        "step": 25, "composed_file": str(checkpoint.relative_to(tmp_path)),
        "composed_sha256": digest,
    }]}), encoding="utf-8")
    jsonl = tmp_path / "eval.jsonl"
    jsonl.write_text("".join(json.dumps({"primary": 0.25}) + "\n" for _ in range(520)), encoding="utf-8")
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({
        "split": "val", "samples_evaluated": 520, "primary": 0.25,
        "eval_errors": 0, "eval_jsonl": str(jsonl),
    }), encoding="utf-8")
    out = tmp_path / "sealed.json"
    result = seal_eval_result(checkpoint, manifest, raw, out, digest, "val", 520)
    assert result["checkpoint_sha256_before"] == result["checkpoint_sha256_after"] == digest
    assert result["manifest_checkpoint_match"]


def test_paired_eval_synthetic_rows_and_guards():
    reference = [
        {"id": f"id-{i}", "category": "a" if i < 2 else "b", "ground_truth": "g", "primary": 0.0}
        for i in range(4)
    ]
    candidate = [{**row, "primary": 1.0} for row in reference]
    report = paired_report(reference, candidate, split="val", seed=7, expected=4, resamples=200)
    assert report["coverage"] == {"reference": 4, "candidate": 4}
    assert report["primary"]["candidate_minus_reference"]["mean"] == 1.0
    assert set(report["per_category"]) == {"a", "b"}
    assert "| overall | 4 |" in markdown_table(report)
    reordered = list(reversed(candidate))
    with pytest.raises(ValueError, match="identity mismatch"):
        paired_report(reference, reordered, split="val", seed=7, expected=4, resamples=10)
    nonfinite = [dict(row) for row in candidate]
    nonfinite[0]["primary"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        paired_report(reference, nonfinite, split="val", seed=7, expected=4, resamples=10)


def test_split_guard_and_metadata_cover_required_train_cases():
    config = load_l3_config(CONFIG)
    rl = config["rl"]
    assert rl["prompt_split"] == "train"
    assert all(0 <= index < 4488 for index in rl["train_prompt_indices"])
    metadata = [rl["train_prompt_expectations"][str(index)] for index in rl["train_prompt_indices"]]
    assert {row["category"] for row in metadata} == REQUIRED_CATEGORIES
    assert any(row["category"] == "lead_lag_with_magnitude" for row in metadata)
    assert any(
        row["category"] == "localize_all_channels"
        and row["n_channels"] == 16
        and row["length"] == 32768
        for row in metadata
    )
    gate = rl["train_prompt_expectations"][str(rl["memory_gate_index"])]
    assert (gate["category"], gate["n_channels"], gate["length"]) == (
        "localize_all_channels", 16, 131072
    )


def test_test_split_command_is_confined_to_longrun_l3_phase():
    forbidden = "--split" + " test"
    for path in (ROOT / "l3").rglob("*"):
        # The frozen brief describes the eventual point-5 test command in prose;
        # only executable/config inputs are subject to the smoke guard.
        if path.is_file() and path.suffix in {".py", ".sh", ".yaml"}:
            occurrences = path.read_text(encoding="utf-8").count(forbidden)
            if path.name == "node_l3_longrun.sh":
                assert occurrences == 3
                text = path.read_text(encoding="utf-8")
                l3_start = text.index("# L3: the only test access")
                l4_start = text.index("# L4: two untouched-seed replicas")
                assert forbidden not in text[:l3_start]
                assert forbidden not in text[l4_start:]
            elif path.name == "node_l3_zoom.sh":
                assert occurrences == 2
                text = path.read_text(encoding="utf-8")
                z3_start = text.index("# Z3: the only test access")
                assert forbidden not in text[:z3_start]
            else:
                assert occurrences == 0
