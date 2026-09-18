"""AnomalyXL RLM taskset (verifiers v1).

The data + scoring half of the env. Reuses the v0 `AnomalyXL` dataset for the parquet
and the battle-tested scorer; adds the v1 wiring: `setup` places the series the model
analyzes as `context.json`, the prompt tells the model to write its answer to
`answer.json` (via the harness's IPython tool), `finalize` reads it back, and `@reward`
scores it. The series stays on the taskset (keyed by idx), not on the Task, so the long
series never rides the wire. The harness is `RLMLocalHarness` (runs upstream `rlm`).
"""

from __future__ import annotations

import json
import math
import os

from verifiers.v1 import metric, reward
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace

# Reuse the v0 env package (installed as `timeseries-qa`): dataset + scoring.
from ts_datasets import DATASETS

# Reuse the `anomalyxl` generator for Mode A (regenerate the series in `setup`).
from anomalyxl.difficulty import Difficulty
from anomalyxl.pipeline import SampleSpec, _build_row

from .generator import GeneratorConfig, _GeneratedTasks


WAVELET_NOTE_SENTENCE = (
    "Before any other action, execute exactly `print(await wavelet_zoom())` in IPython. "
    "The skill is already loaded: do not import it and do not open or print `context.json`. "
    "It returns the actual channel names and a compact ranked list of candidate intervals, "
    "kind hints and magnitude estimates. Usually answer after that first call; use at most "
    "one `zoom` call if a boundary truly needs refinement. "
)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_final_schema(text: str, gold: str) -> bool:
    """Strict, task-aware JSON validity used only for a small format bootstrap reward.

    This deliberately checks no agreement with the gold values; semantic quality remains
    entirely owned by ``correct_answer`` and the unchanged benchmark scorer.
    """
    try:
        category, sep, _ = gold.partition(":")
        if not sep:
            return False
        obj = json.loads(text.strip())
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(obj, dict):
        return False
    keys = set(obj)
    if category == "localize":
        if keys != {"present", "start", "end"} or not isinstance(obj["present"], bool):
            return False
        if not obj["present"]:
            return all(v is None or _is_int(v) for v in (obj["start"], obj["end"]))
        return _is_int(obj["start"]) and _is_int(obj["end"]) and 0 <= obj["start"] < obj["end"]
    if category == "classify_with_evidence":
        return (keys == {"kind", "start", "end"} and isinstance(obj["kind"], str)
                and bool(obj["kind"]) and _is_int(obj["start"]) and _is_int(obj["end"])
                and 0 <= obj["start"] < obj["end"])
    if category == "measure_magnitude":
        value = obj.get("magnitude_sigma")
        return (keys == {"magnitude_sigma"} and isinstance(value, (int, float))
                and not isinstance(value, bool) and math.isfinite(value) and value > 0)
    if category == "localize_all_channels":
        anomalies = obj.get("anomalies")
        if keys != {"anomalies"} or not isinstance(anomalies, list):
            return False
        return all(
            isinstance(a, dict) and set(a) == {"channel", "start", "end"}
            and isinstance(a["channel"], str) and bool(a["channel"])
            and _is_int(a["start"]) and _is_int(a["end"]) and 0 <= a["start"] < a["end"]
            for a in anomalies
        )
    if category == "lead_lag_with_magnitude":
        return (keys == {"direction", "lag_samples"}
                and obj["direction"] in {"lead", "lag", "independent"}
                and _is_int(obj["lag_samples"]) and obj["lag_samples"] >= 0)
    return False


def _build_note(values_note: str) -> str:
    variant = os.environ.get("TIMERLM_NOTE_VARIANT", "default")
    if variant not in {"", "default", "wavelet"}:
        raise ValueError(f"unsupported TIMERLM_NOTE_VARIANT={variant!r}")
    if variant == "wavelet":
        source_note = '{"series": {channel: [values], ...}}. '
        analysis_note = WAVELET_NOTE_SENTENCE
    else:
        source_note = '{"series": {channel: [values], ...}}; load it with json. '
        analysis_note = (
            "Always read the actual channel keys from the data (do not guess names). "
            "Analyze it with numpy and scipy in the IPython tool. "
        )
    return (
        "This is a critical application: when relevant, precisely locate the anomaly's "
        "start and end using statistical thresholds — be precise. The time series is in "
        'the file `context.json` in your working directory, shaped as '
        + source_note
        + values_note
        + analysis_note
        + "When you are done, stop "
        "calling the tool and state your final answer as a single JSON object matching "
        'the question\'s schema exactly (e.g. {"start": 12, "end": 87}), with no prose '
        "and no markdown fences. Format failures score zero."
    )


def _final_text(trace: Trace) -> str:
    """The model's concluding answer — `rlm`'s final message (it states the answer when
    it stops calling the IPython tool)."""
    msgs = trace.assistant_messages
    return str(msgs[-1].content or "") if msgs else ""


class AnomalyXLTask(Task):
    """One AnomalyXL problem. `answer` is the AnomalyXL-encoded gold the scorer reads."""

    answer: str


class GeneratedAnomalyXLTask(AnomalyXLTask):
    """A generator-backed problem. The sample coords + difficulty θ ride on the rollout
    record so `setup` can regenerate the series deterministically."""

    seed: int
    length: int
    n_channels: int
    category: str
    paired_split: tuple[float, float] | None = None
    difficulty: dict


class AnomalyXLTasksetConfig(TasksetConfig):
    dataset: str = "anomalyxl"
    repo_or_path: str | None = None
    split: str = "train"
    eval_size: int = 200
    extra_dataset_kwargs: dict = {}
    generator: GeneratorConfig | None = None
    """When set, draw tasks from the `anomalyxl` generator (Mode A) instead of the parquet."""


class AnomalyXLTaskset(Taskset[AnomalyXLTask, AnomalyXLTasksetConfig]):
    def __init__(self, config: AnomalyXLTasksetConfig) -> None:
        super().__init__(config)
        cls = DATASETS[config.dataset]
        self._ds = cls(
            repo_or_path=config.repo_or_path,
            eval_size=config.eval_size,
            mode="rlm",
            **config.extra_dataset_kwargs,
        )
        self._series: list[dict] = []  # idx -> {channel: [values]}

    def load_tasks(self):
        if self.config.generator is not None:
            note = _build_note(self._ds.rlm_values_note)
            return _GeneratedTasks(self.config.generator, note)
        frame = self._ds.load_split(self.config.split)
        note = _build_note(self._ds.rlm_values_note)
        tasks: list[AnomalyXLTask] = []
        for i, row in enumerate(frame):
            sample = self._ds.to_sample(row)
            self._series.append(sample["series"])
            tasks.append(
                AnomalyXLTask(
                    idx=i,
                    prompt=f"{note}\n\n{sample['question']}",
                    answer=sample["answer"],
                )
            )
        return tasks

    async def setup(self, task: AnomalyXLTask, runtime: Runtime) -> None:
        """Place the series the model analyzes as `context.json` in the workspace. In
        generator mode the series is regenerated from the task's fields (deterministic +
        cheap); otherwise it's read from the parquet-backed cache."""
        if isinstance(task, GeneratedAnomalyXLTask):
            spec = SampleSpec(
                seed=task.seed,
                length=task.length,
                n_channels=task.n_channels,
                category=task.category,
                paired_split=task.paired_split,
                difficulty=Difficulty(**task.difficulty),
            )
            series = _build_row(spec).series
        else:
            series = self._series[task.idx]
        payload = json.dumps({"series": series}).encode()
        await runtime.write("context.json", payload)

    @reward(weight=1.0)
    async def correct_answer(self, task: AnomalyXLTask, trace: Trace) -> float:
        try:
            return self._ds.score_answer(_final_text(trace), task.answer)
        except Exception:
            # Format failures score zero; a reward function must never raise and
            # take down a training step.
            return 0.0

    @reward(weight=1.0)
    async def answer_format(self, task: AnomalyXLTask, trace: Trace) -> float:
        """Small non-negative bootstrap signal for a complete, schema-valid final JSON.

        The 2B pilot otherwise found a reward loophole: an empty voluntary answer scored 0,
        better than the old -0.25 forced-final penalty. 0.05 is enough to teach the output
        contract while leaving the benchmark reward (range 0..1) dominant.
        """
        return 0.05 if _valid_final_schema(_final_text(trace), task.answer) else 0.0

    @metric
    async def forced_final(self, trace: Trace) -> float:
        """Telemetry only: whether the rollout exhausted its turn budget.

        It must not alter reward: penalizing only forced termination made an empty early
        termination strictly preferable and collapsed the 2B policy after about 20 steps.
        """
        for n in trace.nodes:
            m = n.message
            role = getattr(m, "role", None)
            content = getattr(m, "content", None)
            if role == "user" and isinstance(content, str) and content.startswith("FINAL TURN:"):
                return 1.0
        return 0.0

    @metric
    async def extra_metrics(self, task: AnomalyXLTask, trace: Trace) -> dict:
        """The dataset's per-category metrics (IoU / F1 / …) on the model's answer."""
        return self._ds.score_metrics(_final_text(trace), task.answer)
