from __future__ import annotations

import json
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
    k3_kl,
    load_l3_config,
    max_abs_log_ratio,
    normalize_advantages,
    official_score_any,
    ppo_ratio,
    response_token_mask,
    reward_completion,
    grpo_step,
)


CONFIG = ROOT / "l3" / "configs" / "l3_smoke.yaml"


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


def test_no_test_split_command_in_l3_sources():
    forbidden = "--split" + " test"
    for path in (ROOT / "l3").rglob("*"):
        # The frozen brief describes the eventual point-5 test command in prose;
        # only executable/config inputs are subject to the smoke guard.
        if path.is_file() and path.suffix in {".py", ".sh", ".yaml"}:
            assert forbidden not in path.read_text(encoding="utf-8")
