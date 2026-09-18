from __future__ import annotations

import re
from pathlib import Path

import pytest

from wavelet_rope.config import load_runtime_config
from wavelet_rope.errors import ConfigError, RequiredValueError

PROJECT_ROOT = Path(__file__).parents[1]


def _resolved_config(tmp_path: Path, **replacements: str) -> Path:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(exist_ok=True)
    dataset_manifest = dataset_root / "manifest.json"
    dataset_manifest.write_text('{"splits":{"train":4488,"validation":520,"test":800}}')
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir(exist_ok=True)
    snapshot_manifest = snapshot_root / "snapshot.json"
    snapshot_manifest.write_text('{"verified":true}')
    run_root = tmp_path / "runs"
    values = {
        "run_root": run_root.as_posix(),
        "seed": "7",
        "dataset_root": dataset_root.as_posix(),
        "dataset_manifest": dataset_manifest.as_posix(),
        "transformers_version": "DECIDED-IN-TEST",
        "tokenizer_revision": "DECIDED-IN-TEST",
        "mad_epsilon": "1e-6",
        "world_size": "1",
        "optimizer_name": "DECIDED-IN-TEST",
        "beta1": "0.9",
        "beta2": "0.95",
        "optimizer_epsilon": "1e-8",
        "lr_scheduler": "DECIDED-IN-TEST",
        "p3_frontend_lr": "1e-4",
        "p3_lora_lr": "5e-5",
        "segment_loss_reduction": "DECIDED-IN-TEST",
        "checkpoint_steps": "100",
        "keep_last": "2",
        "selection_metric": "DECIDED-IN-TEST",
        "selection_tiebreaker": "DECIDED-IN-TEST",
        "logging_steps": "10",
        "gpu_family": "MI300X",
        "platform": "rocm",
        "attention_backend": "FLEX_PAD512",
        "attention_qk_pad_to": "512",
    }
    values.update(replacements)
    spec = (PROJECT_ROOT / "prototype.spec.json").as_posix()
    text = f'''schema_version = 1
phase = "P0"
smoke_only = true
architecture_spec = "{spec}"
run_root = "{values['run_root']}"
seed = {values['seed']}

[dataset]
root = "{values['dataset_root']}"
manifest = "{values['dataset_manifest']}"
repo = "nz00shuuuu/AnomalyXL"
revision = "df363a467f855e55aaad414b0bdd58590829df9a"
train_relative_path = "anomalyXL/datasets/anomalyxl-train/anomalyxl-precise/train.parquet"
train_rows = 4488
train_sha256 = "b8571d886772a55613c2e8851a774e6a561e19c8272227f5abb70e2d76e70b7d"
validation_relative_path = "anomalyXL/datasets/anomalyxl-train/anomalyxl-precise/val.parquet"
validation_rows = 520
validation_sha256 = "90b805b8f40e203b76873a9cf3a0f986e21700c9185c74acc33c44a183b1d977"
test_relative_path = "anomalyXL/datasets/anomalyxl-precise/data.parquet"
test_rows = 800
test_sha256 = "7b37269d106f70ba7bee5feb504c00059bc880217a56a8279a2631c8bf5e1524"
forbidden_fit_row_counts = [723, 77]

[backbone]
transformers_version = "{values['transformers_version']}"
tokenizer_revision = "{values['tokenizer_revision']}"

[backbone.snapshot]
root = "{snapshot_root.as_posix()}"
verification_manifest = "{snapshot_manifest.as_posix()}"
cache_scope = "run_local_verified_scratch"
allow_shared_cache = false
local_files_only = true
allow_download = false
quantization = "none"

[normalization]
mad_epsilon = {values['mad_epsilon']}

[runtime]
gpu_family = "{values['gpu_family']}"
platform = "{values['platform']}"
world_size = {values['world_size']}
attention_backend = "{values['attention_backend']}"
attention_qk_pad_to = {values['attention_qk_pad_to']}
torch_build = "DECIDED-IN-TEST"
accelerator_runtime = "DECIDED-IN-TEST"
compute_dtype = "bf16"
gradient_checkpointing = true
requires_cuda_gate = true
requires_real_qwen_smoke = true

[optimizer]
name = "{values['optimizer_name']}"
beta1 = {values['beta1']}
beta2 = {values['beta2']}
epsilon = {values['optimizer_epsilon']}
lr_scheduler = "{values['lr_scheduler']}"
p3_frontend_learning_rate = {values['p3_frontend_lr']}
p3_lora_learning_rate = {values['p3_lora_lr']}

[training]
segment_loss_reduction = "{values['segment_loss_reduction']}"

[checkpoint]
every_optimizer_steps = {values['checkpoint_steps']}
keep_last = {values['keep_last']}
save_at_pass_end = true
optimizer_boundary_only = true

[evaluation]
selection_metric = "{values['selection_metric']}"
selection_tiebreaker = "{values['selection_tiebreaker']}"

[logging]
every_optimizer_steps = {values['logging_steps']}
'''.replace('manifest = ', 'authoritative_manifest = ', 1)
    path = tmp_path / f"config-{len(list(tmp_path.glob('config-*')))}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_template_lists_all_unresolved_decisions() -> None:
    path = PROJECT_ROOT / "configs" / "p0.toml"
    config = load_runtime_config(path, allow_required=True)
    assert not config.is_resolved
    assert "dataset.root" in config.unresolved_fields
    assert "backbone.transformers_version" in config.unresolved_fields
    assert "optimizer.beta1" in config.unresolved_fields
    with pytest.raises(RequiredValueError, match="dataset.root"):
        config.require_resolved()
    with pytest.raises(RequiredValueError):
        load_runtime_config(path)


def test_resolved_config_loads_fixed_architecture(tmp_path: Path) -> None:
    config = load_runtime_config(_resolved_config(tmp_path))
    assert config.is_resolved
    assert config.spec["architecture_id"] == "WR-Agent-4B-Wildcard-v1"
    assert config.spec["backbone"]["temporal_rope"]["factorized_qk_dim_inference"] == 352


def test_forbidden_cluster_tree_is_rejected_before_access(tmp_path: Path) -> None:
    path = _resolved_config(
        tmp_path,
        dataset_root="/work/u10970284/airway ATM26 project/dataset",
    )
    with pytest.raises(ConfigError, match="forbidden work tree"):
        load_runtime_config(path, validate_paths=False)


def test_wrong_backend_fails_fast(tmp_path: Path) -> None:
    path = _resolved_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        'attention_backend = "FLEX_PAD512"',
        'attention_backend = "eager"',
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="FLEX_PAD512"):
        load_runtime_config(path)


def test_backend_is_bound_to_gpu_family(tmp_path: Path) -> None:
    # CUDA lane keeps the Astra 14/09 decision: SDPA memory-efficient on A40.
    cuda = load_runtime_config(_resolved_config(
        tmp_path, gpu_family="A40", platform="cuda", attention_backend="EFFICIENT_ATTENTION",
        attention_qk_pad_to="0"))
    assert cuda.raw["runtime"]["attention_backend"] == "EFFICIENT_ATTENTION"
    # A40 with the ROCm backend, or MI300X with the CUDA backend, is a lane mix-up.
    with pytest.raises(ConfigError, match="EFFICIENT_ATTENTION"):
        load_runtime_config(_resolved_config(
            tmp_path, gpu_family="A40", platform="cuda", attention_backend="FLEX_PAD512"))
    with pytest.raises(ConfigError, match="FLEX_PAD512"):
        load_runtime_config(_resolved_config(
            tmp_path, gpu_family="MI300X", platform="rocm", attention_backend="EFFICIENT_ATTENTION"))
    with pytest.raises(ConfigError, match="platform"):
        load_runtime_config(_resolved_config(
            tmp_path, gpu_family="MI300X", platform="cuda"))
    with pytest.raises(ConfigError, match="attention_qk_pad_to"):
        load_runtime_config(_resolved_config(
            tmp_path, gpu_family="MI300X", platform="rocm", attention_qk_pad_to="0"))
