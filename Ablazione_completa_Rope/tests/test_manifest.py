from __future__ import annotations

from pathlib import Path

from wavelet_rope.config import load_runtime_config
from wavelet_rope.manifest import build_experiment_manifest

from test_config import _resolved_config


def test_fingerprint_ignores_output_and_logging_only(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "src" / "package").mkdir(parents=True)
    (project / "src" / "package" / "module.py").write_text("VALUE = 1\n")
    (project / "pyproject.toml").write_text("[project]\nname='test'\nversion='1'\n")

    first_path = _resolved_config(tmp_path)
    first = load_runtime_config(first_path)
    first_manifest = build_experiment_manifest(first, project_root=project)

    second_path = _resolved_config(
        tmp_path,
        run_root=(tmp_path / "other-runs").as_posix(),
        logging_steps="20",
    )
    second = load_runtime_config(second_path)
    second_manifest = build_experiment_manifest(second, project_root=project)

    assert first_manifest.experiment_fingerprint == second_manifest.experiment_fingerprint
    assert first_manifest.operational_config_sha256 != second_manifest.operational_config_sha256


def test_source_change_changes_experiment_fingerprint(tmp_path: Path) -> None:
    project = tmp_path / "project"
    module = project / "src" / "package" / "module.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n")
    (project / "pyproject.toml").write_text("[project]\nname='test'\nversion='1'\n")
    config = load_runtime_config(_resolved_config(tmp_path))
    before = build_experiment_manifest(config, project_root=project)
    module.write_text("VALUE = 2\n")
    after = build_experiment_manifest(config, project_root=project)
    assert before.experiment_fingerprint != after.experiment_fingerprint

