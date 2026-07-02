from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from model_service.core.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_LAUNCHER_SCRIPT = REPO_ROOT / "scripts" / "install_model_service_launcher.sh"
SETUP_JETSON_SCRIPT = REPO_ROOT / "scripts" / "setup_jetson.sh"


def _functional_bash() -> str | None:
    if os.name == "nt":
        return None
    bash_path = shutil.which("bash")
    if not bash_path:
        return None
    result = subprocess.run(
        [bash_path, "-lc", "printf ok"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return bash_path


def test_env_example_parses_as_freezer_template(monkeypatch):
    for key in (
        "MODEL__MACHINE__CABINET_TYPE",
        "MODEL__VISION__CAMERA_LAYOUT",
        "MODEL__VISION__YOLO_MODEL_PATH",
        "MODEL__VISION__HAND_CONFIDENCE_THRESHOLD",
        "MODEL__VISION__TOP_CONFIDENCE_THRESHOLD",
        "MODEL__VISION__SIDE_CONFIDENCE_THRESHOLD",
        "MODEL__WEIGHT__FREEZER_VISION_MULTI_WITHOUT_WEIGHT_ENABLED",
        "MODEL__WEIGHT__FREEZER_DISTINCT_MIXED_PREFERENCE_ENABLED",
        "MODEL__WEIGHT__FREEZER_DISTINCT_MIXED_MAX_EXTRA_RESIDUAL_GRAMS",
        "MODEL__WEIGHT__FREEZER_PRIOR_TRIGGER_DEDUPE_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=REPO_ROOT / ".env.example")

    assert settings.machine.cabinet_type == "freezer"
    assert settings.vision.camera_layout == "dual_top_proxy"
    assert settings.vision.yolo_model_path == "models/set9_imbalance_16.engine"
    assert settings.vision.hand_class_id == 0
    assert settings.vision.hand_confidence_threshold == 0.40
    assert settings.vision.top_confidence_threshold == 0.70
    assert settings.vision.side_confidence_threshold == 0.70
    assert settings.vision.top_weight == 0.60
    assert settings.vision.side_weight == 0.40
    assert settings.vision.top_only_weight == 0.60
    assert settings.vision.side_only_weight == 0.40
    assert settings.vision.freezer_min_vote_ratio == 0.08
    assert settings.vision.freezer_min_vote_count == 4
    assert settings.vision.freezer_motion_min_displacement_px == 12.0
    assert settings.vision.freezer_roi_vertical_region == "upper"
    assert settings.vision.freezer_roi_y_split == 240.0
    assert settings.weight.freezer_weight_tolerance_grams == 15.0
    assert settings.weight.freezer_vision_multi_without_weight_enabled is False
    assert settings.weight.freezer_distinct_mixed_preference_enabled is True
    assert settings.weight.freezer_distinct_mixed_max_extra_residual_grams == 5.0
    assert settings.weight.freezer_prior_trigger_dedupe_enabled is True
    assert settings.trace.sample_export_enabled is False


def test_vision_confidence_defaults_and_env_overrides(monkeypatch):
    for key in (
        "MODEL__VISION__HAND_CONFIDENCE_THRESHOLD",
        "MODEL__VISION__TOP_CONFIDENCE_THRESHOLD",
        "MODEL__VISION__SIDE_CONFIDENCE_THRESHOLD",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)

    assert settings.vision.hand_confidence_threshold == 0.40
    assert settings.vision.top_confidence_threshold == 0.70
    assert settings.vision.side_confidence_threshold == 0.70

    monkeypatch.setenv("MODEL__VISION__HAND_CONFIDENCE_THRESHOLD", "0.45")
    monkeypatch.setenv("MODEL__VISION__TOP_CONFIDENCE_THRESHOLD", "0.72")
    monkeypatch.setenv("MODEL__VISION__SIDE_CONFIDENCE_THRESHOLD", "0.73")

    settings = Settings(_env_file=None)

    assert settings.vision.hand_confidence_threshold == 0.45
    assert settings.vision.top_confidence_threshold == 0.72
    assert settings.vision.side_confidence_threshold == 0.73


def test_setup_shell_scripts_parse_with_bash():
    bash_path = _functional_bash()
    if not bash_path:
        pytest.skip("functional bash is not available")

    subprocess.run([bash_path, "-n", str(SETUP_JETSON_SCRIPT)], check=True)
    subprocess.run([bash_path, "-n", str(INSTALL_LAUNCHER_SCRIPT)], check=True)


def test_install_model_service_launcher_dry_run(tmp_path):
    bash_path = _functional_bash()
    if not bash_path:
        pytest.skip("functional bash is not available")

    project_root = tmp_path / "repo"
    venv_bin = project_root / ".venv" / "bin"
    launcher_bin = tmp_path / "home" / ".local" / "bin"
    profile_path = tmp_path / "home" / ".profile"
    ran_cwd = project_root / "ran.cwd"
    ran_args = project_root / "ran.args"

    venv_bin.mkdir(parents=True)
    (project_root / "scripts").mkdir()
    (venv_bin / "activate").write_text("export FAKE_VENV_ACTIVATED=1\n", encoding="utf-8")
    (venv_bin / "model-service").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'if [[ "${FAKE_VENV_ACTIVATED:-}" != "1" ]]; then exit 42; fi',
                f'printf "%s\\n" "$PWD" > {shlex.quote(str(ran_cwd))}',
                f'printf "%s\\n" "$@" > {shlex.quote(str(ran_args))}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (venv_bin / "model-service").chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "MODEL_SERVICE_PROJECT_ROOT": str(project_root),
            "MODEL_SERVICE_LAUNCHER_BIN_DIR": str(launcher_bin),
            "MODEL_SERVICE_PROFILE_PATH": str(profile_path),
            "PATH": "/usr/bin:/bin",
        }
    )

    subprocess.run([bash_path, str(INSTALL_LAUNCHER_SCRIPT)], check=True, env=env)
    launcher_path = launcher_bin / "model-service"
    assert launcher_path.exists()
    assert "model-service auto-venv launcher" in launcher_path.read_text(encoding="utf-8")
    assert "model-service user launcher path" in profile_path.read_text(encoding="utf-8")

    first_profile = profile_path.read_text(encoding="utf-8")
    subprocess.run([bash_path, str(INSTALL_LAUNCHER_SCRIPT)], check=True, env=env)
    assert profile_path.read_text(encoding="utf-8") == first_profile

    subprocess.run([str(launcher_path), "alpha", "two words"], check=True, env=env)
    assert ran_cwd.read_text(encoding="utf-8").strip() == str(project_root)
    assert ran_args.read_text(encoding="utf-8").splitlines() == ["alpha", "two words"]


def test_install_model_service_launcher_refuses_non_owned_launcher(tmp_path):
    bash_path = _functional_bash()
    if not bash_path:
        pytest.skip("functional bash is not available")

    project_root = tmp_path / "repo"
    launcher_bin = tmp_path / "home" / ".local" / "bin"
    launcher_bin.mkdir(parents=True)
    (launcher_bin / "model-service").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "MODEL_SERVICE_PROJECT_ROOT": str(project_root),
            "MODEL_SERVICE_LAUNCHER_BIN_DIR": str(launcher_bin),
            "MODEL_SERVICE_PROFILE_PATH": str(tmp_path / "home" / ".profile"),
        }
    )

    result = subprocess.run(
        [bash_path, str(INSTALL_LAUNCHER_SCRIPT)],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Refusing to overwrite existing non-model-service launcher" in result.stderr
