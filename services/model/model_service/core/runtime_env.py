from __future__ import annotations

"""Jetson runtime environment bootstrap helpers.

These helpers make the console entry point resilient in fresh shells where the
virtual environment is activated but CUDA/TensorRT library paths were not
restored yet.
"""

from pathlib import Path
import os
import platform
import site
import sys

JETSON_ENV_SENTINEL = "MODEL_SERVICE_JETSON_ENV_READY"


def is_jetson_environment() -> bool:
    """Return True when running on a Linux Jetson device."""
    return (
        sys.platform.startswith("linux")
        and platform.machine().lower() in {"aarch64", "arm64"}
        and Path("/etc/nv_tegra_release").exists()
    )


def _split_path_env(value: str | None) -> list[str]:
    """Split PATH-like variables while dropping empty entries."""
    if not value:
        return []
    return [entry for entry in value.split(os.pathsep) if entry]


def _merge_path_entries(preferred: list[Path], existing: str | None) -> str:
    """Prepend existing directories while preserving order and deduplicating."""
    merged: list[str] = []
    seen: set[str] = set()

    for path in preferred:
        resolved = str(path)
        if path.exists() and resolved not in seen:
            merged.append(resolved)
            seen.add(resolved)

    for entry in _split_path_env(existing):
        if entry not in seen:
            merged.append(entry)
            seen.add(entry)

    return os.pathsep.join(merged)


def _site_package_roots() -> list[Path]:
    """Collect site-package roots visible to the current interpreter."""
    roots: list[Path] = []
    seen: set[str] = set()

    for candidate in site.getsitepackages():
        path = Path(candidate)
        if str(path) not in seen:
            roots.append(path)
            seen.add(str(path))

    user_site = site.getusersitepackages()
    if user_site:
        path = Path(user_site)
        if str(path) not in seen:
            roots.append(path)
            seen.add(str(path))

    return roots


def _iter_candidate_bin_paths(venv_path: Path) -> list[Path]:
    """Return the executable search paths commonly needed on Jetson."""
    return [
        venv_path / "bin",
        Path.home() / ".local" / "bin",
        Path.home() / ".cargo" / "bin",
        Path("/usr/local/cuda/bin"),
    ]


def _iter_candidate_library_paths(venv_path: Path) -> list[Path]:
    """Return shared-library directories needed by CUDA, TensorRT, and wheels."""
    candidates: list[Path] = [
        venv_path / "lib",
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/cuda/compat"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/lib/aarch64-linux-gnu"),
        Path("/usr/lib/aarch64-linux-gnu/tegra"),
        Path("/usr/lib/aarch64-linux-gnu/nvidia"),
        Path("/usr/lib/aarch64-linux-gnu/nvidia/current"),
    ]

    for root in _site_package_roots():
        candidates.append(root / "tensorrt_libs")
        nvidia_dir = root / "nvidia"
        if nvidia_dir.exists():
            # Newer wheel layouts commonly expose libraries under
            # `site-packages/nvidia/<package>/lib`.
            candidates.extend(path for path in nvidia_dir.glob("*/*") if path.name == "lib")
            candidates.extend(path for path in nvidia_dir.glob("*/lib"))

    return candidates


def build_jetson_runtime_environment(
    base_env: dict[str, str] | None = None,
    *,
    venv_path: Path | None = None,
) -> dict[str, str]:
    """Build an environment with the common Jetson runtime paths restored."""
    env = dict(base_env or os.environ)
    resolved_venv = venv_path or Path(env.get("VIRTUAL_ENV", sys.prefix))

    env["PATH"] = _merge_path_entries(
        _iter_candidate_bin_paths(resolved_venv),
        env.get("PATH"),
    )
    env["LD_LIBRARY_PATH"] = _merge_path_entries(
        _iter_candidate_library_paths(resolved_venv),
        env.get("LD_LIBRARY_PATH"),
    )
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env.setdefault("CUDA_PATH", "/usr/local/cuda")
    env[JETSON_ENV_SENTINEL] = "1"
    return env


def bootstrap_runtime_environment(argv: list[str] | None = None) -> None:
    """Re-exec the process with Jetson runtime paths when needed."""
    if not is_jetson_environment():
        return

    if os.environ.get(JETSON_ENV_SENTINEL) == "1":
        return

    env = build_jetson_runtime_environment()
    # Re-enter through `python -m model_service` so both the console script and
    # direct module execution converge on the same startup path.
    args = [sys.executable, "-m", "model_service", *(argv or sys.argv[1:])]
    os.execvpe(sys.executable, args, env)
