from __future__ import annotations

import os

import pytest
from model_service.core import runtime_env


def test_build_jetson_runtime_environment_prepends_paths(tmp_path, monkeypatch):
    venv_path = tmp_path / ".venv"
    venv_bin = venv_path / "bin"
    venv_lib = venv_path / "lib"
    cuda_bin = tmp_path / "cuda" / "bin"
    cuda_lib = tmp_path / "cuda" / "lib64"

    for path in (venv_bin, venv_lib, cuda_bin, cuda_lib):
        path.mkdir(parents=True)

    monkeypatch.setattr(
        runtime_env,
        "_iter_candidate_bin_paths",
        lambda resolved_venv: [venv_bin, cuda_bin],
    )
    monkeypatch.setattr(
        runtime_env,
        "_iter_candidate_library_paths",
        lambda resolved_venv: [venv_lib, cuda_lib],
    )

    env = runtime_env.build_jetson_runtime_environment(
        {
            "PATH": "/existing/bin",
            "LD_LIBRARY_PATH": "/existing/lib",
        },
        venv_path=venv_path,
    )

    assert env[runtime_env.JETSON_ENV_SENTINEL] == "1"
    assert env["CUDA_HOME"] == "/usr/local/cuda"
    assert env["CUDA_PATH"] == "/usr/local/cuda"
    assert env["PATH"].split(os.pathsep)[:2] == [str(venv_bin), str(cuda_bin)]
    assert env["LD_LIBRARY_PATH"].split(os.pathsep)[:2] == [str(venv_lib), str(cuda_lib)]
    assert "/existing/bin" in env["PATH"].split(os.pathsep)
    assert "/existing/lib" in env["LD_LIBRARY_PATH"].split(os.pathsep)


def test_bootstrap_runtime_environment_reexecs_on_jetson(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime_env, "is_jetson_environment", lambda: True)
    monkeypatch.setattr(
        runtime_env,
        "build_jetson_runtime_environment",
        lambda base_env=None, venv_path=None: {
            runtime_env.JETSON_ENV_SENTINEL: "1",
            "PATH": "/prepared/bin",
        },
    )
    monkeypatch.setattr(runtime_env.sys, "executable", "/venv/bin/python")

    def fake_execvpe(file: str, args: list[str], env: dict[str, str]) -> None:
        captured["file"] = file
        captured["args"] = args
        captured["env"] = env
        raise RuntimeError("reexec")

    monkeypatch.delenv(runtime_env.JETSON_ENV_SENTINEL, raising=False)
    monkeypatch.setattr(runtime_env.os, "execvpe", fake_execvpe)

    with pytest.raises(RuntimeError, match="reexec"):
        runtime_env.bootstrap_runtime_environment(["--port", "8002"])

    assert captured["file"] == "/venv/bin/python"
    assert captured["args"] == [
        "/venv/bin/python",
        "-m",
        "model_service",
        "--port",
        "8002",
    ]
    assert captured["env"] == {
        runtime_env.JETSON_ENV_SENTINEL: "1",
        "PATH": "/prepared/bin",
    }


def test_bootstrap_runtime_environment_skips_when_already_bootstrapped(monkeypatch):
    monkeypatch.setattr(runtime_env, "is_jetson_environment", lambda: True)
    monkeypatch.setenv(runtime_env.JETSON_ENV_SENTINEL, "1")

    def fail_exec(*args, **kwargs):
        raise AssertionError("execvpe should not be called")

    monkeypatch.setattr(runtime_env.os, "execvpe", fail_exec)

    runtime_env.bootstrap_runtime_environment(["--host", "0.0.0.0"])
