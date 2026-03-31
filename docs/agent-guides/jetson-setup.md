# Jetson Setup Notes

Use this guide when the task is about direct Jetson runtime behavior rather
than the Windows development host.

## One-Time Setup

Prefer the repository helper on the Jetson device:

```bash
chmod +x scripts/setup_jetson.sh
chmod +x scripts/install_jetson_torch.sh
chmod +x scripts/jetson_env.sh
./scripts/setup_jetson.sh
```

`setup_jetson.sh` creates `.venv` with `--system-site-packages`, repairs a
CPU-only torch install when needed, installs project dependencies, and appends a
Jetson runtime hook to `.venv/bin/activate`.

## Normal Runtime

After the one-time setup, each fresh shell should only need:

```bash
source .venv/bin/activate
model-service
```

Prefer these commands for day-to-day work:

```bash
model-service
pytest services/model/tests -q
```

Acceptable fallback:

```bash
uv run --no-sync model-service
uv run --no-sync pytest services/model/tests -q
```

Avoid plain `uv run ...` once the environment is already prepared, because a
sync step can unexpectedly modify `.venv`.

## Runtime Bootstrap Rule

- `scripts/jetson_env.sh` restores CUDA, TensorRT, and Jetson-specific shared
  library paths in the current shell.
- `setup_jetson.sh` installs that script as an activation hook, so
  `source .venv/bin/activate` restores the paths automatically.
- `model-service` also bootstraps the same runtime paths before importing the
  FastAPI stack. This keeps service startup working even if the shell did not
  source the helper.

## Manual Recovery

If shell-level CUDA checks still fail after activation, recover in this order:

```bash
source .venv/bin/activate
source scripts/jetson_env.sh
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `torch.version.cuda` is `None`, reinstall the Jetson torch wheel:

```bash
./scripts/install_jetson_torch.sh
```

## Compatibility Rules

- Keep NumPy on `1.x`.
- `--system-site-packages` is required to inherit CUDA, TensorRT, and
  JetPack-provided PyTorch.
- Do not use `uv pip install -e ".[dev]"` on Jetson; it can pull CPU-only PyPI
  torch wheels into `.venv`.
