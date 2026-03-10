#!/usr/bin/env bash

set -euo pipefail

PT_FILE="${PT_FILE:-siyeon_best.pt}"
IMGSZ="${IMGSZ:-480}"
MODELS_DIR="${MODELS_DIR:-/models}"
INPUT_PATH="${MODELS_DIR}/${PT_FILE}"
OUTPUT_PATH="${INPUT_PATH%.pt}.engine"

echo "=========================================="
echo "TensorRT engine export"
echo "=========================================="
echo "Input model : ${INPUT_PATH}"
echo "Image size  : ${IMGSZ}"
echo "Output file : ${OUTPUT_PATH}"
echo "=========================================="

if [[ ! -f "${INPUT_PATH}" ]]; then
    echo "ERROR: input model not found: ${INPUT_PATH}" >&2
    ls -la "${MODELS_DIR}" || true
    exit 1
fi

if ! command -v yolo >/dev/null 2>&1; then
    echo "ERROR: yolo CLI not found in PATH" >&2
    exit 1
fi

yolo export \
    model="${INPUT_PATH}" \
    format=engine \
    device=0 \
    half=True \
    imgsz="${IMGSZ}"

if [[ ! -f "${OUTPUT_PATH}" ]]; then
    echo "ERROR: export did not produce ${OUTPUT_PATH}" >&2
    exit 1
fi

echo "=========================================="
echo "Export complete"
echo "=========================================="
echo "Engine file : ${OUTPUT_PATH}"
du -h "${OUTPUT_PATH}"
