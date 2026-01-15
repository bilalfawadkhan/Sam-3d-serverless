#!/usr/bin/env bash
set -euo pipefail

# Activate conda env
source /opt/conda/etc/profile.d/conda.sh
conda activate sam3d-objects

TAG="${SAM3D_TAG:-hf}"
CFG="${SAM3D_CONFIG:-checkpoints/${TAG}/pipeline.yaml}"
HF_REPO="${HF_REPO:-facebook/sam-3d-objects}"

echo "[start] TAG=${TAG}"
echo "[start] CONFIG=${CFG}"
echo "[start] HF_REPO=${HF_REPO}"

# Download checkpoints if missing
if [[ ! -f "${CFG}" ]]; then
  echo "[start] Checkpoints not found at ${CFG}. Downloading from Hugging Face..."

  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[start] ERROR: HF_TOKEN is not set. Add it as a RunPod Serverless secret env var."
    exit 1
  fi

  # Non-interactive login
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null 2>&1 || true

  rm -rf "checkpoints/${TAG}-download"
  mkdir -p "checkpoints/${TAG}-download"

  huggingface-cli download \
    --repo-type model \
    --local-dir "checkpoints/${TAG}-download" \
    --max-workers 1 \
    "${HF_REPO}"

  # Many repos place files into a "checkpoints/" folder. We normalize to checkpoints/${TAG}
  if [[ -d "checkpoints/${TAG}-download/checkpoints" ]]; then
    rm -rf "checkpoints/${TAG}"
    mv "checkpoints/${TAG}-download/checkpoints" "checkpoints/${TAG}"
    rm -rf "checkpoints/${TAG}-download"
  else
    echo "[start] ERROR: unexpected download structure. Directory listing:"
    ls -la "checkpoints/${TAG}-download" || true
    echo "[start] Tip: set HF_REPO to the correct model repo or adjust the move logic."
    exit 1
  fi

  echo "[start] Checkpoints ready at checkpoints/${TAG}"
else
  echo "[start] Found checkpoints at ${CFG}"
fi

# Print basic GPU info (useful in logs)
python - << 'PY'
import torch
print("[start] torch:", torch.__version__)
print("[start] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[start] device:", torch.cuda.get_device_name(0))
PY

# Start RunPod serverless worker (handler)
exec python -u /workspace/handler.py
