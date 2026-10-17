#!/usr/bin/env bash
# One-shot setup on a Linux machine with an NVIDIA GPU (cloud VM, lab server).
#   git clone git@github.com:seregamikhailov/offroad-segmentation-1.git && cd offroad-segmentation-1
#   bash scripts/setup_remote.sh
# Optional environment variables:
#   PYTHON=python3.11        interpreter to use (default: python3)
#   CUDA_INDEX=https://download.pytorch.org/whl/cu126   install PyTorch from a specific CUDA wheel index
#                            (by default pip's own Linux wheels are used, which bundle CUDA)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
if [ -n "${CUDA_INDEX:-}" ]; then
  pip install torch torchvision --index-url "$CUDA_INDEX"
fi
pip install -e .

python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name, f"{p.total_memory / 1e9:.0f} GB")
else:
    raise SystemExit("CUDA is not available: check the NVIDIA driver / CUDA_INDEX")
EOF

python scripts/download_rugd_hf.py                 # ~5.7 GB into data/raw/extracted
python scripts/prepare_rugd.py                     # masks, split lists, classes.json
python scripts/check_sweep.py --sweep configs/sweeps/screening.yaml --device cuda   # also caches pretrained weights
echo "Setup finished. Start the experiments with: bash scripts/run_screening.sh"
