#!/usr/bin/env bash
set -euo pipefail
cd /data/agastyas/assistant-jspace-20261008
test -f reservation.json || { echo 'Reservation confirmation missing; refusing GPU launch.'; exit 2; }
source /data/agastyas/jlens_multi/env.sh
GPU_ID="${1:-0}"
exec 9>"/data/agastyas/user-style-gpu-${GPU_ID}.lock"
flock -n 9 || { echo 'GPU lock held; refusing launch.'; exit 3; }
GPU_UUID=$(nvidia-smi -i "$GPU_ID" --query-gpu=uuid --format=csv,noheader)
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$GPU_UUID"; then
  echo 'GPU has an attached process; refusing launch.'; exit 4
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
echo "$$" > launcher.pid
/data/agastyas/Miniconda3/envs/ml/bin/python scripts/run_study.py capture
test ! -f STOP || exit 0
/data/agastyas/Miniconda3/envs/ml/bin/python scripts/run_study.py oracle
