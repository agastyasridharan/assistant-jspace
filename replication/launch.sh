#!/usr/bin/env bash
# Run only after a current reservation.json names this model's three GPU IDs and expiry.
set -euo pipefail
MODEL_KEY=$1
RUN_ROOT=/data/agastyas/assistant-jspace-llama-20261016
RUN=$RUN_ROOT/$MODEL_KEY
cd /home/agastyas/olens_tulu70b
source env.sh
export PYTHONPATH=/home/agastyas/olens_tulu70b
case "$MODEL_KEY" in
 llama-base) export OLENS_MODEL=meta-llama/Llama-3.1-70B ;;
 tulu) export OLENS_MODEL=allenai/Llama-3.1-Tulu-3-70B ;;
 *) exit 2 ;;
esac
export OLENS_TOKENIZER=allenai/Llama-3.1-Tulu-3-70B
export CUDA_VISIBLE_DEVICES=$("$PY" - "$RUN/reservation.json" <<'PY'
import json,sys,time
r=json.load(open(sys.argv[1]));assert r['expires_at']>time.time(), 'Reservation expired'
assert len(r['gpus'])==3 and len(set(r['gpus']))==3
print(','.join(map(str,r['gpus'])))
PY
)
# Per-GPU locks persist for the entire worker. Check actual process occupancy as well.
IFS=, read -ra GS <<< "$CUDA_VISIBLE_DEVICES"
for i in 0 1 2; do
 gpu=${GS[$i]}; fd=$((200+i)); eval "exec ${fd}>/data/agastyas/assistant-jspace-gpu-${gpu}.lock"
 flock -n "$fd" || { echo "GPU $gpu is locked"; exit 3; }
 [ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU $gpu is occupied"; exit 3; }
done
"$PY" "$RUN_ROOT/run_llama.py" capture --run "$RUN" --inputs "$RUN_ROOT/inputs"
"$PY" "$RUN_ROOT/run_llama.py" oracle --run "$RUN" --inputs "$RUN_ROOT/inputs"
