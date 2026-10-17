#!/usr/bin/env bash
# Resume a validated subject's checkpointed oracle stage on one GPU.
set -euo pipefail
KEY=$1
ROOT=/data/agastyas/assistant-jspace-llama-20261016-v2
RUN=$ROOT/$KEY
cd /home/agastyas/olens_tulu70b
source env.sh
export PYTHONPATH=/home/agastyas/olens_tulu70b
export OLENS_TOKENIZER=allenai/Llama-3.1-Tulu-3-70B
case "$KEY" in
 tulu) export OLENS_MODEL=allenai/Llama-3.1-Tulu-3-70B ;;
 llama-base) export OLENS_MODEL=meta-llama/Llama-3.1-70B ;;
 *) exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=$("$PY" - "$RUN" <<'PY'
import json,sys,time,pathlib
p=pathlib.Path(sys.argv[1]);r=json.load(open(p/'reservation.json'))
assert r['expires_at']>time.time() and len(r['gpus'])==1
assert len(list((p/'records').glob('*.json')))==52
c=json.load(open(p/'checks/causality_summary.json'));assert c['all_exact'] and c['anchors']==164
print(r['gpus'][0])
PY
)
exec 200>"/data/agastyas/assistant-jspace-gpu-${CUDA_VISIBLE_DEVICES}.lock"
flock -n 200
[ -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" ]
export OLENS_ORACLE_DEVICE=cuda:0
exec "$PY" "$ROOT/run_llama.py" oracle --run "$RUN" --inputs "$ROOT/inputs"
