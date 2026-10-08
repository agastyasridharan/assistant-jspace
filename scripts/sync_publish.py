"""One read-only remote sync followed by public study-data publication.

Run by the thread heartbeat while the study is active. No GPU jobs are started here.
"""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE='agastyas@athena02.ig32s.ruhr-uni-bochum.de'
RUN='/data/agastyas/assistant-jspace-20261008'
def call(args,**kw):return subprocess.run(args,cwd=ROOT,check=True,timeout=180,**kw)
try:
    (ROOT/'local/results').mkdir(parents=True,exist_ok=True)
    # Explicit allowlist: no checkpoints, keys, shell state, or unrelated research data.
    call(['rsync','-az','--include=*/','--include=*.json','--exclude=*','-e','ssh -o BatchMode=yes -o ConnectTimeout=12',f'{REMOTE}:{RUN}/results/',str(ROOT/'local/results/')+'/'])
    call([sys.executable,'scripts/export_dashboard.py'])
    call(['git','add','docs/data'])
    changed=subprocess.run(['git','diff','--cached','--quiet','--','docs/data'],cwd=ROOT).returncode
    if changed:
        call(['git','commit','-m','Update live study readouts'])
        call(['git','push','origin','HEAD'])
    statuspath=ROOT/'local/results/status.json'
    print(statuspath.read_text() if statuspath.exists() else 'No study worker has started.')
except Exception as exc:
    print('SYNC_FAILED',str(exc),file=sys.stderr);sys.exit(1)
