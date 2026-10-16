"""Read-only remote synchronization and model-isolated dashboard export."""
import json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
SSH='ssh -o BatchMode=yes -o ConnectTimeout=15 -o HostKeyAlias=athena02.ig32s.ruhr-uni-bochum.de'
remote='agastyas@10.147.189.41:/data/agastyas/assistant-jspace-llama-20261016/'
for key in ['llama-base','tulu']:
 local=R/'local/llama-results'/key;local.mkdir(parents=True,exist_ok=True)
 subprocess.run(['rsync','-az','-e',SSH,'--include=*/','--include=*.json','--exclude=*',remote+key+'/',str(local)+'/'],check=True)
 manifest=R/'replication'/(key+'-manifest.json');m=json.loads(manifest.read_text());p=local/'assets/provenance.json'
 if p.exists():
  prov=json.loads(p.read_text());m['model_revision']=Path(prov['model_path']).name;m['oracle_revision']=next(v for k,v in prov['hashes'].items() if k.endswith('rl/final/adapter.safetensors'));m['oracle_adapter_sha256']=m['oracle_revision'];m['source_manifest_sha256']=prov['source_manifest_sha256'];m['exact_prefix_tokenization']=True;manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
 subprocess.run([sys.executable,str(R/'scripts/export_dashboard.py'),'--results',str(local),'--dest',str(R/'docs/data/models'/key),'--manifest',str(manifest)],check=True)
