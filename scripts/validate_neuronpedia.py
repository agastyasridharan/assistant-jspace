"""Read-only external replication with exact stored token IDs; never logs credentials."""
import json, os, time, argparse
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=1); args=p.parse_args()
out=ROOT/'local/neuronpedia-validation'; out.mkdir(exist_ok=True)
key=os.environ.get('NEURONPEDIA_API_KEY')
if not key:
    for line in Path('/Users/agastyasridharan/ruhr research/natural_language_autoencoders_original/.env').read_text().splitlines():
        if line.strip().startswith('NEURONPEDIA_API_KEY='):
            key=line.strip().split('=',1)[1].strip().strip('\"\''); break
if not key: raise SystemExit('Missing API key')
s=requests.Session(); s.headers.update({'x-api-key':key})
for item in json.loads((ROOT/'local/tokenization.json').read_text())[:args.limit]:
    dest=out/(item['id']+'.json')
    if dest.exists(): continue
    payload={'modelId':'qwen3.6-27b','type':['JPP_LENS'],'topN':8,'temperature':0,'numCompletionTokens':0,'prependBos':False,'enableThinking':False,'inputTokenIds':item['input_ids'],'filterNonWordTokens':False,'stream':False}
    start=time.time()
    r=s.post('https://www.neuronpedia.org/api/lens/prompt',json=payload,timeout=240,allow_redirects=False)
    print(item['id'],r.status_code,round(time.time()-start,2),flush=True)
    if r.status_code!=200:
        (out/'error.json').write_text(json.dumps({'status':r.status_code,'body':r.text[:4000]})); raise SystemExit('API request failed; details saved locally')
    result=r.json()
    dest.write_text(json.dumps({'request':payload,'response':result,'elapsed_seconds':time.time()-start,'fetched_at':time.time()}))
    print('meta',result.get('meta'),flush=True)
