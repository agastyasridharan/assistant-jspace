"""Exhaustive broad User/Assistant reannotation; preserves the earlier analysis."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
import requests
from dotenv import dotenv_values
R=Path(__file__).resolve().parents[1]
O=R/'local/broad-perspectives';O.mkdir(parents=True,exist_ok=True)
source=R/'docs/data/analysis/readouts.json';items=json.loads(source.read_text())
prompt=(R/'analysis/broad-perspective-prompt.txt').read_text()
ph=hashlib.sha256(prompt.encode()).hexdigest()
key=dotenv_values('/Users/agastyasridharan/ruhr research/natural_language_autoencoders_original/.env')['OPENAI_API_KEY_2']
def obj(props):return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}
schema=obj({'annotations':{'type':'array','items':obj({'id':{'type':'string'},'phrases':{'type':'array','items':obj({'index':{'type':'integer'},'user':{'type':'boolean'},'assistant':{'type':'boolean'}})},'note':{'type':'string'}})}})
def batch(start):
 chunk=items[start:start+4];path=O/f'{ph[:12]}-{start:03}.json'
 if path.exists():return json.loads(path.read_text())
 inp=[{'id':x['id'],'prefix':x['prefix'],'phrases':[{'index':i+1,'text':p} for i,p in enumerate(x['phrases'])]} for x in chunk]
 for attempt in range(3):
  try:
   res=requests.post('https://api.openai.com/v1/responses',headers={'Authorization':'Bearer '+key},json={'model':'gpt-5.4','instructions':prompt,'input':json.dumps(inp,ensure_ascii=False),'text':{'format':{'type':'json_schema','name':'broad_perspectives','schema':schema,'strict':True}},'max_output_tokens':10000,'reasoning':{'effort':'medium'},'store':False},timeout=(10,180))
   if res.status_code!=200:raise RuntimeError('API status '+str(res.status_code))
   raw=res.json();txt=''.join(c.get('text','') for o in raw['output'] for c in o.get('content',[]) if c['type']=='output_text');xs=json.loads(txt)['annotations']
   assert len(xs)==len(chunk) and {x['id'] for x in xs}=={x['id'] for x in chunk}
   for x in xs:
    n=len(next(r for r in chunk if r['id']==x['id'])['phrases'])
    assert len(x['phrases'])==n and {p['index'] for p in x['phrases']}==set(range(1,n+1))
   result={'annotations':xs,'model':raw['model'],'usage':raw['usage']};path.write_text(json.dumps(result,ensure_ascii=False));print('Saved',start+len(chunk),'/',len(items),flush=True);return result
  except Exception as e:
   print('Retry batch',start,type(e).__name__,flush=True)
   if attempt==2:raise
   time.sleep(5*(attempt+1))
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:batches=list(pool.map(batch,range(0,len(items),4)))
ann={a['id']:a for b in batches for a in b['annotations']}
rows=[]
for r in items:
 a=ann[r['id']];rows.append({'source':r['source'],'phrases_sha256':hashlib.sha256(json.dumps(r['phrases'],ensure_ascii=False).encode()).hexdigest(),'annotation':{k:[p['index'] for p in a['phrases'] if p[k]] for k in ['user','assistant']},'note':a['note']})
(R/'docs/data/analysis/broad-perspectives.json').write_text(json.dumps(rows,ensure_ascii=False,separators=(',',':')))
(R/'analysis/broad-perspective-provenance.json').write_text(json.dumps({'model':sorted(set(b['model'] for b in batches)),'prompt_sha256':ph,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'readouts':len(rows),'phrases':sum(len(r['phrases']) for r in items),'created_at':time.time(),'status':'Automated exhaustive reannotation; qualitative spot checks, not full independent validation','prior_analysis':'Preserved unchanged; broader labels used in Pairs only'},indent=2))
print('COMPLETE',len(rows),flush=True)
