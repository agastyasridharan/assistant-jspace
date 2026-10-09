import json,hashlib,random,concurrent.futures,time
from pathlib import Path
import requests
from dotenv import dotenv_values
R=Path(__file__).resolve().parents[1];O=R/'local/analysis-v2';O.mkdir(exist_ok=True)
M=json.loads((R/'data/manifest.json').read_text()); recs={r['id']:r for r in M['records']}
axis={'tylenol':['excessive/dangerous dose','ordinary/nonexcessive dose'],'confidentiality':['sharing prohibited/no permission','sharing permitted/permission given'],'tutoring':['answer incorrect','answer correct'],'arithmetic':['answer incorrect','answer correct'],'tests':['test failure / patch not ready','tests passed / patch ready'],'scheduling':['late / timing infeasible','on time / timing feasible'],'booking':['reservation unconfirmed','reservation confirmed'],'blame':['assistant calculation incorrect','assistant calculation correct'],'bereavement':['father died','father retired / alive'],'frustration':['printer unresolved / suggestions unsuccessful','printer fixed / suggestions successful']}
items=[]
for p in sorted((R/'local/results/oracle').glob('*.json')):
 if p.stem=='no-injection':continue
 x=json.loads(p.read_text());r=recs[x['record_id']];record=json.loads((R/'local/results/records'/f"{r['id']}.json").read_text());a=record['anchors'][x['anchor']]
 opaque=hashlib.sha256(p.stem.encode()).hexdigest()[:16]
 items.append({'id':opaque,'source':p.stem,'record_id':r['id'],'anchor':x['anchor'],'sampling':x['sampling'],'seed':x['seed'],'family':r['family'],'task':r['task'],'attitude':r['attitude'],'side':r['side'],'prefix':a['prefix'],'phrases':x['phrases'],'axis_a':axis[r['family']][0],'axis_b':axis[r['family']][1],'truncated':x['truncated']})
random.Random(73129).shuffle(items)
(O/'items.json').write_text(json.dumps(items,ensure_ascii=False))
fields=['explicit_user','implicit_user','assistant_assessment_explicit','assistant_intention_explicit','assessment_unattributed','response_unattributed','other_person','factual','unsupported','state_a_evidence','state_b_evidence']
props={k:{'type':'array','items':{'type':'integer'}} for k in fields};props.update({'id':{'type':'string'},'contradiction_pairs':{'type':'array','items':{'type':'array','items':{'type':'integer'}}},'note':{'type':'string'}})
obj={'type':'object','properties':props,'required':list(props),'additionalProperties':False}
schema={'type':'object','properties':{'annotations':{'type':'array','items':obj}},'required':['annotations'],'additionalProperties':False}
prompt=(R/'analysis/annotator-prompt.txt').read_text();key=dotenv_values('/Users/agastyasridharan/ruhr research/natural_language_autoencoders_original/.env')['OPENAI_API_KEY_2']
(O/'provenance-last-launch.json').write_text(json.dumps({'model':'gpt-5.4','prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'protocol_sha256':hashlib.sha256((R/'analysis/protocol.json').read_bytes()).hexdigest(),'items':len(items),'started':time.time()},indent=2))
def batch(start):
 chunk=items[start:start+3];p=O/f'batch-{start:03}.json'
 if p.exists():return
 inp=[{k:x[k] for k in ['id','prefix','axis_a','axis_b']}|{'phrases':[{'index':j+1,'text':v} for j,v in enumerate(x['phrases'])]} for x in chunk]
 res=requests.post('https://api.openai.com/v1/responses',headers={'Authorization':'Bearer '+key},json={'model':'gpt-5.4','instructions':prompt,'input':json.dumps(inp,ensure_ascii=False),'text':{'format':{'type':'json_schema','name':'annotations','schema':schema,'strict':True}},'max_output_tokens':16000,'reasoning':{'effort':'medium'},'store':False},timeout=240)
 if res.status_code==429:
  time.sleep(float(res.headers.get('Retry-After','30')));raise ValueError('Rate limit; retry after waiting')
 if res.status_code!=200:raise RuntimeError(f'API status {res.status_code} at {start}')
 raw=res.json();txt=''.join(c.get('text','') for o in raw['output'] for c in o.get('content',[]) if c['type']=='output_text');xs=json.loads(txt)['annotations'];assert {x['id'] for x in xs}=={x['id'] for x in chunk}
 for x in xs:
  n=len(next(i for i in chunk if i['id']==x['id'])['phrases'])
  for f in fields:
   if not all(1<=j<=n for j in x[f]):
    (O/f'invalid-{start}.json').write_text(json.dumps(xs,ensure_ascii=False));raise ValueError(f'Invalid phrase index: {start} {f} n={n}')
 p.write_text(json.dumps({'annotations':xs,'model':raw['model'],'usage':raw['usage']},ensure_ascii=False));print(start,'done',flush=True)
def retry(i):
 for attempt in range(3):
  try:return batch(i)
  except ValueError:
   if attempt==2:raise
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:list(pool.map(retry,range(0,len(items),3)))
print('ALL COMPLETE',len(items),flush=True)
