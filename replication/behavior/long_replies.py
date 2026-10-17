"""Exploratory longer base continuations; no changes to primary experiment."""
import ast,hashlib,json,time,sys,re
from pathlib import Path
import torch
from olens import llama as LM,data as D
root=Path('/data/agastyas/assistant-jspace-llama-20261016-v2')
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
p=json.load(open(root/'llama-base/assets/provenance.json'))
rows={r['id']:r for r in json.load(open(root/'llama-base/assets/tokenization.json'))}
manifest={r['id']:r for r in json.load(open(root/'inputs/manifest.json'))['records']}
tok=D.tokenizer();stops=D.stop_ids(tok)
# Reuse the exact study generation function, without executing its CLI.
source=(root/'run_llama.py').read_text();tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<frozen study generate>','exec'))
selected=[f'{f}-stated-{t}-0' for f in ['arithmetic','bereavement','frustration','tylenol'] for t in ['assist','copy']]
def save(path,x):
 q=path.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(path)
def status(stage,**kw):
 save(out/'status.json',dict(stage=stage,time=time.time(),**kw));print(stage,kw,flush=True)
save(out/'protocol.json',{'model':p['model'],'model_path':p['model_path'],'selected':selected,'selection':'Exploratory examples selected after examining existing replies; not a representative frequency estimate.','max_new_tokens':1024,'temperature':0,'stop_ids':sorted(stops),'stop_on_role_headers':False,'prompt_ids':'Identical to saved primary experiment','precision':'BF16','gpus':[5,6],'worker_sha256':hashlib.sha256(source.encode()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
try:
 status('loading')
 model=LM.load_model(p['model_path'],'dense',devices=['cuda:0','cuda:1'],head=True)
 for i,key in enumerate(selected):
  if (out/'STOP').exists():status('paused');sys.exit(0)
  if (out/(key+'.json')).exists():continue
  status('generating',record=key,completed=i,total=8)
  with torch.no_grad():
   ids=rows[key]['input_ids'];tokens,lengths=generate(model,model.embed(torch.tensor([ids],device=model.dev0)),max_new=1024)
  ids_out=tokens[0,:int(lengths[0])].tolist();old=json.load(open(root/'llama-base/records'/(key+'.json')))
  text=tok.decode([x for x in ids_out if x not in stops],skip_special_tokens=False)
  headers=[{'role':m.group(1),'char_position':m.start()} for m in re.finditer(r'<\|(system|user|assistant)\|>',text)]
  first=text[:headers[0]['char_position']].strip() if headers else text.strip()
  same=ids_out[:len(old['reply_token_ids'])]==old['reply_token_ids']
  save(out/(key+'.json'),{'id':key,'prompt':rows[key]['text'],'prompt_ids':rows[key]['input_ids'],'reply':text,'token_ids':ids_out,'tokens':len(ids_out),'truncated':len(ids_out)==1024 and ids_out[-1] not in stops,'role_headers':headers,'first_segment':first,'first_segment_exact_copy':first==manifest[key]['message'].strip(),'prefix_matches_original':same,'completed_at':time.time()})
  assert same,('Original 256-token prefix mismatch',key)
 status('complete',completed=8,total=8)
except Exception as e:
 status('failed',error=str(e));raise
