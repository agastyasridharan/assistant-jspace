"""Frozen source-stimulus replication; run in the O-lens environment on Athena02."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from olens import config as C, data as D, llama as LM, oracle as O
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','capture','oracle']);ap.add_argument('--run',required=True);ap.add_argument('--inputs',required=True);args=ap.parse_args()
ROOT=Path(args.run);IN=Path(args.inputs);ROOT.mkdir(parents=True,exist_ok=True)
for d in ['assets','records','activations','oracle','checks']:(ROOT/d).mkdir(exist_ok=True)
M=json.loads((IN/'manifest.json').read_text());MAP=json.loads((IN/'qwen-sites.json').read_text());L=56
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False));q.replace(p)
def status(stage,**kw):save(ROOT/'status.json',dict(stage=stage,updated_at=time.time(),**kw));print(stage,kw,flush=True)
def gate():
 if (ROOT/'STOP').exists():status('paused',reason='STOP requested');sys.exit(0)
 reservation=json.loads((ROOT/'reservation.json').read_text())
 if time.time()>=reservation['expires_at']:status('paused',reason='Reservation expired');sys.exit(0)
def save_tensor(p,x):
 q=str(p)+'.tmp';torch.save(x,q);os.replace(q,p)
tok=D.tokenizer();stops=D.stop_ids(tok)
def prepare():
 enc=[]
 for r in M['records']:
  text=tok.apply_chat_template([{'role':'system','content':M['system']},{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True)
  e=tok(text,add_special_tokens=False,return_offsets_mapping=True);ids=e['input_ids'];offset=e['offset_mapping']
  if ids[0]!=tok.bos_token_id:ids=[tok.bos_token_id]+ids;offset=[(0,0)]+offset
  start=text.index(r['prompt'])+r['embedded_offset'];aa={}
  for name,span in r['anchors'].items():
   end=start+r['message'].index(span)+len(span);prefix=text[:end]
   pe=tok(prefix,add_special_tokens=False,return_offsets_mapping=True);pi=pe['input_ids'];po=pe['offset_mapping']
   if pi[0]!=tok.bos_token_id:pi=[tok.bos_token_id]+pi;po=[(0,0)]+po
   assert po[-1][1]==end,(r['id'],name,'prefix does not end at anchor')
   assert max(pi)<C.model_config()['vocab_size']
   i=len(pi)-1
   aa[name]=dict(position=i,token_id=pi[-1],token=tok.decode([pi[-1]]),char_end=end,prefix=prefix,prefix_input_ids=pi,full_prompt_prefix_identical=ids[:len(pi)]==pi,tokenization='exact-prefix')
  assert max(ids)<C.model_config()['vocab_size']
  enc.append(dict(id=r['id'],text=text,input_ids=ids,anchors=aa))
 assert len(enc)==52
 sites=[]
 for r in enc:
  for a,v in r['anchors'].items():
   if a=='pre_fact':continue
   can=next(k for k,w in r['anchors'].items() if w['position']==v['position'])
   key=r['id']+'__'+can
   if key not in [x['key'] for x in sites]:sites.append(dict(key=key,record_id=r['id'],anchor=can,position=v['position']))
 assert {s['key'] for s in sites}=={s['key'] for s in MAP}
 assert sum(s['sampled'] for s in MAP)==39
 lens=Path(C.ROOT)/'rl/final';assert lens.exists()
 files=[lens/'adapter.safetensors',lens/'meta.json',Path(C.model_dir())/'config.json',Path(C.FP8_DIR)/'model.safetensors.index.json']
 provenance={'model':C.MODEL,'model_path':C.model_dir(),'oracle_backbone':C.FP8_DIR,'adapter':str(lens),'tokenizer_path':C.tokenizer_dir(),'layer':L,'hashes':{str(p):digest(p) for p in files},'source_manifest_sha256':digest(IN/'manifest.json'),'code_sha256':digest(__file__),'native_code':{str(p):digest(p) for p in Path(LM.__file__).parent.glob('*.py')},'torch':torch.__version__}
 save(ROOT/'assets/tokenization.json',enc);save(ROOT/'assets/sites.json',sites);save(ROOT/'assets/provenance.json',provenance);save(ROOT/'checks/oracle_sites.json',MAP)
 save(ROOT/'checks/environment.json',provenance);status('prepared',records=52,anchors=sum(len(x['anchors']) for x in enc),sites=len(sites))
@torch.no_grad()
def generate(model,emb,max_new=256,temperature=0,top_p=.9):
 B,Lp,_=emb.shape;blk=model.blocks[0]
 kv=[(torch.empty(B,blk.Hkv,Lp+max_new,blk.D,device=d,dtype=model.dtype),torch.empty(B,blk.Hkv,Lp+max_new,blk.D,device=d,dtype=model.dtype)) for d in model.block_dev]
 h,_=model.forward_blocks(emb,kv=kv,p0=0);logits=model.logits(h[:,-1]);stop=torch.tensor(sorted(stops),device=logits.device);done=torch.zeros(B,dtype=torch.bool,device=logits.device);lens=torch.full((B,),max_new,dtype=torch.long,device=logits.device);out=[]
 for t in range(max_new):
  if temperature:
   probs=torch.softmax(logits.float()/temperature,-1);p,idx=probs.sort(descending=True);mask=p.cumsum(-1)-p>=top_p;p=p.masked_fill(mask,0);p=p/p.sum(-1,keepdim=True);nxt=idx.gather(-1,torch.multinomial(p,1))[:,0]
  else:nxt=logits.argmax(-1)
  nxt=torch.where(done,stop[0],nxt);out.append(nxt);hit=torch.isin(nxt,stop)&~done;lens=torch.where(hit,t+1,lens);done|=hit
  if bool(done.all()) or t==max_new-1:break
  h,_=model.forward_blocks(model.embed(nxt[:,None]),kv=kv,p0=Lp+t);logits=model.logits(h[:,-1])
 return torch.stack(out,1).cpu(),lens.cpu()
def verify_provenance():
 p=json.loads((ROOT/'assets/provenance.json').read_text())
 assert p['code_sha256']==digest(__file__),'Worker changed after freeze'
 assert p['model_path']==C.model_dir() and p['oracle_backbone']==C.FP8_DIR
 assert all(digest(k)==v for k,v in p['hashes'].items()),'Checkpoint changed'
 assert all(digest(k)==v for k,v in p['native_code'].items()),'Native code changed'
 return p
@torch.no_grad()
def capture():
 gate();p=verify_provenance();enc={r['id']:r for r in json.loads((ROOT/'assets/tokenization.json').read_text())}
 status('loading_subject');model=LM.load_model(p['model_path'],'dense',devices=['cuda:0','cuda:1'],head=True)
 def act(ids,positions):
  x=torch.tensor([ids],device=model.dev0);_,caps=model.forward_blocks(model.embed(x),capture=({L},torch.tensor(positions)));v=caps[L].detach().float();assert torch.isfinite(v).all();return v.cpu()
 canaries=['The capital of France is','The mitochondria is known as the powerhouse of the','For a classic margarita, combine tequila, lime juice and a splash of']
 for i,text in enumerate(canaries):
  gate();ids=tok.encode(text,add_special_tokens=False);save_tensor(ROOT/f'activations/canary-{i}.pt',act(ids,[-1])[0])
  if i==0:
   emb=model.embed(torch.tensor([ids],device=model.dev0));a,la=generate(model,emb,max_new=8);b,lb=LM.generate(model,emb,8,stops,0);assert torch.equal(a,b) and torch.equal(la,lb),'Greedy generator parity'
 for i in range(0,52,2):
  pair=M['records'][i:i+2];cp=ROOT/'checks'/(pair[0]['pair_id']+'.json')
  if not cp.exists():
   gate();aa,bb=[enc[r['id']]['input_ids'] for r in pair];common=next((j for j,(a,b) in enumerate(zip(aa,bb)) if a!=b),min(len(aa),len(bb)))
   a,b=act(aa,[common-1])[0],act(bb,[common-1])[0];rel=float((a-b).norm()/a.norm().clamp_min(1e-10));assert rel<.01,(pair[0]['pair_id'],rel)
   save(cp,{'common_tokens':common,'relative_l2':rel,'layer':L})
  for r in pair:
   gate();dest=ROOT/'records'/(r['id']+'.json')
   if dest.exists():continue
   e=enc[r['id']];names=list(e['anchors']);positions=[e['anchors'][n]['position'] for n in names];status('capture',record=r['id'],completed=len(list((ROOT/'records').glob('*.json'))))
   vectors={};rel=0.0
   for n in names:
    gate();ai=e['anchors'][n]['prefix_input_ids'];v=act(ai,[-1])[0];vectors[n]=v.to(torch.bfloat16)
   # Causal check under identical tokenization, with additional tokens after the anchor.
   ai=e['anchors'][names[-1]]['prefix_input_ids'];extension=tok.encode(' Additional context.',add_special_tokens=False)
   check=act(ai+extension,[len(ai)-1])[0];v=vectors[names[-1]].float();rel=float((v-check).norm()/v.norm().clamp_min(1e-10))
   assert rel<.01,('Exact-prefix causal check',r['id'],rel)
   save_tensor(ROOT/'activations'/(r['id']+'.pt'),vectors)
   toks,lens=generate(model,model.embed(torch.tensor([e['input_ids']],device=model.dev0)));ids=toks[0,:lens[0]].tolist();reply=tok.decode([x for x in ids if x not in stops],skip_special_tokens=False)
   save(dest,dict(id=r['id'],anchors=e['anchors'],jpp={},reply=reply,reply_token_ids=ids,reply_truncated=len(ids)==256 and ids[-1] not in stops,copy_exact=reply.strip()==r['message'].strip(),prefix_relative_l2=rel,completed_at=time.time()))
 status('capture_complete',records=52)
@torch.no_grad()
def oracle():
 gate();p=verify_provenance();status('loading_oracle');model=LM.load_model(p['oracle_backbone'],'fp8',devices=['cuda:2'],head=True);meta=json.loads((Path(p['adapter'])/'meta.json').read_text());model.load_lora('ao',load_file(str(Path(p['adapter'])/'adapter.safetensors')),meta['r'],meta['alpha'])
 ids,mp=D.ao_prompt(tok,L,10,8,D.marker_id(tok));save(ROOT/'checks/oracle_prompt.json',{'input_ids':ids,'text':tok.decode(ids),'marker':mp,'layer':L,'injection':'native normalized input embedding, alpha=16000'})
 def decode(vec,seed=0,sampling=False):
  torch.manual_seed(seed);x=torch.tensor([ids],device=model.dev0);emb=model.embed(x)
  if vec is not None:emb=O.inject(emb,torch.tensor([mp]),vec[None])
  with LM.adapter('ao'):tt,ll=generate(model,emb,temperature=.7 if sampling else 0,top_p=.9)
  tokens=tt[0,:ll[0]].tolist();finished=bool(tokens) and tokens[-1] in stops;body=tokens[:-1] if finished else tokens;raw=tok.decode(tokens,skip_special_tokens=False);text=tok.decode(body,skip_special_tokens=False)
  phrases=[line for line in text.split('\n') if line.strip()];lengths=[len(tok.encode(line,add_special_tokens=False)) for line in phrases]
  return dict(raw=raw,token_ids=tokens,phrases=phrases,phrase_lengths=lengths,format_valid=len(phrases)==10 and all(n==8 for n in lengths),truncated=not finished and len(tokens)>=256,sampling=sampling,seed=seed,completed_at=time.time())
 gate();prior=decode(None);can=[decode(torch.load(ROOT/f'activations/canary-{i}.pt',weights_only=True)) for i in range(3)];expected=[['paris'],['cell'],['triple sec','cointreau','orange']];hits=[any(w in x['raw'].lower() for w in ws) for x,ws in zip(can,expected)];save(ROOT/'checks/oracle_calibration.json',dict(results=can,expected=expected,hits=hits,no_injection=prior));assert sum(hits)>=2 and any(x['raw']!=prior['raw'] for x in can),('Oracle calibration failed',hits);save(ROOT/'oracle/no-injection.json',prior)
 sites=json.loads((ROOT/'assets/sites.json').read_text());sampled={s['key'] for s in MAP if s['sampled']}
 for sampling in [False,True]:
  for s in sites:
   if sampling and s['key'] not in sampled:continue
   vec=torch.load(ROOT/'activations'/(s['record_id']+'.pt'),weights_only=True)[s['anchor']]
   for seed in [1701,1702,1703,1704,1705] if sampling else [0]:
    gate();key=s['key']+(f'__seed{seed}' if sampling else '');dest=ROOT/'oracle'/(key+'.json')
    if dest.exists():continue
    status('oracle_samples' if sampling else 'oracle',site=key,completed=len(list((ROOT/'oracle').glob('*__seed*.json'))) if sampling else len([f for f in (ROOT/'oracle').glob('*.json') if '__seed' not in f.stem and f.stem!='no-injection']))
    save(dest,dict(**decode(vec,seed,sampling),**s))
 status('complete',records=52,oracle_sites=156,oracle_samples=195)
try:
 {'prepare':prepare,'capture':capture,'oracle':oracle}[args.stage]()
except Exception as e:
 import traceback
 status('failed',error=str(e));(ROOT/'failure.txt').write_text(traceback.format_exc());raise
