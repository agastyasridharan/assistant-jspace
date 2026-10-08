"""Resumable, fail-closed pilot. CUDA devices are assigned only by the launcher.

J++ transport follows safety-research/jpp_lens at the pinned revision:
row residual @ J.T, followed by the model's final norm and output embedding.
Oracle prompt and injection use the checkpoint's own released utilities.
"""
import argparse, gc, hashlib, json, os, platform, sys, time, traceback
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
M=json.loads((ROOT/'data/manifest.json').read_text())
PATHS=json.loads((ROOT/'assets/paths.json').read_text())
ENC={x['id']:x for x in json.loads((ROOT/'assets/tokenization.json').read_text())}
VOC=json.loads((ROOT/'assets/vocabulary.json').read_text())
OUT=ROOT/'results'
for folder in ['records','activations','oracle','checks','trajectory']:
    (OUT/folder).mkdir(exist_ok=True,parents=True)
OROOT=Path('/data/hf_cache')/('models--'+M['oracle_repo'].replace('/','--'))/'snapshots'/M['oracle_revision']
sys.path.insert(0,str(OROOT/'code'))
from oracle_lens.common import get_decoder_layers, build_oracle_prompt, ResidualInjector
from oracle_lens.oracle_data import parse_generated_phrases

def atomic(path, obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False)); tmp.replace(path)

def status(stage, **kw):
    obj=dict(stage=stage,updated_at=time.time(),pid=os.getpid(),host=platform.node(),**kw)
    atomic(OUT/'status.json',obj); print(json.dumps(obj),flush=True)

def check_stop():
    if (ROOT/'STOP').exists():
        status('paused',reason='STOP file present'); raise SystemExit(0)

def model_load(path):
    return AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa',local_files_only=True).eval()

def final_norm(model):
    for chain in [('model','language_model','norm'),('model','norm'),('language_model','model','norm')]:
        cur=model
        try:
            for attr in chain: cur=getattr(cur,attr)
            return cur
        except AttributeError: pass
    raise RuntimeError('Cannot identify final normalization')

def activation_capture(model,ids,layers,verify=False):
    saved={}; handles=[]; blocks=get_decoder_layers(model)
    def hook(layer):
        def capture(mod,args,out):
            h=out[0] if isinstance(out,tuple) else out
            saved[layer]=h.detach().clone()
        return capture
    for layer in sorted(set(layers)|{44,63}): handles.append(blocks[layer].register_forward_hook(hook(layer)))
    pre={}
    def prehook(mod,args,kwargs):
        h=args[0] if args else kwargs['hidden_states']; pre['h']=h.detach().clone()
    handles.append(blocks[45].register_forward_pre_hook(prehook,with_kwargs=True))
    x=torch.tensor([ids],device='cuda')
    try:
        with torch.inference_mode(): out=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1)
        assert torch.equal(saved[44],pre['h']),'Block 44 output != block 45 input'
        assert all(torch.isfinite(v).all().item() for v in saved.values()),'Nonfinite activations'
        if verify:
            norm=final_norm(model)
            pred=model.get_output_embeddings()(norm(saved[63][:,-1:,:]))
            assert torch.allclose(pred.float(),out.logits.float(),atol=0.01,rtol=0.001),'Final norm/head parity failed'
        return saved
    finally:
        for handle in handles: handle.remove()

def pair_invariant(model,records):
    a,b=[ENC[r['id']] for r in records]
    common=0
    for x,y in zip(a['input_ids'],b['input_ids']):
        if x!=y:break
        common+=1
    assert common>0
    aa=activation_capture(model,a['input_ids'],M['layers'])
    bb=activation_capture(model,b['input_ids'],M['layers'])
    metrics={}
    for layer in M['layers']:
        x,y=aa[layer][0,common-1].float(),bb[layer][0,common-1].float()
        rel=((x-y).norm()/x.norm().clamp_min(1e-10)).item()
        metrics[str(layer)]=dict(relative_l2=rel,max_abs=(x-y).abs().max().item())
        assert rel<0.01,('Causal-prefix check failed',records[0]['pair_id'],layer,rel)
    return dict(common_tokens=common,metrics=metrics)

def capture():
    status('loading_subject')
    tok=AutoTokenizer.from_pretrained(PATHS['model'],local_files_only=True)
    model=model_load(PATHS['model'])
    torch.set_float32_matmul_precision('highest')
    ckpt=torch.load(PATHS['lens'],map_location='cpu',weights_only=True,mmap=True)
    assert ckpt['config']['d_model']==5120
    jac={l:ckpt['parameters']['jacobians'][l].float().cuda() for l in M['layers']}
    norm=final_norm(model); head=model.get_output_embeddings().weight.detach().float()
    vocab_size=head.shape[0]
    semantic=torch.tensor([any(c.isalnum() for c in tok.decode([i])) and i not in tok.all_special_ids for i in range(vocab_size)],device='cuda')
    atomic(OUT/'checks/environment.json',dict(torch=torch.__version__,transformers=transformers.__version__,python=sys.version,
        model_type=type(model).__name__,gpu=torch.cuda.get_device_name(),lens_config=ckpt['config'],manifest_sha256=PATHS['manifest_sha256'],lens_sha256=PATHS['lens_sha256']))
    # Calibrate on fixed neutral prompts, with no study-dependent marker selection.
    canaries=['The capital of France is','The mitochondria is known as the powerhouse of the','For a classic margarita, combine tequila, lime juice and a splash of']
    for ci,text in enumerate(canaries):
        ids=tok.encode(text,add_special_tokens=False)
        acts=activation_capture(model,ids,[44],verify=True)
        torch.save(acts[44][0,-1].cpu(),OUT/f'activations/canary-{ci}.pt')
    # Check every matched pair before any results from that pair are accepted.
    for idx in range(0,len(M['records']),2):
        check_stop()
        pair=M['records'][idx:idx+2]
        checkfile=OUT/'checks'/f"{pair[0]['pair_id']}.json"
        if not checkfile.exists():
            atomic(checkfile,pair_invariant(model,pair))
        for rec in pair:
            check_stop(); dest=OUT/'records'/f"{rec['id']}.json"
            if dest.exists():continue
            enc=ENC[rec['id']]; status('capture',record=rec['id'],completed=len(list((OUT/'records').glob('*.json'))),total=52)
            acts=activation_capture(model,enc['input_ids'],M['layers'],verify=True)
            pos=enc['user_positions']; anchor_positions={a['position'] for a in enc['anchors'].values()}
            assert anchor_positions.issubset(set(pos))
            group=VOC[rec['family']]
            concept_ids=sorted({i for g in group.values() for i in g['ids']})
            trajectories={}; anchor_reads={}
            prefix_ids={p:set(enc['input_ids'][:p+1]) for p in anchor_positions}
            with torch.inference_mode():
                for layer in M['layers']:
                    rows=[]
                    for first in range(0,len(pos),24):
                        batchpos=pos[first:first+24]
                        h=acts[layer][0,batchpos].float() @ jac[layer].T
                        logits=F.linear(norm(h),head).float()
                        values,ids=logits.topk(50,dim=-1)
                        fv,fi=logits.masked_fill(~semantic[None],float('-inf')).topk(50,dim=-1)
                        for j,p in enumerate(batchpos):
                            raw=[dict(id=i,token=tok.decode([i]),logit=v) for i,v in zip(ids[j].tolist(),values[j].tolist())]
                            filtered=[dict(id=i,token=tok.decode([i]),logit=v) for i,v in zip(fi[j].tolist(),fv[j].tolist())]
                            row=dict(position=p,token_id=enc['input_ids'][p],token=tok.decode([enc['input_ids'][p]]),raw=raw,filtered=filtered)
                            if p in anchor_positions:
                                cand=[]
                                for i in concept_ids:
                                    v=logits[j,i].item()
                                    cand.append(dict(id=i,token=tok.decode([i]),logit=v,rank=int((logits[j]>v).sum().item()+1),lexical_echo=i in prefix_ids[p]))
                                row['concepts']=cand
                                row['scores']={name:logits[j,g['ids']].mean().item() for name,g in group.items()}
                                anchor_reads.setdefault(str(p),{})[str(layer)]=row
                            rows.append(row)
                    trajectories[str(layer)]=rows
            # Save raw workspace vectors before generation. Capture hooks have been removed.
            vectors={name:acts[44][0,a['position']].cpu() for name,a in enc['anchors'].items()}
            torch.save(vectors,OUT/'activations'/f"{rec['id']}.pt")
            atomic(OUT/'trajectory'/f"{rec['id']}.json",trajectories)
            del acts,trajectories
            x=torch.tensor([enc['input_ids']],device='cuda')
            with torch.inference_mode():
                seq=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=256,do_sample=False,
                    temperature=None,top_p=None,top_k=None,pad_token_id=tok.eos_token_id,use_cache=True)
            generated=seq[0,len(enc['input_ids']):].tolist()
            reply=tok.decode(generated,skip_special_tokens=True)
            eos=model.generation_config.eos_token_id
            stop_ids=set(eos if isinstance(eos,list) else [eos])|{tok.eos_token_id}
            truncated=len(generated)>=256 and generated[-1] not in stop_ids
            atomic(dest,dict(id=rec['id'],anchors=enc['anchors'],jpp=anchor_reads,reply=reply,reply_token_ids=generated,
                reply_truncated=truncated,copy_exact=(reply==rec['message']) if rec['task']=='copy' else None,
                completed_at=time.time(),token_count=len(enc['input_ids']),manifest_sha256=PATHS['manifest_sha256']))
    status('capture_complete',completed=52,total=52)

def oracle_decode():
    assert len(list((OUT/'records').glob('*.json')))==52,'Capture incomplete'
    status('loading_oracle')
    tok=AutoTokenizer.from_pretrained(PATHS['model'],local_files_only=True)
    model=model_load(OROOT/'oracle_lens_model')
    ids,marker=build_oracle_prompt(tok,M['oracle_n'],M['oracle_k'],M['oracle_placeholder'])
    assert len(tok.encode(M['oracle_placeholder'],add_special_tokens=False))==1
    assert ids.count(tok.convert_tokens_to_ids(M['oracle_placeholder']))==1
    atomic(OUT/'checks/oracle_prompt.json',dict(input_ids=ids,text=tok.decode(ids),marker=marker,placeholder=M['oracle_placeholder'],note=M['oracle_placeholder_note']))
    injector=ResidualInjector(model,45)
    @torch.inference_mode()
    def decode(vectors,sample=False,seed=0):
        torch.manual_seed(seed)
        n=len(vectors); x=torch.tensor([ids]*n,device='cuda')
        if vectors[0] is None:
            injector.set(None,None)
        else:
            injector.set(torch.tensor([marker]*n,device='cuda'),torch.stack(vectors).cuda())
        try:
            seq=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=256,do_sample=sample,
                temperature=0.7 if sample else None,top_p=0.9 if sample else None,top_k=0 if sample else None,
                pad_token_id=tok.eos_token_id,use_cache=True)
        finally:injector.set(None,None)
        results=[]
        stop={tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')}
        for g in seq[:,len(ids):].tolist():
            # Remove only batch padding after first EOS; preserve all actual generated text and token IDs.
            end=next((i+1 for i,t in enumerate(g) if t in stop),len(g)); g=g[:end]
            phrases=parse_generated_phrases(g,tok,10)
            raw=tok.decode(g,skip_special_tokens=False)
            lengths=[len(p) for p in phrases]
            results.append(dict(raw=raw,token_ids=g,phrases=[tok.decode(p) for p in phrases],phrase_lengths=lengths,
                format_valid=len(phrases)==10 and all(n==8 for n in lengths),truncated=len(g)>=256 and g[-1] not in stop,
                sampling=sample,seed=seed,completed_at=time.time()))
        return results
    # Canary words are fixed before any study decode. At least 2/3 must recover expected topic.
    vectors=[torch.load(OUT/f'activations/canary-{i}.pt',weights_only=True) for i in range(3)]
    canary_results=decode(vectors)
    expected=[['paris'],['cell'],['triple sec','cointreau','orange']]
    hits=[any(word in o['raw'].lower() for word in words) for o,words in zip(canary_results,expected)]
    prior=decode([None])[0]
    atomic(OUT/'checks/oracle_calibration.json',dict(results=canary_results,expected=expected,hits=hits,no_injection=prior))
    assert sum(hits)>=2,('Oracle calibration failed',hits)
    assert any(o['raw']!=prior['raw'] for o in canary_results),'Oracle insensitive to injected activations'
    atomic(OUT/'oracle/no-injection.json',prior)
    sites=[]
    for r in M['records']:
        vecs=torch.load(OUT/'activations'/f"{r['id']}.pt",weights_only=True)
        for anchor,a in ENC[r['id']]['anchors'].items():
            if anchor=='pre_fact':continue
            # Duplicate-position labels point to the same readout; only one decode is needed.
            canonical=next(k for k,v in ENC[r['id']]['anchors'].items() if v['position']==a['position'])
            key=f"{r['id']}__{canonical}"
            if any(s['key']==key for s in sites):continue
            sites.append(dict(key=key,record_id=r['id'],anchor=canonical,position=a['position'],vector=vecs[anchor]))
    count=(len(sites)+3)//4
    sampled=set(s['key'] for s in sorted(sites,key=lambda s:hashlib.sha256(s['key'].encode()).hexdigest())[:count])
    # All oracle inputs are identical except injected vectors, so paired/unrelated controls are exact references.
    control_map=[]
    for s in sites:
        rec=next(r for r in M['records'] if r['id']==s['record_id'])
        paired=next(r for r in M['records'] if r['pair_id']==rec['pair_id'] and r['side']!=rec['side'])
        pairsite=next(x for x in sites if x['record_id']==paired['id'] and x['anchor']==s['anchor'])
        unrelated=next(x for x in sites if x['anchor']==s['anchor'] and next(r for r in M['records'] if r['id']==x['record_id'])['family']!=rec['family']) if s['anchor'] not in ['relief'] else next(x for x in sites if x['anchor']=='primary' and 'arithmetic' in x['key'])
        control_map.append(dict(key=s['key'],paired=pairsite['key'],unrelated=unrelated['key'],no_injection='no-injection',sampled=s['key'] in sampled))
    atomic(OUT/'checks/oracle_sites.json',control_map)
    for first in range(0,len(sites),8):
        check_stop()
        batch=[s for s in sites[first:first+8] if not (OUT/'oracle'/f"{s['key']}.json").exists()]
        if not batch:continue
        status('oracle',completed=sum((OUT/'oracle'/f"{s['key']}.json").exists() for s in sites),total=len(sites),sample_total=count*5)
        outputs=decode([s['vector'] for s in batch])
        for s,o in zip(batch,outputs):atomic(OUT/'oracle'/f"{s['key']}.json",dict(**o,key=s['key'],record_id=s['record_id'],anchor=s['anchor'],position=s['position']))
    # Sample individually to make each seeded result independent of batch assignment/resume boundaries.
    for s in sites:
        if s['key'] not in sampled:continue
        for seed in M['sample_seeds']:
            check_stop(); dest=OUT/'oracle'/f"{s['key']}__seed{seed}.json"
            if dest.exists():continue
            status('oracle_samples',site=s['key'],seed=seed,completed=len(list((OUT/'oracle').glob('*__seed*.json'))),total=count*5)
            o=decode([s['vector']],True,seed)[0]
            atomic(dest,dict(**o,key=s['key'],record_id=s['record_id'],anchor=s['anchor'],position=s['position']))
    status('complete',records=52,oracle_sites=len(sites),oracle_samples=count*5)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['capture','oracle']);args=parser.parse_args()
    try:
        assert hashlib.sha256((ROOT/'data/manifest.json').read_bytes()).hexdigest()==PATHS['manifest_sha256'],'Manifest changed since preflight'
        check_stop()
        capture() if args.stage=='capture' else oracle_decode()
    except Exception as e:
        status('failed',error=str(e),traceback=traceback.format_exc());raise
