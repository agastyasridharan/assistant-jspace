"""Isolated diagnostic; never modifies experiment records or validation thresholds."""
import json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from olens import llama as LM, data as D

root = Path(sys.argv[1])
out = Path(sys.argv[2])
p = json.loads((root/'assets/provenance.json').read_text())
rows = json.loads((root/'assets/tokenization.json').read_text())
r = next(x for x in rows if x['id']=='confidentiality-stated-copy-1')
a = list(r['anchors'].values())[-1]
ids = a['prefix_input_ids']
tok = D.tokenizer()
extra = tok.encode(' Additional context.', add_special_tokens=False)
alt = [tok.encode(' hello', add_special_tokens=False)[0]] * len(extra)
result = {'record':r['id'], 'anchor':list(r['anchors'])[-1], 'prefix_length':len(ids),
          'extension':extra, 'model':p['model'], 'torch':torch.__version__, 'tests':{}}
def report(name, value):
    result['tests'][name] = value
    out.write_text(json.dumps(result, indent=2))
    print(name, json.dumps(value), flush=True)
print('loading', p['model'], flush=True)
model = LM.load_model(p['model_path'], 'dense', devices=['cuda:0','cuda:1'], head=False)
layers = {1,8,16,28,40,56}
@torch.no_grad()
def act(seq):
    x = torch.tensor([seq], device=model.dev0)
    _, caps = model.forward_blocks(model.embed(x), end=56,
                                  capture=(layers,torch.tensor([len(ids)-1])))
    return {k:v[0].float().cpu() for k,v in caps.items()}
def diff(x,y):
    return {str(k):{'relative_l2':float((x[k]-y[k]).norm()/x[k].norm()),
                    'equal':torch.equal(x[k],y[k]),
                    'cosine':float(F.cosine_similarity(x[k],y[k],dim=0))} for k in sorted(x)}
b = act(ids)
report('identical_input_repeat',diff(b,act(ids)))
e = act(ids+extra)
report('default_append',diff(b,e))
report('same_length_future_substitution',diff(e,act(ids+alt)))
original_sdpa = LM.sdpa
def math_attention(q,k,v,causal):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q.float(),k.float(),v.float(),
            is_causal=causal,enable_gqa=True).to(q.dtype)
LM.sdpa = math_attention
bm = act(ids)
report('fp32_math_attention_append',diff(bm,act(ids+extra)))
report('attention_backend_difference',diff(b,bm))
original_linear = LM.Lin.forward
fixed_rows = len(ids)+len(extra)
def padded_linear(self,x):
    shape=x.shape
    flat=x.reshape(-1,shape[-1])
    assert self.mode=='dense' and LM._ACTIVE[0] is None
    padded=F.pad(flat,(0,0,0,fixed_rows-flat.shape[0]))
    return F.linear(padded,self.w)[:flat.shape[0]].view(*shape[:-1],self.fout)
LM.Lin.forward = padded_linear
bp = act(ids)
report('fixed_linear_shapes_math_attention_append',diff(bp,act(ids+extra)))
LM.sdpa = original_sdpa
bp = act(ids)
report('fixed_linear_shapes_default_attention_append',diff(bp,act(ids+extra)))
LM.Lin.forward = original_linear
result['completed_at']=time.time()
out.write_text(json.dumps(result,indent=2))
print('complete',flush=True)
