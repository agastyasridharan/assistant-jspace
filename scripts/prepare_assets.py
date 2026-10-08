"""CPU-only asset acquisition and tokenizer/anchor validation."""
import hashlib,json,os,sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

root=Path(__file__).resolve().parents[1]
m=json.loads((root/'data/manifest.json').read_text())
base=Path('/data/hf_cache')/('models--'+m['model'].replace('/','--'))/'snapshots'/m['model_revision']
tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
def tokenize(record):
    text=tok.apply_chat_template([dict(role='system',content=m['system']),dict(role='user',content=record['prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    start=text.index(record['prompt'])+record['embedded_offset']
    positions={}
    for name,span in record['anchors'].items():
        end=start+record['message'].index(span)+len(span)
        hits=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a<end<=b]
        assert len(hits)==1,(record['id'],name,end,hits)
        i=hits[0]; a,b=encoded['offset_mapping'][i]
        assert b==end,(record['id'],name,'anchor crosses a token boundary',text[a:b],text[end:b])
        positions[name]=dict(position=i,token_id=encoded['input_ids'][i],token=tok.decode([encoded['input_ids'][i]]),char_end=end,prefix=text[:end])
    user_positions=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a>=start and b<=start+len(record['message']) and b>a]
    return dict(id=record['id'],text=text,input_ids=encoded['input_ids'],offsets=encoded['offset_mapping'],anchors=positions,user_positions=user_positions)

(root/'assets').mkdir(exist_ok=True)
encoded=[tokenize(r) for r in m['records']]
audit={}
for f in m['families']:
    audit[f['id']]={}
    for name,words in zip(['user','assessment','response'],f['concepts']):
        accepted, rejected=[],[]
        for word in words:
            for spelling in [word,' '+word,word.capitalize(),' '+word.capitalize()]:
                ids=tok.encode(spelling,add_special_tokens=False)
                (accepted if len(ids)==1 else rejected).append(dict(text=spelling,ids=ids))
        audit[f['id']][name]=dict(ids=sorted({r['ids'][0] for r in accepted}),accepted=accepted,rejected=rejected)
        assert audit[f['id']][name]['ids'],(f['id'],name)
(root/'assets/tokenization.json').write_text(json.dumps(encoded,ensure_ascii=False,indent=2))
(root/'assets/vocabulary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print('TOKENIZATION_VALIDATED',len(encoded),sum(len(x['anchors']) for x in encoded),flush=True)
path=hf_hub_download(m['jpp_repo'],m['jpp_filename'],revision=m['jpp_revision'],cache_dir=str(root/'assets/hf'))
digest=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()
(root/'assets/paths.json').write_text(json.dumps(dict(model=str(base),lens=path,lens_sha256=digest,manifest_sha256=hashlib.sha256((root/'data/manifest.json').read_bytes()).hexdigest()),indent=2))
print('ASSETS_READY',digest,flush=True)
