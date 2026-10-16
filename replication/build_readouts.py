"""Build prefix-only annotation inputs from one completed replica, preserving every output."""
import argparse,hashlib,json,random
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('model',choices=['llama-base','tulu']);args=ap.parse_args();root=Path(__file__).resolve().parents[1];data=root/'docs/data/models'/args.model
m=json.loads((data/'manifest.json').read_text());items=[]
for rec in m['records']:
 d=json.loads((data/'records'/(rec['id']+'.json')).read_text())
 for sampling,field in [(False,'oracle'),(True,'samples')]:
  for anchor,outs in d[field].items():
   for o in outs if sampling else [outs]:
    source=rec['id']+'__'+anchor+(f"__seed{o['seed']}" if sampling else '')
    items.append(dict(id=hashlib.sha256((args.model+source).encode()).hexdigest()[:16],source=source,record_id=rec['id'],anchor=anchor,sampling=sampling,seed=o['seed'],family=rec['family'],task=rec['task'],attitude=rec['attitude'],side=rec['side'],prefix=d['anchors'][anchor]['prefix'],phrases=o['phrases'],truncated=o['truncated']))
assert len(items)==351 and sum(not i['sampling'] for i in items)==156
random.Random(73129).shuffle(items);(data/'analysis').mkdir(exist_ok=True);(data/'analysis/readouts.json').write_text(json.dumps(items,ensure_ascii=False));print(args.model,len(items),sum(len(x['phrases']) for x in items))
