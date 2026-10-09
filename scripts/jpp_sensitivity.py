import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1]; M=json.load(open(R/'data/manifest.json'));V=json.load(open(R/'local/vocabulary.json'))
def value(r,a,word):
 ids={x['ids'][0] for g in V[r['family']].values() for x in g['accepted'] if x['text'].strip().lower()==word}
 d=json.load(open(R/'local/results/records'/f"{r['id']}.json"));read=d['jpp'][str(d['anchors'][a]['position'])]['44']; lookup={x['id']:x['logit'] for x in read['concepts']};return statistics.mean(lookup[i] for i in ids)
rows=[]
for r in M['records']:
 if r['side']!=0 or r['family'] not in ['tylenol','arithmetic','tutoring']:continue
 c=next(x for x in M['records'] if x['pair_id']==r['pair_id'] and x['side']==1)
 for anchor in ['fact','primary','request']:
  for word in (['dangerous','toxic','overdose'] if r['family']=='tylenol' else ['incorrect','error']):
   change=value(r,anchor,word)-value(c,anchor,word)
   if r['family']!='tylenol':change-=value(r,anchor,'correct')-value(c,anchor,'correct')
   rows.append(dict(family=r['family'],task=r['task'],attitude=r['attitude'],anchor=anchor,candidate=word,delta=change))
json.dump(rows,open(R/'docs/data/analysis/jpp-sensitivity.json','w'),indent=2)
for f in ['tylenol','arithmetic','tutoring']:
 for anchor in ['fact','request']:
  for word in (['dangerous','toxic','overdose'] if f=='tylenol' else ['incorrect','error']):
   ds=[r['delta'] for r in rows if r['family']==f and r['anchor']==anchor and r['candidate']==word];print(f,anchor,word,sum(v>0 for v in ds),len(ds),round(statistics.mean(ds),3))
