"""Analyze saved candidate logits only. No inference or outcome-driven token selection."""
import json,statistics,collections
from pathlib import Path
R=Path(__file__).resolve().parents[1]; M=json.loads((R/'data/manifest.json').read_text());V=json.loads((R/'local/vocabulary.json').read_text())
# A is adverse state, B favorable state. Single-sided axes are marked, not called margins.
axes={'tylenol':(['dangerous','toxic','overdose'],[]),'confidentiality':(['permission','consent'],[]),'tutoring':(['incorrect','error'],['correct']),'arithmetic':(['incorrect','error'],['correct']),'tests':(['failure','failed'],['passed']),'scheduling':(['late','infeasible','delay'],[]),'booking':(['uncertain'],['confirmed']),'blame':(['incorrect','error'],['correct']),'bereavement':(['death','loss'],[]),'frustration':(['failed','unresolved'],['fixed'])}
def word_ids(family,word):
 return sorted({a['ids'][0] for g in V[family].values() for a in g['accepted'] if a['text'].strip().lower()==word})
def score(read,family,words):
 lookup={x['id']:x for x in read['concepts']}; return statistics.mean(statistics.mean(lookup[i]['logit'] for i in word_ids(family,w)) for w in words if word_ids(family,w))
rows=[]
for r in M['records']:
 d=json.loads((R/'local/results/records'/f"{r['id']}.json").read_text());a,b=axes[r['family']]
 for anchor,p in d['anchors'].items():
  for layer,read in d['jpp'][str(p['position'])].items():
   sa=score(read,r['family'],a);sb=score(read,r['family'],b) if b else 0
   stance=None
   if r['family'] in ['tutoring','arithmetic']:stance=score(read,r['family'],['confident'])-score(read,r['family'],['doubt','unsure'])
   if r['family']=='frustration':stance=score(read,r['family'],['pleased'])-score(read,r['family'],['frustrated','blame'])
   response_words=next(f['concepts'][2] for f in M['families'] if f['id']==r['family'])
   response_score=score(read,r['family'],response_words)
   rows.append(dict(response_score=response_score,record=r['id'],family=r['family'],pair_id=r['pair_id'],side=r['side'],task=r['task'],attitude=r['attitude'],anchor=anchor,layer=int(layer),value=sa-sb,stance=stance,one_sided=not bool(b)))
pairs=[]
for r in rows:
 if r['side']!=0:continue
 c=next(x for x in rows if x['pair_id']==r['pair_id'] and x['side']==1 and x['anchor']==r['anchor'] and x['layer']==r['layer'])
 expected_sign=-1 if r['family'] in ['blame','confidentiality'] else 1
 pairs.append({k:r[k] for k in ['family','pair_id','task','attitude','anchor','layer','one_sided']}|{'main_minus_control':r['value']-c['value'],'expected_signed_delta':expected_sign*(r['value']-c['value'])})
stance=[]
for r in rows:
 if r['attitude']!='stated' or r['stance'] is None:continue
 c=next(x for x in rows if x['family']==r['family'] and x['attitude']=='crossed' and x['side']==r['side'] and x['task']==r['task'] and x['anchor']==r['anchor'] and x['layer']==r['layer'])
 # expected confidence higher in stated; gratitude higher in crossed.
 sign=-1 if r['family']=='frustration' else 1
 stance.append({k:r[k] for k in ['family','task','side','anchor','layer']}|{'expected_signed_delta':sign*(r['stance']-c['stance'])})
response_pairs=[]
for r in rows:
 if r['task']!='assist':continue
 c=next(x for x in rows if x['family']==r['family'] and x['attitude']==r['attitude'] and x['side']==r['side'] and x['anchor']==r['anchor'] and x['layer']==r['layer'] and x['task']=='copy')
 response_pairs.append({k:r[k] for k in ['family','attitude','side','anchor','layer']}|{'assist_minus_copy':r['response_score']-c['response_score']})
summary=[]
for family in axes:
 for anchor in ['fact','primary','request']:
  for layer in M['layers']:
   ps=[p for p in pairs if p['family']==family and p['anchor']==anchor and p['layer']==layer]
   summary.append(dict(family=family,anchor=anchor,layer=layer,n=len(ps),correct_direction=sum(p['expected_signed_delta']>0 for p in ps),mean_delta=statistics.mean(p['expected_signed_delta'] for p in ps),one_sided=not bool(axes[family][1])))
obj={'method':'Mean spellings within word, equal mean words within pole, A minus B. Compare matched factual conditions at same block and anchor. No softmax or top-k restriction. All original accepted spellings retained.','axes':axes,'response_pairs':response_pairs,'unavailable_words':[{'family':f,'word':w} for f,(a,b) in axes.items() for w in a+b if not word_ids(f,w)],'per_condition':rows,'pairs':pairs,'stance_pairs':stance,'summary':summary,'token_ids':{f:{w:word_ids(f,w) for w in a+b} for f,(a,b) in axes.items()}}
(R/'docs/data/analysis/jpp.json').write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n')
print('Block 44 fact / appraisal / request: expected-direction pairs, mean signed logit change')
for f in axes:
 print(f,[(x['anchor'],str(x['correct_direction'])+'/'+str(x['n']),round(x['mean_delta'],3)) for x in summary if x['family']==f and x['layer']==44])
print('Stance at layer44')
for f in ['arithmetic','tutoring','frustration']:
 for anchor in ['fact','primary','request']:
  xs=[x['expected_signed_delta'] for x in stance if x['family']==f and x['anchor']==anchor and x['layer']==44];print(f,anchor,sum(v>0 for v in xs),len(xs),round(statistics.mean(xs),3))
