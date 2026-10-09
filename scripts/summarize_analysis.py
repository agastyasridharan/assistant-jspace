import json,statistics,time,collections,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]; O=R/'local/analysis-v2';D=R/'docs/data/analysis'
items=json.loads((O/'items.json').read_text()); labels={}
for p in O.glob('batch-*.json'):
 for a in json.loads(p.read_text())['annotations']:labels[a['id']]=a
assert len(labels)==len(items)==351,(len(labels),len(items))
manual={}
for line in (R/'analysis/primary-review.tsv').read_text().splitlines():
 rid,*vs=line.split('\t');manual[rid]={k:[] if v=='-' else list(map(int,v.split(','))) for k,v in zip(['explicit_user','implicit_user','assistant_assessment_explicit','assistant_intention_explicit'],vs)}
secondary=json.loads((R/'analysis/secondary-review.json').read_text())
rows=[]; disagreements=[]
for item in items:
 a=labels[item['id']];reviewed=dict(a)
 if item['anchor']=='request' and not item['sampling']:
  for k,v in manual[item['record_id']].items():
   if bool(a[k])!=bool(v):disagreements.append({'record':item['record_id'],'field':k,'automated':a[k],'reviewed':v})
   reviewed[k]=v
 if item['source'] in secondary:
  reviewed.update({k:v for k,v in secondary[item['source']].items() if k!='reason'})
 row={**item,'automated':a,'annotation':reviewed,'primary_reviewed':item['anchor']=='request' and not item['sampling']}
 for k in ['explicit_user','implicit_user','other_person','factual','unsupported']:row[k]=bool(reviewed[k])
 row['assistant_strict']=bool(reviewed['assistant_assessment_explicit'] or reviewed['assistant_intention_explicit'])
 row['assistant_compatible']=bool(row['assistant_strict'] or reviewed['assessment_unattributed'] or reviewed['response_unattributed'])
 row['internal_contradiction']=bool(reviewed['contradiction_pairs'])
 row['balance']=(len(reviewed['state_a_evidence'])-len(reviewed['state_b_evidence']))/len(item['phrases'])
 row['state']='both' if reviewed['state_a_evidence'] and reviewed['state_b_evidence'] else 'a' if reviewed['state_a_evidence'] else 'b' if reviewed['state_b_evidence'] else 'absent'
 rows.append(row)
lookup={x['source']:x for x in rows}
M=json.loads((R/'data/manifest.json').read_text()); recs={r['id']:r for r in M['records']}
def at(rid,anchor):
 rec=json.loads((R/'local/results/records'/f'{rid}.json').read_text());p=rec['anchors'][anchor]['position'];canonical=next(k for k,v in rec['anchors'].items() if v['position']==p);return lookup[rid+'__'+canonical]
def expected(row):return 'a' if row['side']==(1 if row['family']=='blame' else 0) else 'b'
pairs=[]
for rid,r in recs.items():
 if r['side']!=0:continue
 cid=next(x['id'] for x in recs.values() if x['pair_id']==r['pair_id'] and x['side']==1)
 for anchor in ['fact','primary','request']:
  a,b=at(rid,anchor),at(cid,anchor);sign=-1 if r['family']=='blame' else 1
  pairs.append({'family':r['family'],'task':r['task'],'attitude':r['attitude'],'anchor':anchor,'main':rid,'control':cid,'main_state':a['state'],'control_state':b['state'],'clean_both':a['state']==expected(a) and b['state']==expected(b),'expected_signed_balance_change':sign*(a['balance']-b['balance'])})
def conflict(row):
 f=row['family'];s=row['side'];att=row['attitude']
 if f in ['arithmetic','tutoring']:return (att=='stated' and s==0) or (att=='crossed' and s==1)
 return f in ['tylenol','confidentiality','tests','scheduling','booking','blame'] and s==0
for row in rows:
 row['assistant_control_supported']=row['assistant_strict']
 if row['anchor']=='request' and not row['sampling'] and conflict(row):
  pair=next(p for p in pairs if p['anchor']=='request' and row['record_id'] in [p['main'],p['control']])
  if pair['clean_both'] and row['annotation']['assessment_unattributed']:row['assistant_control_supported']=True
fields=['explicit_user','implicit_user','assistant_strict','assistant_control_supported','assistant_compatible','other_person','factual','unsupported','internal_contradiction']
def counts(xs):return {'n':len(xs),**{k:{'count':sum(x[k] for x in xs),'percent':100*sum(x[k] for x in xs)/len(xs)} for k in fields}}
primary=[r for r in rows if r['primary_reviewed']]
summary={'primary':counts(primary),'by_task':{t:counts([r for r in primary if r['task']==t]) for t in ['assist','copy']},'by_family':{f['id']:counts([r for r in primary if r['family']==f['id']]) for f in M['families']},'by_anchor':{a:counts([at(rid,a) for rid in recs]) for a in ['fact','primary','request']},'overlap':dict(collections.Counter(f"user={int(r['explicit_user'])},implicit={int(r['implicit_user'])},assistant={int(r['assistant_strict'])}" for r in primary)),'review_presence_disagreements':disagreements,'primary_reviewed_count':52}
sampling=[]
for key in sorted({r['source'].split('__seed')[0] for r in rows if r['sampling']}):
 xs=[r for r in rows if r['sampling'] and r['source'].split('__seed')[0]==key];assert len(xs)==5
 sampling.append({'site':key,'family':xs[0]['family'],'task':xs[0]['task'],'anchor':xs[0]['anchor'],'counts':counts(xs),'greedy':counts([lookup[key]]),'annotator_only':True})
summary['sampling']={'sites':len(sampling),'samples':sum(x['counts']['n'] for x in sampling),'consistency':{k:{'all_five':sum(s['counts'][k]['count']==5 for s in sampling),'none':sum(s['counts'][k]['count']==0 for s in sampling),'mixed':sum(0<s['counts'][k]['count']<5 for s in sampling)} for k in ['explicit_user','implicit_user','assistant_strict']}}
summary['first_ten_automated']={k:{'first_ten':sum(any(j<=10 for f in ([k] if k!='assistant_strict' else ['assistant_assessment_explicit','assistant_intention_explicit']) for j in r['automated'][f]) for r in primary),'all_phrases':sum(any(r['automated'][f] for f in ([k] if k!='assistant_strict' else ['assistant_assessment_explicit','assistant_intention_explicit'])) for r in primary)} for k in ['explicit_user','implicit_user','assistant_strict']}
summary['o_controls']=[{'family':f['id'],'anchor':a,'n':len(ps),'clean_both':sum(p['clean_both'] for p in ps),'directional':sum(p['expected_signed_balance_change']>0 for p in ps)} for f in M['families'] for a in ['fact','primary','request'] for ps in [[p for p in pairs if p['family']==f['id'] and p['anchor']==a]]]
summary['secondary_review']=secondary
summary['first_ten_reviewed']={k:sum(any(j<=10 for f in ([k] if k!='assistant_strict' else ['assistant_assessment_explicit','assistant_intention_explicit']) for j in r['annotation'][f]) for r in primary) for k in ['explicit_user','assistant_strict']}
summary['scope_note']='Primary core percentages were audited by the research assistant after reading all 52 full request-boundary decodes. Secondary anchor, fidelity, contradiction, control, and sample labels are automated and provisional. Representative manual evidence indices are not exhaustive. Neither is independent human validation. Strict user attribution requires an actor-linked mental state or goal, not a bare request label. Headline strict Assistant includes explicit self-owned offers, judgments, refusals, apologies or complete response presentation; ambiguous truncated role prefixes do not suffice.'
summary['created_at']=time.time();summary['protocol_sha256']=hashlib.sha256((R/'analysis/protocol.json').read_bytes()).hexdigest()
for name,obj in [('summary',summary),('readouts',rows),('o-controls',pairs),('sample-stability',sampling)]: (D/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n')
print(json.dumps({k:summary[k] for k in ['primary','by_task','sampling','first_ten_automated']},indent=2))
