"""Descriptive, model-isolated role attribution analysis; frozen automated labels primary."""
import json,hashlib,statistics,copy
from pathlib import Path
from collections import Counter,defaultdict
R=Path(__file__).resolve().parents[1]
def stats(rs,k=None):
 out={'n':len(rs),'presence':dict(user=0,assistant=0,both=0,neither=0),'mean_phrase_fraction':dict(user=0.,assistant=0.)}
 if not rs:return out
 for r in rs:
  n=min(len(r['phrases']),k) if k else len(r['phrases']);assert n
  u={i for i in r['labels']['user'] if i<=n};a={i for i in r['labels']['assistant'] if i<=n}
  for c,v in [('user',u),('assistant',a)]:out['presence'][c]+=bool(v);out['mean_phrase_fraction'][c]+=len(v)/n/len(rs)
  out['presence']['both']+=bool(u and a);out['presence']['neither']+=not(u or a)
 return out
results={}
for key in ['qwen','llama-base','tulu']:
 p=R/'docs/data' if key=='qwen' else R/'docs/data/models'/key
 rows=json.load(open(p/'analysis/readouts.json'));labs={r['source']:r for r in json.load(open(p/'analysis/broad-perspectives.json'))}
 for r in rows:
  b=labs[r['source']];assert b['phrases_sha256']==hashlib.sha256(json.dumps(r['phrases'],ensure_ascii=False).encode()).hexdigest();r['labels']=b['annotation']
 greedy=[r for r in rows if not r['sampling']];lookup={r['source']:r for r in greedy};request=[r for r in greedy if r['anchor']=='request'];assert len(request)==52 and len(greedy)==156 and len(rows)==351
 def pairs(rs,k=None):
  out={}
  for c in ['user','assistant']:
   ds=[]
   for a in rs:
    if a['task']!='assist':continue
    b=lookup[a['source'].replace('-assist-','-copy-')]
    ds.append(stats([a],k)['mean_phrase_fraction'][c]-stats([b],k)['mean_phrase_fraction'][c])
   out[c]={'n':len(ds),'mean_assist_minus_copy':statistics.mean(ds),'positive':sum(x>1e-9 for x in ds),'negative':sum(x< -1e-9 for x in ds),'ties':sum(abs(x)<=1e-9 for x in ds)}
  return out
 out={'request':stats(request),'request_first10':stats(request,10),'request_by_task':{t:stats([r for r in request if r['task']==t]) for t in ['assist','copy']},'request_first10_by_task':{t:stats([r for r in request if r['task']==t],10) for t in ['assist','copy']},'request_by_family':{f:stats([r for r in request if r['family']==f]) for f in sorted({r['family'] for r in request})},'paired_request':pairs(request),'paired_request_first10':pairs(request,10)}
 # Same 44 condition positions in all three models; excludes Tylenol special anchors and merged confidentiality anchor.
 common=[r for r in greedy if r['anchor']=='primary' and r['family'] not in ['tylenol','confidentiality']];assert len(common)==44
 out['matched_anchors']={a:{t:stats([lookup[r['record_id']+'__'+a] for r in common if r['task']==t]) for t in ['assist','copy']} for a in ['fact','primary','request']}
 out['paired_matched_anchors']={a:pairs([lookup[r['record_id']+'__'+a] for r in common]) for a in ['fact','primary','request']}
 out['paired_primary_by_family']={f:pairs([r for r in common if r['family']==f]) for f in sorted({r['family'] for r in common})}
 out['primary_equal_family_assistant_difference']=statistics.mean(v['assistant']['mean_assist_minus_copy'] for v in out['paired_primary_by_family'].values())
 out['primary_without_auxiliary']=pairs([r for r in common if r['family'] not in ['bereavement','frustration']])
 groups=defaultdict(list)
 for r in rows:
  if r['sampling']:groups[r['source'].split('__seed')[0]].append(r)
 assert len(groups)==39 and all(len(v)==5 for v in groups.values())
 out['samples']={}
 for c in ['user','assistant']:
  hist=Counter();agreements=0;ranges=[]
  for source,rs in groups.items():
   hist[sum(bool(r['labels'][c]) for r in rs)]+=1;agreements+=sum(bool(r['labels'][c])==bool(lookup[source]['labels'][c]) for r in rs)
   ds=[stats([r])['mean_phrase_fraction'][c] for r in rs];ranges.append(max(ds)-min(ds))
  out['samples'][c]={'presence_count_out_of_five_histogram':dict(sorted(hist.items())),'agreement_with_greedy':agreements,'comparisons':195,'median_density_range':statistics.median(ranges)}
 out['factual_pairs']={}
 for a in ['fact','primary','request']:
  rr=[r for r in greedy if r['anchor']==a and r['side']==0];result={}
  for c in ['user','assistant']:
   ds=[stats([r])['mean_phrase_fraction'][c]-stats([lookup[r['source'].replace('-0__','-1__')]])['mean_phrase_fraction'][c] for r in rr]
   result[c]={'n':len(ds),'mean_main_minus_control':statistics.mean(ds),'positive':sum(x>1e-9 for x in ds),'negative':sum(x< -1e-9 for x in ds),'ties':sum(abs(x)<=1e-9 for x in ds)}
  out['factual_pairs'][a]=result
 if key!='qwen':
  review=json.load(open(p/'analysis/request-review.json'));rr=copy.deepcopy(request);by={r['source']:r for r in rr}
  for change in review['corrections']:
   r=by[change['source']]
   for c,value in change['after'].items():
    s=set(r['labels'][c]);s.add(change['index']) if value else s.discard(change['index']);r['labels'][c]=sorted(s)
  out['review_sensitivity']={'corrections':len(review['corrections']),'request':stats(rr)}
  checks=json.load(open(p/'checks.json'));ns=checks['numerical_sensitivity'];out['numerical_sensitivity']={'token_identical':ns['identical_readouts'],'n':ns['total'],'changed':ns['total']-ns['identical_readouts'],'mean_exact_phrase_jaccard':statistics.mean(x['exact_phrase_jaccard'] for x in ns['sites']),'changed_mean_exact_phrase_jaccard':statistics.mean(x['exact_phrase_jaccard'] for x in ns['sites'] if not x['identical_tokens']),'interpretation':'Lexical stability only; changed diagnostic readouts were not exhaustively perspective-labeled.'}
  out['copy_behavior']={'full_exact':sum(json.load(open(p/'records'/(r['record_id']+'.json')))['copy_exact'] for r in request if r['task']=='copy'),'n':26}
 out['provenance']={'readouts_sha256':hashlib.sha256((p/'analysis/readouts.json').read_bytes()).hexdigest(),'labels_sha256':hashlib.sha256((p/'analysis/broad-perspectives.json').read_bytes()).hexdigest(),'denominator':'52 final-request greedy readouts = 26 messages x 2 instructions; not 52 independent scenarios. Labels overlap. Phrase fractions equally weight readouts.'}
 results[key]=out
 if key!='qwen':(p/'analysis/summary.json').write_text(json.dumps(out,indent=2)+'\n')
 print(key,json.dumps({k:out[k] for k in ['request','request_first10','request_by_task','paired_matched_anchors','samples']}))
(R/'docs/data/replication-comparison.json').write_text(json.dumps(results,indent=2)+'\n')
