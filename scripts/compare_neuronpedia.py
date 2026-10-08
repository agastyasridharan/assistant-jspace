"""Compare decoded raw top-eight J++ tokens; API does not return readout token IDs."""
import json, statistics, math
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'local/neuronpedia-validation'
rows=[]; mismatches=[]; examples=[]; verified=0
for item in json.loads((ROOT/'local/tokenization.json').read_text()):
 path=out/(item['id']+'.json')
 if not path.exists(): continue
 v=json.loads(path.read_text()); r=v['response']; ids=v['request']['inputTokenIds']
 assert r['meta']['types']==['JPP_LENS']
 assert [t['id'] for t in r['tokens']]==ids, path
 verified+=len(ids)
 tokens={t['position']:t for t in r['tokens']}
 traj=json.loads((ROOT/'local/results/trajectory'/path.name).read_text())
 rec=json.loads((ROOT/'local/results/records'/path.name).read_text())
 anchors={a['position'] for a in rec['anchors'].values()}
 layers=r['meta']['layers_by_type']['JPP_LENS']
 for layer, values in traj.items():
  for a in values:
   assert tokens[a['position']]['id']==a['token_id'], (path,layer,a['position'])
   b=next(s for s in tokens[a['position']]['results'] if s['type']=='JPP_LENS')
   ix=layers.index(int(layer)); remote=b['top_tokens'][ix]; probs=b['top_probs'][ix]
   local=[x['token'] for x in a['raw'][:8]]
   overlap=len(set(local)&set(remote))/8
   logits={x['token']:x['logit'] for x in a['raw']}
   errors=[]
   if remote[0] in logits and probs[0]>0:
    for word,prob in zip(remote[1:],probs[1:]):
     if word in logits and prob>0: errors.append(abs((logits[word]-logits[remote[0]])-math.log(prob/probs[0])))
   row={'record':path.stem,'layer':int(layer),'position':a['position'],'anchor':a['position'] in anchors,'top1':local[0]==remote[0],'overlap':overlap,'ordered8':local==remote,'set8':set(local)==set(remote),'log_ratio_errors':errors,'local_margin':a['raw'][0]['logit']-a['raw'][1]['logit'],'duplicate_decoding':len(set(local))<8 or len(set(remote))<8}
   rows.append(row)
   if not row['top1']: mismatches.append({**row,'local':local,'neuronpedia':remote})
   if path.stem=='tylenol-stated-assist-0' and a['position'] in anchors and int(layer)==44:
    examples.append({'anchor':[k for k,v in rec['anchors'].items() if v['position']==a['position']],'position':a['position'],'token':a['token'],'local':local,'neuronpedia':remote})
def agg(rs):
 errors=[e for r in rs for e in r['log_ratio_errors']]
 return {'readouts':len(rs),'top1_agreement':statistics.mean(r['top1'] for r in rs),'mean_top8_overlap':statistics.mean(r['overlap'] for r in rs),'identical_top8_set':statistics.mean(r['set8'] for r in rs),'identical_top8_order':statistics.mean(r['ordered8'] for r in rs),'mean_absolute_log_ratio_error':statistics.mean(errors),'median_absolute_log_ratio_error':statistics.median(errors),'duplicate_decoding_readouts':sum(r['duplicate_decoding'] for r in rs)}
summary={'records':len(list(out.glob('*-0.json')))+len(list(out.glob('*-1.json'))),'verified_input_token_ids':verified,'all_user_positions':agg(rows),'unique_anchor_positions':agg([r for r in rows if r['anchor']]),'by_layer':{str(l):agg([r for r in rows if r['layer']==l]) for l in sorted(set(r['layer'] for r in rows))},'top1_mismatches':len(mismatches),'mismatch_local_margin_median':statistics.median(r['local_margin'] for r in mismatches) if mismatches else None,'mismatch_local_margin_max':max((r['local_margin'] for r in mismatches),default=0),'examples':examples,'limitations':['Readout matching uses decoded token strings because Neuronpedia does not return readout vocabulary IDs.','Neuronpedia runtime model/lens revisions and numerical precision are not exposed by this API; exact artifact identity is unverified.','API returns top eight only. Local ranks 9-50 and fixed-concept scores are not externally validated.','Numerical agreement validates computation, not whether a readout represents the user or Assistant.']}
summary['records']=len({r['record'] for r in rows})
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
(out/'mismatches.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in mismatches))
print(json.dumps(summary,indent=2,ensure_ascii=False))
