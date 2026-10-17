"""Record this assistant's phrase-by-phrase request review, without altering GPT labels."""
import json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[2]
fixes={
'llama-base':[
('arithmetic-crossed-copy-0',6,False,False,'Unrelated incomplete question about off-the-shoulder clothing; no identifiable focal user state or helper response.'),
('arithmetic-crossed-copy-1',1,True,True,'Contains a user request to explain the steps and an Answer heading; retain both components, consistently with the paired readout.'),
('scheduling-stated-copy-0',2,True,True,'Thanks followed by Here is the timetable spans user acknowledgement and helper response; identical phrase is both-labeled in the paired control.'),
('tutoring-stated-assist-0',5,False,False,'Unrelated incomplete clothing question has unresolved ownership.'),
('tutoring-stated-assist-0',10,False,False,'Repeated unrelated incomplete clothing question has unresolved ownership.'),
('tutoring-stated-assist-1',9,False,False,'Unrelated incomplete clothing question has unresolved ownership.')],
 'tulu':[
('arithmetic-stated-copy-0',2,False,False,'Unrelated incomplete clothing question has unresolved ownership.'),
('arithmetic-stated-copy-0',7,False,False,'Repeated unrelated incomplete clothing question has unresolved ownership.'),
('arithmetic-stated-copy-1',3,False,False,'Unrelated incomplete clothing question has unresolved ownership.'),
('arithmetic-stated-copy-1',10,False,False,'Repeated unrelated incomplete clothing question has unresolved ownership.'),
('bereavement-stated-copy-0',10,False,False,'Third-party conquest fragment has no clear connection to focal user or helper; paired control leaves it unlabeled.'),
('blame-stated-copy-0',6,False,False,'Unrelated incomplete clothing question has unresolved ownership.'),
('blame-stated-copy-0',2,True,True,'User question followed by Answer heading; preserve both components consistently with identical phrase in paired control.'),
('frustration-stated-assist-1',8,False,False,'and why followed by a generic numbered heading does not specify user state or helper response; identical phrase in paired main is unassigned.'),
('frustration-stated-copy-1',4,False,False,'Unrelated incomplete clothing question has unresolved ownership.'),
('tutoring-stated-copy-0',7,False,False,'Unrelated incomplete clothing question has unresolved ownership.')]
}
for key,fs in fixes.items():
 p=R/'docs/data/models'/key/'analysis';rows=json.load(open(p/'readouts.json'));labels={r['source']:r for r in json.load(open(p/'broad-perspectives.json'))}
 requests=sorted([r for r in rows if not r['sampling'] and r['anchor']=='request'],key=lambda r:r['source']);assert len(requests)==52
 changes=[]
 for source,i,u,a,reason in fs:
  source+='__request';r=next(r for r in requests if r['source']==source);old=labels[source]['annotation']
  changes.append({'source':source,'index':i,'phrase':r['phrases'][i-1],'before':{'user':i in old['user'],'assistant':i in old['assistant']},'after':{'user':u,'assistant':a},'reason':reason})
 review={'reviewer':'Codex assistant; not a human review or independent blinded annotator','scope':'All 52 final-request greedy readouts, all 520 phrases, read with source prefixes and automated labels. Other anchors and samples are not exhaustively reviewed.','created_at':time.time(),'source_sha256':hashlib.sha256((p/'readouts.json').read_bytes()).hexdigest(),'automated_labels_sha256':hashlib.sha256((p/'broad-perspectives.json').read_bytes()).hexdigest(),'reviewed_sources':[r['source'] for r in requests],'corrections':changes,'limitations':['Many decoded phrases are request continuations, generic answer scaffolding, or unrelated fragments. Broad attribution is not evidence of factual fidelity or a specific psychological state.','Ambiguous ownership remains in greetings, draft-email fragments, generic clarification questions, and answer headings. The review does not eliminate classifier uncertainty.','Original automated labels remain unchanged and are the primary comparable analysis; correction sensitivity is reported separately.']}
 (p/'request-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
 prov=json.load(open(p/'broad-perspective-provenance.json'));prov['status']='Automated exhaustive annotation; all final-request greedy readouts reviewed by Codex, with separate corrections';prov['review']='request-review.json';(p/'broad-perspective-provenance.json').write_text(json.dumps(prov,indent=2))
 print(key,len(changes),'corrections')
