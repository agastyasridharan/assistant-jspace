"""Descriptive boundary audit; no semantic classifier or primary-data edits."""
import json,re,html,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];dest=R/'docs/data/base-behavior';dest.mkdir(parents=True,exist_ok=True)
manifest=json.load(open(R/'data/manifest.json'));records={r['id']:r for r in manifest['records']}
pattern=re.compile(r'<\|(system|user|assistant)\|>')
def analyze(key,d):
 text=d['reply'];headers=list(pattern.finditer(text));first=text[:headers[0].start()].strip() if headers else text.strip();segments=[]
 for i,h in enumerate(headers):
  end=headers[i+1].start() if i+1<len(headers) else len(text)
  content=text[h.end():end].strip();complete=i+1<len(headers) or not d.get('reply_truncated',d.get('truncated',False))
  segments.append({'role':h.group(1),'text':content,'complete':complete,'repeats_supplied_user_text':h.group(1)=='user' and content in [records[key]['prompt'].strip(),records[key]['message'].strip()]})
 return {'prefix_matches_original':d.get('prefix_matches_original'),'generated_tokens':d.get('tokens',len(d.get('reply_token_ids',[]))),'id':key,'task':records[key]['task'],'family':records[key]['family'],'prompt':records[key]['prompt'],'message':records[key]['message'],'reply':text,'first_segment':first,'first_segment_exact_copy':first==records[key]['message'].strip(),'full_exact_copy':text.strip()==records[key]['message'].strip(),'segments':segments,'truncated':d.get('reply_truncated',d.get('truncated',False)),'source_sha256':hashlib.sha256(json.dumps(d,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
short=[analyze(k,json.load(open(R/'local/llama-results-v2/llama-base/records'/(k+'.json')))) for k in records]
long=[]
for p in sorted((R/'local/base-behavior').glob('*-0.json')):
 d=json.load(open(p));long.append(analyze(d['id'],d))
summary={}
for task in ['assist','copy']:
 rs=[r for r in short if r['task']==task]
 summary[task]={'n':len(rs),'role_headers':sum(bool(r['segments']) for r in rs),'first_segment_exact_copy':sum(r['first_segment_exact_copy'] for r in rs),'full_exact_copy':sum(r['full_exact_copy'] for r in rs),'truncated':sum(r['truncated'] for r in rs),'complete_new_user_turn':sum(any(s['role']=='user' and s['complete'] and not s['repeats_supplied_user_text'] for s in r['segments']) for r in rs)}
for name,value in [('short-replies',short),('long-replies',long),('summary',summary)]:
 (dest/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
e=html.escape
head=(R/'docs/index.html').read_text().split('<body>')[0].replace('<title>Assistant perspective · Qwen3.6</title>','<title>Base model · generated conversations</title>')
parts=[head,'<body><header><h1>Base model <span>· generated conversations</span></h1><a href="index.html?model=llama-base">Back to paired readouts</a></header><main style="max-width:1100px">',
'<p>These are actual continuations from base Llama-3.1-70B, not O-lens readouts. They use the same Tulu-formatted prompts, system instruction, BF16 weights and greedy decoding as the study. Textual role headers do not stop generation; the original EOS stop rules remain in place.</p>',
'<p><strong>Initial response and whole-output compliance differ.</strong> A model may copy correctly, then invent another user turn. The audit below splits on literal system/user/assistant headers. This is a descriptive parsing rule, not a semantic measure of persona.</p>',
'<table><thead><tr><th>256-token outputs</th><th>Assist</th><th>Copy</th></tr></thead><tbody>']
for label,key in [('Outputs inspected','n'),('Contain generated role headers','role_headers'),('First segment exactly copies message','first_segment_exact_copy'),('Whole output exactly copies message','full_exact_copy'),('Contain a complete user turn different from supplied text','complete_new_user_turn'),('Reach token cap','truncated')]:
 parts.append('<tr><td>'+label+'</td>'+''.join('<td>'+str(summary[t][key])+'/26</td>' for t in ['assist','copy'])+'</tr>')
parts+=['</tbody></table><p class="hint">“Different user turn” is literal text inequality, not proof of a novel meaning; repeated transcripts and new dialogue are distinguished in the examples. Header-free or unfinished continuations remain unresolved.</p>',
'<p>The eight longer continuations are an exploratory selection made after inspecting the short outputs: original main Assist/Copy prompts for arithmetic, bereavement, frustration and Tylenol. They extend the cap to 1,024 tokens and check that the first 256 tokens reproduce the original output. They are excluded from primary lens analysis.</p>',
f'<h2>Longer continuations · {len(long)} / 8 available</h2>']
def card(r):
 return '<details><summary>'+e(r['id'])+(' · first segment copies exactly' if r['first_segment_exact_copy'] else '')+'</summary><div class="meta">Input user content</div><div class="raw">'+e(r['prompt'])+'</div><div class="meta">Initial assistant-position segment</div><div class="raw">'+e(r['first_segment'])+'</div><div class="meta">Full generated continuation · '+('token cap reached' if r['truncated'] else 'stopped')+'</div><div class="raw">'+e(r['reply'])+'</div></details>'
if len(long)==8:
 parts.append('<p>All eight continuations reached the 1,024-token cap and exactly reproduced their original 256-token prefixes. All generated further role headers. Seven are dominated by repeated exchanges; the frustration Assist example instead develops several different user objections and assistant suggestions. These are selected exploratory examples, not a population estimate.</p>')
parts += [card(r) for r in long]
parts += ['<h2>All original 52 continuations</h2>']+[card(r) for r in short]
parts += ['<p><a href="data/base-behavior/short-replies.json">Download original continuations and boundary audit</a> · <a href="data/base-behavior/long-replies.json">Download longer continuations</a></p></main></body></html>']
(R/'docs/base-behavior.html').write_text('\n'.join(parts))
print(json.dumps({'summary':summary,'long':len(long)}))
