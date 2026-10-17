import json,sys
from pathlib import Path
k=sys.argv[1];start=int(sys.argv[2]);p=Path('docs/data/models')/k/'analysis'
rs=sorted([r for r in json.load(open(p/'readouts.json')) if not r['sampling'] and r['anchor']=='request'],key=lambda r:r['source'])
ba={b['source']:b for b in json.load(open(p/'broad-perspectives.json'))}
for n,r in enumerate(rs[start:start+26],start):
 b=ba[r['source']]['annotation'];print('\n',n,r['source']);print('PREFIX',r['prefix'].split('<|user|>')[-1].strip())
 for i,t in enumerate(r['phrases'],1):print(str(i)+('U' if i in b['user'] else '')+('A' if i in b['assistant'] else '')+':',t)
