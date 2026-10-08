"""Export only study data, never environment variables, logs, or credentials."""
import argparse,gzip,hashlib,json,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def write(path,obj):
    data=json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if path.exists() and path.read_text()==data:return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(data);tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default=str(ROOT/'local/results'));args=ap.parse_args()
    results=Path(args.results);dest=ROOT/'docs/data';dest.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    write(dest/'manifest.json',manifest)
    records=[];oracle_count=sample_count=0
    for rec in manifest['records']:
        f=results/'records'/f"{rec['id']}.json"
        if not f.exists():continue
        record=json.loads(f.read_text());record['oracle']={};record['samples']={}
        for o in sorted((results/'oracle').glob(rec['id']+'__*.json')):
            value=json.loads(o.read_text())
            if '__seed' in o.stem:
                record['samples'].setdefault(value['anchor'],[]).append(value);sample_count+=1
            else:record['oracle'][value['anchor']]=value;oracle_count+=1
        write(dest/'records'/f"{rec['id']}.json",record)
        records.append(dict(id=rec['id'],truncated=record['reply_truncated'],oracle=len(record['oracle']),samples=sum(map(len,record['samples'].values()))))
        t=results/'trajectory'/f"{rec['id']}.json"
        target=dest/'trajectory'/f"{rec['id']}.json.gz"
        if t.exists() and not target.exists():
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(gzip.compress(t.read_bytes(),mtime=0))
    checks={}
    for name in ['oracle_calibration','oracle_prompt','oracle_sites','environment']:
        p=results/'checks'/f'{name}.json'
        if p.exists():checks[name]=json.loads(p.read_text())
    write(dest/'checks.json',checks)
    status=json.loads((results/'status.json').read_text()) if (results/'status.json').exists() else dict(stage='preparing',reason='Worker status not yet available')
    # Tracebacks and remote filesystem details are kept locally, not published.
    status.pop('traceback',None);status.pop('pid',None)
    if 'error' in status:status['error']=status['error'][:500]
    summary=dict(run_id=manifest['run_id'],status=status,records=records,total_records=52,completed_records=len(records),
        oracle_sites=oracle_count,oracle_samples=sample_count,manifest_sha256=hashlib.sha256((ROOT/'data/manifest.json').read_bytes()).hexdigest(),
        invariants_checked=len([p for p in (results/'checks').glob('*.json') if '-assist.' in p.name or '-copy.' in p.name]),
        expected_oracle_sites=len(checks.get('oracle_sites',[])) or None)
    write(dest/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['completed_records','oracle_sites','oracle_samples']}))

if __name__=='__main__':main()
