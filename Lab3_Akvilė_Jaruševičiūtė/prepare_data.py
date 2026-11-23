"""Verify/copy the instructor-supplied Oz dataset; no downloads or remote code."""
from pathlib import Path
import argparse,json,hashlib,shutil
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
m=json.loads(Path(__file__).with_name('oz_manifest.json').read_text());payload=a.source.read_bytes()
assert hashlib.sha256(payload).hexdigest()==m['selected_sha256'],'Dataset hash differs; obtain the reviewed Oz revision.'
rows=[json.loads(s) for s in payload.decode().splitlines() if s.strip()]
assert {r['id'] for r in rows}=={r['id'] for r in m['ids']}
a.out.mkdir(parents=True,exist_ok=True);target=a.out/'oz12.jsonl'
if target.resolve()!=a.source.resolve():shutil.copyfile(a.source,target)
print('Verified12 Oz excerpts:',target)
