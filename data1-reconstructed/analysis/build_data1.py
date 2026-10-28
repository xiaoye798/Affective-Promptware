"""The old random demo generator is retired. Validate the current reconstruction."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1]
n=0
for p in (R/'outputs').glob('*/*/*.jsonl'):
 for line in p.read_text().splitlines():
  r=json.loads(line)
  if r.get('provenance_status')!='data1':continue
  s=r['source_record'];assert bytes(s['raw_token_ids']).decode()==r['raw_response']
  b=R/r['raw_batch_source']['path'];assert hashlib.sha256(b.read_bytes()).hexdigest()==s['raw_batch_sha256']
  assert json.loads(b.read_text())['records'][0]['raw_generated_text']==r['raw_response'];n+=1
print('data1 internally consistent outputs:',n)
