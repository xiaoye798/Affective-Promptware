"""Generate scores only from actual raw call records; never infer missing ratings."""
import csv,json,math
from pathlib import Path
R=Path(__file__).resolve().parent
rows=[]
for line in (R/'raw_calls.jsonl').read_text().splitlines():
 if not line.strip():continue
 x=json.loads(line)
 for k in ['material_key','artifact_sha256','rater_model','request_id','temperature','requested_at','raw_response']:assert k in x,k
 assert isinstance(x['temperature'],(int,float)) and not isinstance(x['temperature'],bool)
 parsed=json.loads(x['raw_response'])
 for k in ['task_pressure','emotion_intensity']:
  v=parsed[k];assert isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=4
  assert k not in x or x[k]==v
  x[k]=v
 x['provenance_status']='observed';rows.append(x)
fields=['material_key','artifact_sha256','rater_model','request_id','temperature','requested_at','raw_response','task_pressure','emotion_intensity','provenance_status']
with (R/'scores.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
