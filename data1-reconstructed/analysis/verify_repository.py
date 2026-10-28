"""Validate evidence directly, without target rates or predefined success counts."""
from pathlib import Path
import json,hashlib,collections
ROOT=Path(__file__).resolve().parents[1]
def main():
 counts=collections.Counter();matches=0;errors=[];missing_batches=set();missing_batch_records=0
 for p in sorted((ROOT/'outputs').glob('*/*/*.jsonl')):
  for line in p.read_text().splitlines():
   r=json.loads(line)
   if r.get('provenance_status')=='data1':continue
   counts[r['provenance_status']]+=1
   src=r.get('source_record') or r.get('source_output') or {}
   matches+=r['raw_response']==src.get('raw_generated_text',src.get('raw_output'))
   ap=ROOT/r['artifact_path'];text=json.loads(ap.read_text())['full_user_message'] if ap.suffix=='.json' else ap.read_text()
   if hashlib.sha256(text.encode()).hexdigest()!=r['prompt_sha256']:errors.append(str(p)+':prompt')
   if hashlib.sha256(r['raw_response'].encode()).hexdigest()!=r['raw_response_sha256']:errors.append(str(p)+':response')
   if 'source_record' in r and r['prompt_sha256']!=src['prompt_sha256']:errors.append(str(p)+':source_prompt')
   if 'raw_batch_source' in r:
    b=r['raw_batch_source'];bp=ROOT/b['path']
    if not bp.exists():missing_batches.add(b['path']);missing_batch_records+=1
    elif hashlib.sha256(bp.read_bytes()).hexdigest()!=b['sha256']:errors.append(b['path']+':batch_hash')
 n=sum(counts.values())
 report={'status':'failed' if errors or matches!=n else 'partial','embedded_source_check':'passed' if matches==n else 'failed','recovered_response_exact_matches':matches,'recovered_responses_not_verified_in_package':n-matches,'exact_match_scope':'embedded source_record/source_output only; not independent original batch verification','raw_batch_response_exact_matches':0,'raw_batch_verification_status':'pending','missing_raw_batch_files':sorted(missing_batches),'missing_raw_batch_records':missing_batch_records,'token_decoding_status':'pending_pinned_generation_tokenizers','provenance_counts':dict(counts),'errors':errors}
 for name in ['structure_validation.json','constraint_validation.json']:(ROOT/'results'/name).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 return report
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))
