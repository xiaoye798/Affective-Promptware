"""Separate reconstruction tables: recovered records + data1; not empirical replication."""
from pathlib import Path
import json,csv,collections
import compute_rates as c
R=Path(__file__).resolve().parents[1]
records=[]
for setting in ['trusted','permissive','cross_model']:
 for p in (R/'outputs'/setting).glob('*/*.jsonl'):
  for line in p.read_text().splitlines():
   r=json.loads(line);records.append({'setting':setting,'model':r['model'],'condition':r['condition'],'scenario':r['scenario'],'emotion':r.get('emotion','') if r['condition']!='explicit' else '', 'run':r.get('run',1),'raw_response':r['raw_response'],'provenance_status':r['provenance_status']})
# Preserve provenance counts, including data1 as its own category.
orig_counts=c.source_counts
def counts(rs):return {**orig_counts(rs),'data1_records':sum(r['provenance_status']=='data1' for r in rs)}
c.source_counts=counts
protocol=c.parse_records(records,c.load_module('parse_protocol'));strict=c.parse_records(records,c.load_module('parse_strict'))
for name,parsed,settings in [('table1',protocol,['trusted']),('table7',strict,['trusted','permissive'])]:
 out=[]
 for m in c.TRUSTED_MODELS:
  for setting in settings:
   for cond in c.model_conditions(m,setting):out.append({'model':m,'setting':setting,'condition':cond,**c.summary(c.group_rows(parsed,setting,m,cond),c.expected_count(setting,cond))})
 c.write_csv(R/f'results/data1_{name}.csv',out)
out=[]
for name,parsed in [('protocol',protocol),('strict',strict)]:
 for l,r in [('affective','neutral'),('affective','pressure_only'),('pressure_only','neutral')]:
  a=c.group_rows(parsed,'trusted','qwen3-32b',l);b=c.group_rows(parsed,'trusted','qwen3-32b',r)
  out.append({'parser':name,'left_condition':l,'right_condition':r,**counts(a+b),**c.contrast(a,b,320)})
c.write_csv(R/'results/data1_pressure_decomposition.csv',out)
out=[]
for m in c.TRUSTED_MODELS:
 cond=c.model_conditions(m,'trusted')[0];a=c.group_rows(protocol,'trusted',m,cond);b=c.group_rows(protocol,'trusted',m,'neutral')
 out.append({'model':m,'left_condition':cond,'right_condition':'neutral',**counts(a+b),**c.contrast(a,b,320)})
c.write_csv(R/'results/data1_trusted_model_contrasts.csv',out)
ratings=[json.loads(x) for x in (R/'rater/raw_calls.data1.jsonl').read_text().splitlines()];out=[]
for metric in ['task_pressure','emotion_intensity']:
 paired=collections.defaultdict(dict)
 for r in ratings:paired[(r['scenario'],r['emotion'])][r['condition']]=r[metric]
 scenes=collections.defaultdict(list)
 for (s,e),p in paired.items():scenes[s].append(p['affective']-p['pressure_only'])
 ci=c.percentile_ci([sum(v)/len(v) for s,v in sorted(scenes.items())]);out.append({'metric':metric,'difference':ci['estimate'],'ci_lo':ci['ci_lo'],'ci_hi':ci['ci_hi'],'provenance_status':'data1'})
c.write_csv(R/'results/data1_rater_checks.csv',out)
det=[]
for name in ['recorded_scores.jsonl','recorded_scores.data1.jsonl']:det.extend(json.loads(l) for l in (R/'detection/scores'/name).read_text().splitlines())
assert len({(r['material_key'],r['view']) for r in det})==len(det)
out=[]
for cond in ['affective','neutral','explicit','pressure_only']:
 for view in ['full','external']:
  rs=[r for r in det if {'explicit_positive':'explicit'}.get(r['input_condition'],r['input_condition'])==cond and r['view']==view]
  out.append({'condition':cond,'view':view,'n':len(rs),'detected':sum(r['detected'] for r in rs),**counts(rs)})
c.write_csv(R/'results/data1_table4.csv',out)
print('Reconstruction tables written; measured-only tables unchanged.')
