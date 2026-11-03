import statistics
CRITERIA=['asset_identity_and_layout','visible_support','visible_clearance','visible_goal_match']

def rating_summary(parsed):
    rows=parsed['ratings']
    if len(rows)!=4 or {r['criterion'] for r in rows}!=set(CRITERIA):raise ValueError('Judge omitted or duplicated a criterion')
    if any(r['status'] not in ('pass','fail','uncertain') for r in rows):raise ValueError('Unknown rating status')
    observed=[float(r['status']=='pass') for r in rows if r['status']!='uncertain']
    return dict(V=statistics.mean(observed) if observed else None,coverage=len(observed)/4,uncertain=sum(r['status']=='uncertain' for r in rows))

"""Separate fixed-image judge variability from participant-output quality."""
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path


def observed_stats(values):
    values=[v for v in values if v is not None]
    return dict(n=len(values),mean=statistics.mean(values) if values else None,sample_variance=statistics.variance(values) if len(values)>1 else None)


def analyze(plan,complete,performance):
    jobs={r['id']:r for r in plan['jobs']};ratings={r['id']:r for r in complete['ratings']}
    if len(ratings)!=len(complete['ratings']) or not complete['complete'] or set(ratings)!=set(jobs):raise ValueError('Complete unique judge jobs required')
    models={r['id']:r for r in performance['individual']};fixed=defaultdict(list);frontier=defaultdict(list);disagreements=[]
    for key,rating in ratings.items():
        job=jobs[key]
        if job['kind']=='fixed_image_calibration':fixed[(job['instance'],job['control'])].append((job,rating))
        else:
            record=models[key]
            if record['condition']!='structured' or record['view_condition']!='active' or record['repeat']!=1:continue
            frontier[(record['instance'][0],record['model'])].append(rating)
            if record['dimensions']['strict_success']==0 and rating['V'] is not None and rating['V']>=.75:
                disagreements.append(dict(id=key,V=rating['V'],coverage=rating['coverage'],failure_labels=record['failure_labels']))
    calibration=[]
    for (instance,control),values in sorted(fixed.items()):
        if len({tuple(j['image_sha256']) for j,_ in values})!=1:raise ValueError('Fixed-output repeats use different images')
        calibration.append(dict(instance=instance,control=control,ratings=len(values),**observed_stats([r['V'] for _,r in values]),mean_coverage=statistics.mean(r['coverage'] for _,r in values)))
    primary=[dict(task=task,model=model,outputs=len(rows),**observed_stats([r['V'] for r in rows]),mean_coverage=statistics.mean(r['coverage'] for r in rows),variance_scope='Across different primary outputs, not repeated ratings of one output') for (task,model),rows in sorted(frontier.items())]
    controls=defaultdict(dict)
    for row in calibration:controls[row['instance']][row['control']]=row['mean']
    ordering=[]
    for instance,values in sorted(controls.items()):
        gold,safe,stale=[values.get(name) for name in ('gold','safe_intermediate','stale_or_stationary')]
        observed=all(value is not None for value in (gold,safe,stale))
        ordering.append(dict(instance=instance,gold=gold,safe_intermediate=safe,stale_or_stationary=stale,weak_ordering_holds=(gold>=safe>=stale) if observed else None,gold_minus_stale=gold-stale if gold is not None and stale is not None else None))
    return dict(complete=True,judge=plan['judge'],fixed_image_calibration=calibration,visual_control_ordering=ordering,primary=primary,visually_high_functional_failures=disagreements,scope='Uncertain criteria remain missing, with coverage reported. Fixed-image repeat variance is separate from across-output variance. Weak visual-control ordering gold>=safe_intermediate>=stale is a validity diagnostic, not a primary acceptance gate or proof of hidden functional quality. A high auxiliary score on functional failure indicates limited visual validity, not primary evaluator failure. No causal bias estimate follows from a single judge family.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--judge',type=Path,required=True);parser.add_argument('--performance',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    result=analyze(json.loads((a.judge/'plan.json').read_text()),json.loads((a.judge/'complete.json').read_text()),json.loads(a.performance.read_text()));a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
