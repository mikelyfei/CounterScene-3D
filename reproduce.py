"""Offline release verification, table reconstruction and optional native replay."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def bound(root, name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Release path escapes its root')
    return path


def verify(root, files):
    for name,digest in files.items():
        if hashlib.sha256(bound(root,name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Release content differs: '+name)


def scoring_signature(final):
    fields={'success','satisfied','total','physical','violated_types','mirror_errors','locked_moved','operation_checks','hard_ok','hard_satisfied','hard_total','preserved','preserved_total','soft_feasible_ok','soft_feasible_total','conflict_tp','conflict_fp','conflict_fn','edit','gold_edit','collateral_moves','hidden_mean_iou','seen_mean_iou','overfit_gap','metric_centres_ok','metric_centre_errors_m','waypoints_ok','waypoints','segments_ok','segments','limits_ok','payload_ok','joint_travel_deg','min_certified_clearance_m','true_satisfied','overtrust','conflicts_resolved','conflict_count','clean_disturbed','detect_tp','detect_fp','detect_fn','forced_support_repairs','unnecessary_clean_disturbed'}
    result={k:final[k] for k in ('success','execution_valid','native_schema_valid','stages_completed','stage_executions','stage_successes','stages_required') if k in final}
    result['stages']=[{k:v for k,v in stage.items() if k in fields} for stage in final.get('stage_results',[])]
    return result


def same_measurements(left,right):
    if isinstance(left,bool) or isinstance(right,bool):return type(left)==type(right) and left==right
    if isinstance(left,(int,float)) and isinstance(right,(int,float)):return math.isclose(left,right,rel_tol=1e-9,abs_tol=1e-9)
    if isinstance(left,dict) and isinstance(right,dict):return left.keys()==right.keys() and all(same_measurements(left[k],right[k]) for k in left)
    if isinstance(left,list) and isinstance(right,list):return len(left)==len(right) and all(same_measurements(a,b) for a,b in zip(left,right))
    return left==right


def reward_values(final,stage_reward):
    required=final['stages_required'];stages=final.get('stage_results',[])
    if type(required) is not int or required<1 or len(stages)>required:raise ValueError('Invalid reward stage coverage')
    rewards=[stage_reward(stage) for stage in stages]
    return dict(R_det=sum(row['R_det'] for row in rewards)/required,R_shaped=sum(row['R_shaped'] for row in rewards)/required,physical_stage_fraction=sum(row['physical_gate'] for row in rewards)/required)


def load_module(path):
    spec=importlib.util.spec_from_file_location('published_'+path.stem,path);module=importlib.util.module_from_spec(spec)
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),module.__dict__)
    return module


def reconstruct_studies(root,studies,performance):
    output={}
    if 'generation_variance' in performance:
        expected={(row['instance'],row['model']):row for row in performance['generation_variance']};groups=defaultdict(list)
        for row in performance['individual']:
            if row['condition']=='structured' and row['view_condition']=='active':groups[(row['instance'],row['model'])].append(row)
        panels=[]
        for key,values in sorted(groups.items()):
            if len(values)<2:continue
            if key not in expected or len(values)!=expected[key]['expected_generations']:raise ValueError('Generation panel coverage differs')
            row=dict(instance=key[0],model=key[1],generations=len(values),expected_generations=expected[key]['expected_generations'])
            for metric in values[0]['dimensions']:
                observed=[float(value['dimensions'][metric]) for value in values if value['dimensions'][metric] is not None]
                row.update({metric+'_mean':statistics.mean(observed) if observed else None,metric+'_n':len(observed),metric+'_population_variance':statistics.pvariance(observed) if observed else None})
            panels.append(row)
        if not same_measurements(panels,performance['generation_variance']):raise ValueError('Recomputed generation variability differs')
        output['participant_generation_variance']=panels
    if 'native_repeatability' in studies:
        reward=load_module(root/'reward.py');samples=[]
        for case in studies['native_repeatability']['samples']:
            evidence=case['scoring_evidence']
            if len(evidence)!=case['runs'] or case['runs']<2:raise ValueError('Native repeatability coverage differs')
            if any(not same_measurements(scoring_signature(evidence[0]),scoring_signature(row)) for row in evidence[1:]):raise ValueError('Native rerun measurements differ')
            rewards=[reward_values(row,reward.stage_reward) for row in evidence]
            variances={key:statistics.pvariance(row[key] for row in rewards) for key in ('R_det','R_shaped')}
            if not same_measurements(rewards,case['rewards']) or not same_measurements(variances,case['reward_population_variance']):raise ValueError('Native replay reward statistics differ')
            samples.append(dict(instance=case['instance'],model=case['model'],runs=case['runs'],reward_population_variance=variances))
        output['native_repeatability']=samples
    if 'auxiliary_protocol' in studies:
        visual=load_module(root/'analysis/visual_judge.py')
        for row in studies['auxiliary_judge']['ratings']:
            if row.get('status')=='no_executed_artifact':continue
            if not same_measurements(visual.rating_summary(row),{key:row[key] for key in ('V','coverage','uncertain')}):raise ValueError('Visual rating disagrees with criterion evidence')
        result=visual.analyze(studies['auxiliary_protocol'],studies['auxiliary_judge'],performance)
        if not same_measurements(result,studies['auxiliary_analysis']):raise ValueError('Recomputed auxiliary analysis differs')
        output['auxiliary_analysis']=result
    if 'probe_exact_replay' in studies:
        probe=load_module(root/'analysis/probe_selection.py');pools=[]
        for pool in studies['probe_exact_replay']['pools']:
            result=probe.compare_pool(pool['candidates'])
            if not same_measurements(result,{key:pool[key] for key in result}):raise ValueError('Recomputed public-reward selection differs')
            pools.append(dict(instance=pool['instance'],**result))
        output['probe_selection']=pools
    return output


def table(records):
    groups=defaultdict(list)
    for record in records:
        if record['repeat']==1 and record['condition']=='structured' and record['view_condition']=='active':
            groups[(record['instance'][0],record['model'])].append(record)
    rows=[]
    for (family,model),values in sorted(groups.items()):
        row=dict(task=family,model=model,instances=len(values),expected_instances=6,failure_labels=','.join(sorted({label for r in values for label in r['failure_labels']})))
        for metric in values[0]['dimensions']:
            observed=[float(r['dimensions'][metric]) for r in values if r['dimensions'][metric] is not None]
            row.update({metric+'_mean':statistics.mean(observed) if observed else None,metric+'_n':len(observed),metric+'_population_variance':statistics.pvariance(observed) if observed else None})
        for metric in ('R_det','R_shaped'):
            scores=[[s[metric] for s in r['numerical_rescores']] for r in values]
            if any(len(s)!=3 or any(v is None for v in s) for s in scores):raise ValueError('Missing three-repeat numerical evidence')
            row.update({metric+'_rescore_mean':statistics.mean(statistics.mean(s) for s in scores),metric+'_rescore_within_output_variance_mean':statistics.mean(statistics.pvariance(s) for s in scores),metric+'_rescore_outputs':len(scores)})
        row['rescore_repeats_per_output']=3
        row['rescore_scope']='Numerical rescoring of retained native measurements; no fresh render or model generation'
        rows.append(row)
    return rows


def with_auxiliary(rows,studies):
    judge=studies.get('auxiliary_analysis',{});primary={(r['task'],r['model']):r for r in judge.get('primary',[])}
    output=[]
    for row in rows:
        aux=primary.get((row['task'],row['model']),{});fixed=[r['sample_variance'] for r in judge.get('fixed_image_calibration',[]) if r['instance'].startswith(row['task']+'-') and r['sample_variance'] is not None]
        native=[case for case in studies.get('native_repeatability',{}).get('samples',[]) if case['instance'][0]==row['task'] and case['model']==row['model']]
        native_variance={key+'_native_replay_variance_mean':statistics.mean(case['reward_population_variance'][key] for case in native) if native else None for key in ('R_det','R_shaped')}
        output.append(dict(row,**native_variance,native_replay_outputs=len(native),native_replay_runs_per_output=native[0]['runs'] if native else None,V_aux_mean=aux.get('mean'),V_aux_n=aux.get('n'),V_aux_coverage=aux.get('mean_coverage'),V_aux_across_output_sample_variance=aux.get('sample_variance'),V_aux_fixed_control_variance_mean=statistics.mean(fixed) if fixed else None,V_aux_variance_scope='Fixed-control image repeats pooled within task; auxiliary evidence, separate from participant generation variance'))
    return output


def replay(package,instance,actions,out,expected,condition='structured',view_condition='active',expected_signature=None):
    out.mkdir(parents=True,exist_ok=False)
    result=out/'result.json'
    command=[sys.executable,str(package/'scorer.py'),'--instance',instance,'--actions',str(actions),'--artifacts',str(out/'artifacts'),'--out',str(result),'--condition',condition,'--view-condition',view_condition]
    process=subprocess.run(command)
    if not result.exists():raise RuntimeError('Native replay did not produce a result')
    final=json.loads(result.read_text(encoding='utf-8'))['result'];actual=final['success']
    if actual!=expected or process.returncode!=(0 if actual else 1):raise RuntimeError('Native replay outcome differs')
    if expected_signature is not None and not same_measurements(scoring_signature(final),expected_signature):raise RuntimeError('Native replay scoring measurements differ')
    return final


def run(args):
    root=args.root.resolve()
    release=json.loads((root/'release_manifest.json').read_text(encoding='utf-8'))
    verify(root,release['files'])
    performance=json.loads((root/'performance.json').read_text(encoding='utf-8'))
    if not performance['complete']:raise ValueError('Final release requires completed evaluation')
    rows=table(performance['individual'])
    if rows!=performance['primary']:raise ValueError('Reconstructed task/model table differs')
    if len(rows)!=10 or any(r['instances']!=6 for r in rows):raise ValueError('Incomplete primary coverage')
    studies=json.loads((root/'studies.json').read_text(encoding='utf-8'));reconstructed_studies=reconstruct_studies(root,studies,performance)
    rows=with_auxiliary(rows,studies);reward=load_module(root/'reward.py')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'study_reconstruction.json').write_text(json.dumps(reconstructed_studies,indent=2),encoding='utf-8')
    with (out/'benchmark_results.tsv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,list(dict.fromkeys(k for r in rows for k in r)),delimiter='\t');writer.writeheader()
        writer.writerows({k:'NA' if v is None else v for k,v in r.items()} for r in rows)
    sessions=root/'sessions'
    replay_index=[]
    for index,record in enumerate(performance['individual']):
        folder=bound(sessions,record['id']);manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        verify(folder,{'actions.jsonl':manifest['actions_sha256'],'scoring_evidence.json':manifest['scoring_evidence_sha256']})
        if manifest['source_result_sha256']!=record['record_sha256'] or manifest['dimensions']!=record['dimensions']:raise ValueError('Session and performance provenance differ')
        evidence=json.loads((folder/'scoring_evidence.json').read_text(encoding='utf-8'))
        if not same_measurements(scoring_signature(evidence),manifest['scoring_signature']):raise ValueError('Reward evidence and native scoring signature differ')
        measured=reward_values(evidence,reward.stage_reward)
        if any(not same_measurements(value,record['dimensions'][key]) for key,value in measured.items() if key in record['dimensions']):raise ValueError('Recomputed reward differs from published result')
        if args.replay:
            # Short numbered directories preserve Blender's Windows path headroom.
            destination=out/'model_replay'/f'{index:03d}'
            replay_index.append(dict(id=record['id'],directory=destination.relative_to(out).as_posix()))
            (out/'model_replay_index.json').write_text(json.dumps(replay_index,indent=2),encoding='utf-8')
            fresh=replay(root/'tasks'/record['instance'][0],record['instance'],folder/'actions.jsonl',destination,manifest['success'],record['condition'],record['view_condition'],manifest['scoring_signature'])
            if not same_measurements(reward_values(fresh,reward.stage_reward),measured):raise ValueError('Fresh native reward differs from retained evidence')
    if args.replay:
        for family in 'ABCDE':
            package=root/'tasks'/family
            manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'));verify(package,manifest['files'])
            for item in manifest['instances']:
                instance=item['instance'];replay(package,instance,package/'evaluator'/instance/'gold_actions.jsonl',out/'gold_replay'/instance,True)
    print('Verified release hashes, recomputed rewards and reconstructed ten task/model rows. '+('Native model and gold replays matched.' if args.replay else 'Use --replay for native execution; retained-evidence reward reconstruction is not fresh native scoring.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path,default=Path('reproduced'))
    parser.add_argument('--replay',action='store_true',help='Execute all retained actions and gold using external Blender; no API calls')
    run(parser.parse_args())
