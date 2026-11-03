"""Portable reconstruction of the learning-oriented extension from native evidence."""
import math,statistics


def equal(left,right):
    return math.isclose(left,right,rel_tol=1e-9,abs_tol=1e-9)


def pool_summary(rows):
    selected=max(range(len(rows)),key=lambda i:rows[i]['public_reward'])
    return dict(selected_index=selected,first_hidden=rows[0]['hidden_reward'],random_hidden=statistics.mean(r['hidden_reward'] for r in rows),selected_hidden=rows[selected]['hidden_reward'],selected_success=bool(rows[selected]['success']))


def reconstruct(data,stage_reward):
    original=[pool_summary(p['candidates']) for p in data['original_pools']]
    controls=[]
    for row in data['controls']:
        measured={}
        for label in ('control','baseline'):
            evidence=row[label+'_evidence'];stages=evidence['stage_results'];required=evidence['stages_required']
            value=sum(stage_reward(stage)['R_shaped'] for stage in stages)/required
            strict=int(len(stages)==required and all(s.get('success',False) for s in stages))
            if not equal(value,row['R_shaped'] if label=='control' else row['baseline_R_shaped']) or strict!=row['strict_success' if label=='control' else 'baseline_strict_success']:raise ValueError('Control reward evidence differs')
            measured[label]=dict(R_shaped=value,strict_success=strict)
        controls.append(dict(instance=row['instance'],model=row['model'],condition=row['condition'],view_condition=row['view_condition'],**measured))
    arms=[];valid=0;zero=0;count=0
    for arm in data['arms']:
        stages=[]
        for stage in arm['stages']:
            rows=stage['proposals']
            for row in rows:
                value=stage_reward(row['scoring_evidence'])['R_shaped']
                if not equal(value,row['hidden_reward']):raise ValueError('Candidate reward evidence differs')
                if bool(row['scoring_evidence'].get('success'))!=row['success']:raise ValueError('Candidate strict outcome differs')
                valid+=int(row['native_schema_valid']);zero+=int(value==0);count+=1
            stages.append(pool_summary(rows))
        value=sum(s['selected_hidden'] for s in stages)/arm['stages_required']
        success=len(stages)==arm['stages_required'] and all(s['selected_success'] for s in stages)
        if not equal(value,arm['hidden_reward']) or success!=arm['success']:raise ValueError('Episode selection differs')
        arms.append(dict(instance=arm['instance'],arm=arm['arm'],hidden_reward=value,success=success,stages=stages))
    actual=[]
    for study in data['actual_reward']:
        for row in study['candidates']:
            if not equal(stage_reward(row['scoring_evidence'])['R_shaped'],row['R_shaped']):raise ValueError('Actual-signal reward evidence differs')
        selected=max(range(len(study['candidates'])),key=lambda i:study['candidates'][i]['R_shaped']);best=study['candidates'][selected]
        if not equal(best['R_shaped'],study['selected_reward']) or best['strict_success']!=study['selected_strict_success']:raise ValueError('Actual-signal selection differs')
        actual.append(dict(study=study['study'],selected_reward=best['R_shaped'],selected_strict_success=best['strict_success']))
    baseline={key:statistics.mean(r[key] for r in original) if original else None for key in ('first_hidden','random_hidden','selected_hidden')}
    return dict(original_baseline=baseline,controls=controls,arms=arms,actual_reward=actual,native_valid=valid,candidates=count,zero_reward=zero)
