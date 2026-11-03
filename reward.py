"""Explicit native reward components; infrastructure and inapplicable metrics stay distinct."""
import math


def fraction(numerator,denominator):return numerator/denominator if denominator else None


def stage_reward(stage):
    geometry=stage.get('geometry',{});scope=stage.get('stage_scope')
    valid=stage.get('reason') not in ('invalid_output','no_submission','local_evaluation_failure') and geometry.get('valid_output',True)
    issues=scope['violations'] if scope is not None else stage.get('physical',geometry.get('physical',[]))
    motion=stage.get('native_report');motion_safe=True
    if motion is not None:
        arm=motion.get('arm_report',{});payload=arm.get('payload') or {}
        rows=motion.get('segments',[])+motion.get('fixed',[])+arm.get('segments',[])+arm.get('limits',[])+payload.get('carry_segments',[])+payload.get('arm_vs_balls',[])
        motion_safe=bool(arm.get('valid_output') and motion.get('segments') and arm.get('segments') and all(row.get('ok',False) for row in rows) and stage.get('limits_ok',False))
    safe=bool(valid and not issues and not stage.get('locked_moved') and motion_safe)
    components=dict(functional_success=float(bool(stage.get('success'))))
    for name,numerator,denominator in [('constraints','satisfied','total'),('hard_constraints','hard_satisfied','hard_total'),('feasible_soft','soft_feasible_ok','soft_feasible_total'),('true_constraints','true_satisfied','total'),('waypoints','waypoints_ok','waypoints'),('motion_segments','segments_ok','segments'),('preservation','preserved','preserved_total')]:
        if numerator in stage and denominator in stage:
            value=fraction(stage[numerator],stage[denominator])
            if value is not None:components[name]=value
    for prefix,name in [('conflict','infeasibility_f1'),('detect','checker_f1')]:
        if prefix+'_tp' in stage:
            tp=stage[prefix+'_tp'];denominator=2*tp+stage[prefix+'_fp']+stage[prefix+'_fn']
            if denominator:components[name]=2*tp/denominator
    for key in ('hidden_mean_iou','metric_centres_ok','limits_ok','payload_ok'):
        if stage.get(key) is not None:components[key]=float(stage[key])
    if stage.get('operation_checks'):components['operation_targets']=sum(row['ok'] for row in stage['operation_checks'])/len(stage['operation_checks'])
    if 'unnecessary_clean_disturbed' in stage:components['clean_preservation']=1/(1+len(stage['unnecessary_clean_disturbed']))
    if any(not math.isfinite(value) or not 0<=value<=1 for value in components.values()):raise ValueError('Invalid evaluator reward component')
    return dict(R_det=float(bool(stage.get('success')) and safe),R_shaped=sum(components.values())/len(components) if safe else 0.,physical_gate=safe,components=components)


def record_reward(record):
    if not record['evaluation_eligible']:return None
    final=record['summary']['final'];stages=final.get('stage_results',[]);required=final.get('stages_required',1)
    return sum(stage_reward(stage)['R_shaped'] for stage in stages)/required
