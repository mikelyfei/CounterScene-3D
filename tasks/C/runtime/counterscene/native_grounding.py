"""Private reconstruction metrics from visible, actual-triangle ray hits."""
import math


def visible_view_sets(task,mesh):
    import itertools
    public=['reference']+task['queryable_views'];coverage={}
    for oid in mesh['instances']:
        coverage[oid]={name:view['visible_ids'].count(oid)/view['unoccluded_pixels'][oid] if view['unoccluded_pixels'][oid]>=8 else 0 for name,view in mesh['view_evidence'].items() if name in public}
    sufficient=[]
    for extra in itertools.combinations(task['queryable_views'],task['query_budget']):
        names=['reference',*extra]
        if all(max(coverage[oid][name] for name in names)>=.3 for oid in coverage):sufficient.append(names)
    return coverage,sufficient


def compare_views(reference,candidate):
    for key in ('camera','width','height'):
        if key not in reference or reference[key]!=candidate.get(key):raise ValueError('Native camera/raster mismatch')
    length=reference['width']*reference['height']
    for view in (reference,candidate):
        if len(view.get('visible_ids',[]))!=length or len(view.get('visible_depth_m',[]))!=length:raise ValueError('Complete actual-triangle pixel evidence required')
    gold,cand=reference['visible_ids'],candidate['visible_ids']
    names=(set(gold)|set(cand))-{None}
    if not names:raise ValueError('No measurable native foreground')
    iou={}
    for name in sorted(names):
        intersection=sum(a==name and b==name for a,b in zip(gold,cand))
        union=sum(a==name or b==name for a,b in zip(gold,cand))
        iou[name]=intersection/union
    errors=[abs(a-b) for name,other,a,b in zip(gold,cand,reference['visible_depth_m'],candidate['visible_depth_m']) if name is not None and name==other and a is not None and b is not None]
    if any(not math.isfinite(value) for value in errors):raise ValueError('Nonfinite native depth evidence')
    return dict(object_iou=iou,mean_object_iou=sum(iou.values())/len(iou),depth_abs_err=sum(errors)/len(errors) if errors else None)


def score_grounding(task,private,placements,candidate,target,held):
    from .native_scoring import check_native_scene
    report=check_native_scene(task,placements,candidate)
    if not report['valid_output']:return dict(success=False,reason='invalid_output',geometry=report)
    target_report=check_native_scene(task,private['gold'],target)
    if not target_report.get('all_satisfied'):return dict(success=False,reason='invalid_native_target_evidence')
    try:
        views={name:compare_views(target['view_evidence'][name],candidate['view_evidence'][name]) for name in set(private['hidden_cameras'])|set(held)}
        errors={}
        for oid in target['instances']:
            a=target['instances'][oid]['world_bounds'];b=candidate['instances'][oid]['world_bounds']
            errors[oid]=math.dist([(lo+hi)/2 for lo,hi in a],[(lo+hi)/2 for lo,hi in b])
        hidden=sum(views[name]['mean_object_iou'] for name in private['hidden_cameras'])/len(private['hidden_cameras'])
        seen=sum(views[name]['mean_object_iou'] for name in held)/len(held) if held else None
    except (KeyError,ValueError,ZeroDivisionError) as error:
        return dict(success=False,reason='missing_native_view_evidence',error=str(error))
    return dict(success=report['all_satisfied'] and hidden>=.8 and all(d<=.05 for d in errors.values()),geometry=report,
                hidden_mean_iou=hidden,seen_mean_iou=seen,overfit_gap=seen-hidden if seen is not None else None,
                metric_centres_ok=all(d<=.05 for d in errors.values()),metric_centre_errors_m=errors,
                per_view={name:row['mean_object_iou'] for name,row in views.items()},
                evidence_scope='Actual native-triangle visible masks at 160x120 and actual bounds centres; no proxy box image score')
