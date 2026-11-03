"""Frozen, generated task definitions. Private solutions never enter observations."""
from pathlib import Path
import copy
import json
from .program import literal_program

ASSETS=Path(__file__).parent/'assets'


def spec(task,positions,path=None,side='left',phase='initial'):
    return dict(schema_version='1.0',task_id=task['id'],instance_id=task['instance_id'],phase=phase,
                placements=[dict(id=k,center_m=v,yaw_deg=0.) for k,v in positions.items()],
                joints=[dict(id='cabinet_door',hinge_side=side)] if 'door' in task else [],path_xy_m=path or [])


def definitions():
    catalog={
        'red':dict(size_m=[.26,.22,.24],color='#ef6b65',support_z_m=.8),
        'blue':dict(size_m=[.24,.24,.32],color='#659bea',support_z_m=.8),
        'gold':dict(size_m=[.22,.18,.18],color='#efc46d',support_z_m=.8)}
    t1=dict(id='t1',instance_id='frozen_001',title='Occluded workbench',bounds_m=[1.6,.8],
            catalog=catalog,fixed=[],support_bounds_m=[0,0,1.6,.8],clearance_m=.02,max_steps=8,
            relations=[['red','left_of','blue'],['blue','left_of','gold'],['red','behind','blue']],
            free_zone=[1.3,0,1.6,.8],expected_calls='4-7',
            capabilities=['visual grounding','spatial planning','execution','support repair'])
    gold1=spec(t1,{'red':[.25,.58,.92],'blue':[.7,.24,.96],'gold':[1.1,.5,.89]})
    start1=spec(t1,{'red':[.9,.25,.97],'blue':[.6,.25,.96],'gold':[1.42,.5,.89]})
    cat2={'box_a':dict(size_m=[.2,.2,.3],color='#659bea',support_z_m=0),
          'box_b':dict(size_m=[.24,.24,.4],color='#efc46d',support_z_m=0)}
    door=dict(size_m=[.6,.04,.7],hinges={'left':[1.,1.],'right':[1.6,1.]},target_side='left')
    # Open frame behind hinge plane, physically checked rather than exempted.
    frame=dict(id='frame',size_m=[.8,.1,.8],center_m=[1.3,.82,.4],yaw_deg=0,color='#64748b')
    t2=dict(id='t2',instance_id='frozen_001',title='The door that jams halfway',bounds_m=[3.,2.4],
            catalog=cat2,fixed=[frame],door=door,clearance_m=.02,max_steps=12,
            support_bounds_m=[0,0,3,2.4],placement_region=[.15,.15,2.85,2.25],expected_calls='6-10',
            capabilities=['reference grounding','kinematic reasoning','feedback repair'])
    gold2=spec(t2,{'box_a':[2.25,1.55,.15],'box_b':[2.25,.5,.2]})
    # Both endpoints are clear; at 45 degrees the door crosses box_a.
    start2=spec(t2,{'box_a':[1.36,1.36,.15],'box_b':[2.25,.5,.2]})
    t3=copy.deepcopy(t2)
    t3.update(id='t3',title='Replan after the world changes',bounds_m=[4.,3.],
              support_bounds_m=[0,0,4,3],placement_region=[.15,.15,3.85,2.85],
              start=[.3,.4],goal=[3.7,.4],vehicle_radius_m=.2,path_clearance_m=.03,
              max_steps=16,expected_calls='9-14',event_seed=271828,
              event={'id':'equipment_delivery_271828','object':dict(id='new_equipment',
                     size_m=[.55,.55,.8],center_m=[2.,.4,.4],yaw_deg=0,color='#ef6b65')},
              capabilities=['joint planning','visual change detection','replanning','regression checking'])
    t3['fixed'].append(dict(id='workbench',size_m=[1.,.45,.8],center_m=[3.2,2.6,.4],yaw_deg=0,color='#64748b'))
    gold3=spec(t3,{'box_a':[2.7,1.9,.15],'box_b':[3.5,1.8,.2]},
               [[.3,.4],[.5,2.1],[2.,2.1],[2.1,1.25],[3.7,1.],[3.7,.4]],phase='post_event')
    # Simpler route passes below the cabinet, above the delivered obstacle.
    gold3['path_xy_m']=[[.3,.4],[.5,2.1],[2.,2.1],[2.1,1.25],[3.7,1.],[3.7,.4]]
    start3=spec(t3,{'box_a':[2.7,1.9,.15],'box_b':[3.5,1.8,.2]},[[.3,.4],[3.7,.4]])
    items=[(t1,start1,gold1),(t2,start2,gold2),(t3,start3,gold3)]
    for task,_,_ in items:task['observation_protocol']='hinge-pin-v2'
    return items


PROMPTS={
 't1':'''Repair the three colored objects to match the left/right and front/back relationships in the reference views, using world coordinates (+Y is behind). Preserve identity and size. All footprints must lie on the 1.6 x 0.8 m workbench, bottoms at z=0.8 m within 0.002 m. Keep 0.02 m between objects. The hatched pick-up strip x=1.3..1.6 must be empty. Use extra views to resolve depth; the front view alone is ambiguous. Do not move the workbench. Exact target positions are not required, but visible edge ordering needs at least 0.02 m separation.''',
 't2':'''Match the hinge side shown in the reference image. Keep all catalog objects and dimensions. Repair the door and relocate the two boxes inside the declared placement region. Boxes must rest on the floor. The door opens from 0 to 90 degrees and must maintain at least 0.03 m from every box and every fixed frame component at EVERY intermediate angle. Endpoints alone are insufficient. Left hinge opens toward +Y; right hinge opens clockwise toward +Y. Fixed geometry cannot move.''',
 't3':'''Arrange the workshop and plan a continuous ground-level path from A to B for a radius 0.20 m disk, plus 0.03 m clearance. All path segments and the full disk must stay inside the room and avoid every obstacle, including the door at any opening angle. Preserve floor support, object spacing, placement regions and the reference hinge side. First build a feasible initial candidate and checkpoint it. One equipment delivery then changes the map; inspect the new observation and revise your program. Final submission requires a post-event artifact. The checkpoint event happens once and repeated checkpoint calls are idempotent.'''
}

CONTRACT='''Return exactly one JSON action per turn: {"tool":"build_scene","source":"def build(public_params):\\n    return {...}"}. Other tools: inspect_reference(view=front/top/side/iso), inspect_output(artifact_id,view,angle), check_public(artifact_id), checkpoint(artifact_id), submit(artifact_id). Use only own artifact IDs. Angles: 0..90. SceneSpec keys exactly: schema_version="1.0", task_id, instance_id, phase (initial/post_event), placements=[{id,center_m:[x,y,z],yaw_deg}], joints=[{id:"cabinet_door",hinge_side:"left"|"right"}] (empty for t1), path_xy_m (empty for t1/t2, 2-12 xy points for t3). Immutable catalog supplies sizes/materials. Python subset: one build(public_params) function, assignments, return, if/else, numeric arithmetic, comparisons, indexing, list/dict literals, one-level list comprehensions, range/len/min/max/abs/round. No imports, attributes, mutation, loops or external calls. Use public_params["phase"] for phase. Tools consume budget even when invalid. No score for number of calls or explanatory text.'''


def public(task,phase='initial'):
    # Explicit publication boundary: future private fields must not silently
    # become agent observations when a task definition gains audit metadata.
    fields={'id','instance_id','title','bounds_m','catalog','fixed','door','clearance_m',
            'max_steps','support_bounds_m','placement_region','expected_calls','capabilities',
            'start','goal','vehicle_radius_m','path_clearance_m','free_zone','observation_protocol'}
    out={k:copy.deepcopy(v) for k,v in task.items() if k in fields}
    if 'door' in out:
        out['door'].pop('target_side',None)
    out['phase']=phase
    out['task_id']=task['id']
    if 'door' in task:out['visual_legend']='The pink HINGE pin marks the actual hinge axis in reference and output images.'
    if task.get('reference_motion_probe_required'):
        out['visual_legend']='The closed reference intentionally conceals the hinge axis. Inspect a non-zero reference door angle (0..90 degrees) to determine its installation side. The pink HINGE pin appears on open poses and candidate outputs.'
    if task.get('reference_views') and len(task['reference_views'])<4:
        out['reference_views_available']=list(task['reference_views'])
    if task.get('reference_max_angle',90)==0:
        out['visual_legend']='The closed reference intentionally conceals the hinge axis. In this condition reference images are available only at 0 degrees. The pink HINGE pin appears on candidate outputs.'
    if task.get('diagnostic_feedback'):out['diagnostic_feedback']=task['diagnostic_feedback']
    if phase=='post_event':
        if task.get('event_visibility')=='image_only' and not task.get('post_event_images',True):
            out['event_observation']='A delivered obstacle has been added. In this condition its pose is provided neither numerically nor in images; public checks still evaluate all new geometry.'
        elif task.get('event_visibility')=='image_only':
            out['event_observation']='A delivered obstacle is visible in the updated trusted images. Its pose is not provided numerically. All new geometry participates in collision checks.'
        else:out['fixed'].append(copy.deepcopy(task['event']['object']))
    return out


def task_prompt(task):
    text=PROMPTS[task['id']]
    if task.get('reference_views') and len(task['reference_views'])<4:
        text=text.replace(' Use extra views to resolve depth; the front view alone is ambiguous.','')
        text+=' In this condition inspect_reference provides only these views: '+', '.join(task['reference_views'])+'.'
    if task.get('reference_max_angle',90)==0:
        return text+' The closed reference conceals hinge identity. In this condition inspect_reference accepts only angle 0.'
    if task.get('diagnostic_feedback')=='minimal':
        text+=' In this condition check_public reports only pass/fail per criterion, without measured distances.'
    if task.get('post_event_images') is False:
        text+=' In this condition images are not shown after the equipment delivery.'
    if task.get('reference_motion_probe_required'):
        text+=' The initial closed reference conceals hinge identity. inspect_reference accepts an optional angle in degrees; request a non-zero angle to observe the reference mechanism. This is an information-gathering operation, not a required call-count bonus.'
    if task.get('reference_motion_disabled'):
        text+=' In this explicit observation-control condition, only closed reference poses are available: nonzero reference-angle requests are disabled. All candidate inspection and physical checks remain available.'
    return text


def load(task_id):
    for task,start,gold in definitions():
        if task['id']==task_id:return task,start,gold
    raise ValueError(f'unknown task: {task_id}')


def export(root):
    root=Path(root)
    for task,start,gold in definitions():
        dest=root/task['id']
        for name in ('public','private'):(dest/name).mkdir(parents=True,exist_ok=True)
        (dest/'prompt.md').write_text('# '+task['title']+'\n\n'+PROMPTS[task['id']]+'\n\n'+CONTRACT+'\n',encoding='utf-8')
        for name,value in [('manifest.json',public(task)),('start_scene.json',start)]:
            (dest/'public'/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
        (dest/'public'/'starter.py').write_text(literal_program(start),encoding='utf-8')
        (dest/'private'/'oracle.json').write_text(json.dumps(task,indent=2),encoding='utf-8')
        (dest/'private'/'gold.py').write_text(literal_program(gold),encoding='utf-8')
        (dest/'private'/'gold.json').write_text(json.dumps(gold,indent=2),encoding='utf-8')
        (dest/'scorer.py').write_text('"""Run from repository: python -m counterscene score --task '+task['id']+' --scene scene.json"""\nfrom counterscene.scoring import evaluate\n',encoding='utf-8')
        (dest/'feasibility_notes.md').write_text('# Feasibility\n\nGold uses fixed, generous support and clearance margins; all dimensions are in metres. Run `python -m counterscene reproduce` to recompute every criterion and inspect the generated front/top/side/isometric views. T2 initial box intersects the door near 45 degrees while both endpoints clear. T3 gold path goes above the delivered equipment and below the cabinet. The continuous sweep proof subtracts a Lipschitz motion bound; the disk path checks complete segments. This is geometric feasibility, not dynamic stability.\n',encoding='utf-8')
        (dest/'hackability.md').write_text('# Threat model\n\nD (bounded): misidentify objects or mirror hinge; fixed identity colors and reference views, but perception remains uncertain. G (bounded): hide floating boxes or diagonal overlap; full footprints, height intervals and support checks prevent these within cuboid assumptions. F (hard within representation): endpoint-only door tests and waypoint-only paths are vulnerable; continuous bounds and segment distances close these attacks, but arbitrary meshes and physics are outside scope. V (soft/vibes): lighting, occlusion and judge drift can manipulate visual opinions. Images must originate from trusted reconstruction; self-reported images/scores are rejected. E/I: no imports, deletion, scaling or arbitrary CPython; interpreted subset is deliberately narrow.\n',encoding='utf-8')
