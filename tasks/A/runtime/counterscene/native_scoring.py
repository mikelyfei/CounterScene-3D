"""Native support/nonintersection verifier; incomplete evidence fails closed."""
from .spatial_suite import core


def check_native_scene(task,placements,mesh):
    report=core.check_scene(task,placements)
    if not report['valid_output']:return report
    required={o['id'] for o in task['objects']}
    if set(mesh.get('instances',{}))!=required:
        return dict(valid_output=False,output_problems=['Native instance coverage mismatch'],constraints=[],physical=[],all_satisfied=False)
    from .asset_scene import instance_specs
    try:specs=instance_specs(task,placements)
    except (ValueError,KeyError,TypeError):
        return dict(valid_output=False,output_problems=['Invalid native instance transforms or scale'],constraints=[],physical=[],all_satisfied=False)
    import itertools
    expected={frozenset(pair) for pair in itertools.combinations(required,2)}
    pairs=mesh.get('pairs',[])
    actual=[frozenset(p.get('objects',[])) for p in pairs]
    support_ids=[p.get('object') for p in mesh.get('support_witnesses',[])]
    if len(actual)!=len(expected) or set(actual)!=expected or len(support_ids)!=len(required) or set(support_ids)!=required:
        return dict(valid_output=False,output_problems=['Incomplete or duplicate native pair/support evidence'],constraints=[],physical=[],all_satisfied=False)
    for oid,spec in specs.items():
        actual=mesh['instances'][oid]
        matched=(actual.get('asset')==spec['asset'] and actual.get('position')==spec['position'] and abs(actual.get('yaw',-999)-spec['yaw'])<1e-9 and abs(actual['world_bounds'][2][1]-actual['world_bounds'][2][0]-spec['height'])<1e-5)
        if not matched:return dict(valid_output=False,output_problems=['Native evidence is not bound to candidate transform: '+oid],constraints=[],physical=[],all_satisfied=False)
    supports={r['object']:r for r in mesh.get('support_witnesses',[])}
    def covered(oid,host):return host in supports.get(oid,{}).get('footprint_supports',[])
    physical=[r for r in report['physical'] if r['check'] not in ('supported','approved_full_footprint_support','no_interpenetration','inside_room')]
    policy=task.get('scene_policy',{})
    for oid,desc in mesh['instances'].items():
        bounds=desc['world_bounds']
        if any(b[0]<-core.TOL or b[1]>task['room'][i]+core.TOL for i,b in enumerate(bounds)):physical.append(dict(check='inside_room',objects=[oid],native_evidence=True))
        witness=supports.get(oid,{})
        floor=witness.get('status')=='floor_contact'
        hosts=witness.get('footprint_supports',[])
        if not floor and not hosts:physical.append(dict(check='supported',objects=[oid],native_evidence='actual full footprint'))
        if not floor and oid in policy.get('surface_objects',[]):
            allowed=set(policy.get('flat_supports',[]))
            if oid in policy.get('shelf_items',[]):allowed.update(policy.get('shelf_supports',[]))
            if not set(hosts)&allowed:physical.append(dict(check='approved_full_footprint_support',objects=[oid],native_evidence=True))
    for pair in mesh.get('pairs',[]):
        if pair['status'] not in ('disjoint','within_axis_tolerance','certified_support_contact'):physical.append(dict(check='no_interpenetration',objects=pair['objects'],native_status=pair['status']))
    locked_moved=set(core.edit_cost(task.get('initial',{}),placements)['moved'])&set(task.get('locked',[]))
    for oid in sorted(locked_moved):physical.append(dict(check='locked_instance_unchanged',objects=[oid],native_evidence=True))
    rows=[]
    for c,old in zip(task['constraints'],report['constraints']):
        row=dict(old)
        if c['type']=='on_top_of':
            row['satisfied']=covered(*c['args']);row['evidence']='Actual projected footprint covered on named host within height tolerance'
        elif c['type']=='on_shelf':
            child,host=c['args'];bottom=mesh['instances'][child]['world_bounds'][2][0]
            base=mesh['instances'][host]['world_bounds'][2][0];height=mesh['instances'][host]['world_bounds'][2][1]-base
            boards=c.get('range',list(range(4)))
            row['satisfied']=covered(child,host) and any(abs(bottom-(base+k*(height-core.SHELF_T)/3+core.SHELF_T))<=core.TOL for k in boards)
            row['evidence']='Actual host coverage plus named scored board height'
        elif c['type'] in ('occludes','appears_left_of','appears_larger_than'):
            view=mesh.get('reference_view',{});a,b=c['args']
            if view.get('camera')!=task.get('camera'):
                row['satisfied']=False;row['native_evidence_missing']=True
            elif c['type']=='appears_left_of':
                centres=view['projected_centres'];row['value']=centres[b][0]-centres[a][0];row['satisfied']=row['value']>0
            elif c['type']=='appears_larger_than':
                pixels=view['unoccluded_pixels'];row['value']=pixels[a]/pixels[b] if pixels[b] else None;row['satisfied']=row['value'] is not None and row['value']>1
            else:
                row['value']=view['pair_occlusion'][a][b];row['satisfied']=row['value'] is not None and row['value']>=.5
            row['evidence']='Actual native mesh reference-camera measurements'
        elif c['type']=='touching':
            pair=next((p for p in mesh.get('pairs',[]) if set(p['objects'])==set(c['args'])),{})
            distance=pair.get('surface_distance_upper_bound_m')
            row['satisfied']=distance is not None and distance<=core.TOL and pair.get('status') in ('disjoint','within_axis_tolerance','certified_support_contact')
            row['native_evidence_missing']=distance is None
        rows.append(row)
    return dict(valid_output=True,output_problems=[],constraints=rows,physical=physical,all_satisfied=not physical and all(r['satisfied'] for r in rows),limits=['Static stability not simulated','Native image relations fail closed until actual-view evidence is supplied'])
