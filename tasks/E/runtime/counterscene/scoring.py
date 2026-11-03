"""Transparent criteria, measured evidence, dimension-normalized rewards."""
import math
from .contracts import reconstruct,ContractError,digest
from .geometry import footprint,zinterval,polygon_distance,solid_distance,rectangle,segment_polygon,sweep,door_object

VERSION='cuboid-2.0.0'


def evaluate(task,scene,phase='initial',visual=None):
    checks=[]
    def add(cid,dimension,passed,value,threshold,status=None):
        checks.append(dict(criterion_id=cid,dimension=dimension,passed=bool(passed),
                           measured_value=value,threshold=threshold,status=status or ('pass' if passed else 'fail')))
    try:
        scene,objects=reconstruct(task,scene,phase)
    except (ContractError,TypeError,KeyError,ValueError) as e:
        return dict(scorer_version=VERSION,status='invalid',error=str(e),E=1,I=0,D=0,G=0,F=0,V=visual,
                    R_visual=0,R_sum=0,R_safe=0,R_det=0,R_shaped_det=0,R_shaped=None if visual is None else 0,
                    success=False,checks=[],failure_labels=['INTEGRITY'])
    movable={o['id']:o for o in objects if o['id'] in task['catalog']}
    b=task['support_bounds_m']
    for name,o in movable.items():
        poly=footprint(o)
        bottom=zinterval(o)[0]
        error=abs(bottom-o['support_z_m'])
        add('support:'+name,'G',error<=.002+1e-10,error,'abs(bottom-support)<=0.002 m')
        inside=all(b[0]-1e-9<=p[0]<=b[2]+1e-9 and b[1]-1e-9<=p[1]<=b[3]+1e-9 for p in poly)
        add('bounds:'+name,'G',inside,poly,'full footprint within support bounds')
        if 'placement_region' in task:
            r=task['placement_region']
            valid=all(r[0]<=p[0]<=r[2] and r[1]<=p[1]<=r[3] for p in poly)
            add('region:'+name,'D',valid,poly,'full footprint within allowed region')
    for i,a in enumerate(objects):
        for bobj in objects[i+1:]:
            # Fixed-fixed collisions cannot be caused by the candidate.
            if a['id'] not in movable and bobj['id'] not in movable:continue
            za,zb=zinterval(a),zinterval(bobj)
            overlaps=min(za[1],zb[1])-max(za[0],zb[0])>1e-9
            gap=solid_distance(a,bobj)
            add('separation:'+a['id']+':'+bobj['id'],'G',gap>=task['clearance_m']-1e-9,
                {'distance_m':gap,'height_overlap':overlaps},f">={task['clearance_m']} m")
    for a,relation,bname in task.get('relations',[]):
        pa,pb=footprint(movable[a]),footprint(movable[bname])
        gap=min(p[0] for p in pb)-max(p[0] for p in pa) if relation=='left_of' else min(p[1] for p in pa)-max(p[1] for p in pb)
        add('relation:'+a+':'+relation+':'+bname,'D',gap>=.02-1e-9,gap,'>=0.02 m edge separation')
    if 'free_zone' in task:
        x0,y0,x1,y1=task['free_zone']
        zone=rectangle([(x0+x1)/2,(y0+y1)/2],[x1-x0,y1-y0])
        gap=min(polygon_distance(footprint(o),zone) for o in movable.values())
        add('pickup_zone','F',gap>=.02-1e-9,gap,'>=0.02 m from reserved strip')
    if 'door' in task:
        side=scene['joints'][0]['hinge_side']
        add('hinge_side','D',side==task['door']['target_side'],side,'reference side')
        motion=sweep(task['door'],side,objects)
        add('door_sweep','F',motion['status']=='pass',motion,'certified clearance >=0.03 m',motion['status'])
    if 'start' in task:
        path=scene['path_xy_m']
        endpoint_error=max(math.dist(path[0],task['start']),math.dist(path[-1],task['goal']))
        add('path_endpoints','F',endpoint_error<=.01,endpoint_error,'<=0.01 m')
        radius=task['vehicle_radius_m']+task['path_clearance_m']
        w,h=task['bounds_m']
        boundary=min(min(p[0],p[1],w-p[0],h-p[1]) for p in path)
        add('path_boundary','F',boundary>=radius-1e-9,boundary,f'>={radius} m')
        # Ground disk collides with any solid intersecting its 0..0.4 m body.
        blockers=[footprint(o) for o in objects if zinterval(o)[0]<.4 and zinterval(o)[1]>0]
        gap=min(segment_polygon(a,b,poly) for a,b in zip(path,path[1:]) for poly in blockers)
        add('path_segments','F',gap>=radius-1e-9,gap,f'>={radius} m for entire segments')
        # Door can occupy any pose: inflate sampled footprints by motion bound.
        door=task['door'];samples=121
        bound=math.hypot(door['size_m'][0],door['size_m'][1]/2)*math.pi/2/(samples-1)/2
        door_gap=min(segment_polygon(a,b,footprint(door_object(door,side,90*i/(samples-1))))
                     for i in range(samples) for a,b in zip(path,path[1:]))-bound
        add('path_door_envelope','F',door_gap>=radius-1e-9,door_gap,f'>={radius} m certified')
        add('event_acknowledged','F',phase=='post_event',phase,'post_event required for final success')
    dims={d:sum(c['passed'] for c in checks if c['dimension']==d)/max(1,sum(c['dimension']==d for c in checks)) for d in 'DGF'}
    success=all(c['passed'] for c in checks)
    det=(.35*dims['D']+.25*dims['G']+.30*dims['F'])/.9
    if visual is not None and (type(visual) not in (float,int) or not 0<=visual<=1):raise ValueError('V must be in [0,1] or missing')
    total=.9*det+.1*visual if visual is not None else None
    failed=[c['criterion_id'] for c in checks if not c['passed']]
    labels=[]
    if any(x.startswith('relation:') or x=='hinge_side' for x in failed):labels.append('PERCEPTION')
    if 'door_sweep' in failed:labels.append('KINEMATIC')
    if any(x.startswith('path_') for x in failed):labels.append('PLAN')
    if any(x.startswith('support:') for x in failed):labels.append('SUPPORT')
    if phase=='post_event' and 'path_segments' in failed:labels.append('STALE_STATE')
    return dict(scorer_version=VERSION,status='ok',scene_hash=digest(scene),E=1,I=1,**dims,V=visual,
                R_visual=visual,R_sum=total,R_safe=min(total,1. if success else .49) if total is not None else None,
                experimental_shaping_version='failure-scale-v1',R_shaped_det=det if success else .49*det,
                R_shaped=(.5+.5*total if success else .49*total) if total is not None else None,
                R_det=min(det,1. if success else .49),success=success,checks=checks,failure_labels=labels)
