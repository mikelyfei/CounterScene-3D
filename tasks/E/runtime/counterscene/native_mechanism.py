"""Additional continuous enclosure checks for the rendered robot's real tool geometry.

The existing arm certificate covers link capsules, but not the wider gripper,
wrist connector or pedestal. These bounds match spatial_scene_props.mechanism.
"""
import math
from .spatial_suite import kinematics as k,pick_place as pp

# Blender's near-vertical cylinder orientation needs a measured 0.1mm enclosure allowance.
CONNECTOR_RADIUS=.0181
VERIFIED_LINK_RADIUS=.0401
TOOL_BRIDGE_BOTTOM=.125
TOOL_CONNECTOR_TOP=.14
TOOL_TOP=.155


def executed_ball_timeline(task,sub):
    """Ideal grasp execution: misses do not attach; release uses the actual tip."""
    import copy
    mech=sub['mechanism'];rest={b['id']:b['pickup'][:] for b in task['balls']};holding=None;frames=[]
    events={e['keyframe']:e for e in sub['events']}
    for index,q in enumerate(sub['keyframes']):
        tip=k.fk(mech,q)[3];event=events.get(index)
        if event and event['action']=='grasp' and holding is None and math.dist(tip,rest[event['ball']])<=task['reach_tolerance_m']:holding=event['ball']
        elif event and event['action']=='release' and holding==event['ball']:
            rest[holding]=list(tip);holding=None
        positions=copy.deepcopy(rest)
        if holding:positions[holding]=None
        frames.append(positions)
    return frames


def tool_geometry(mech,q):
    points=k.fk(mech,q);elbow,tip=points[2:]
    length=math.dist(elbow,tip)
    wrist=[tip[i]-.13*(tip[i]-elbow[i])/length for i in range(3)]
    connector=[tip[0],tip[1],tip[2]+TOOL_CONNECTOR_TOP]
    box=[tip[0]-.105,tip[0]+.105,tip[1]-.03,tip[1]+.03,tip[2]-.04,tip[2]+TOOL_TOP]
    return points,wrist,connector,box


def _box_gap(a,b):
    return math.sqrt(sum(max(a[2*i]-b[2*i+1],b[2*i]-a[2*i+1],0)**2 for i in range(3)))


def check_tool_motion(task,sub):
    import copy
    task=copy.deepcopy(task);task['verified_link_radius_m']=VERIFIED_LINK_RADIUS
    # Released objects persist as real spheres; enclosing boxes conservatively
    # cover both arm and carried-payload sweeps in the existing certificate.
    existing={obstacle['id'] for obstacle in task['obstacles']}
    for obj in task.get('resting_objects',[]):
        if obj['id'] in existing:raise ValueError('Released object ID collides with fixture ID')
        position=obj['position'];radius=obj['radius_m']
        if len(position)!=3 or not math.isfinite(radius) or radius<=0 or any(not math.isfinite(v) for v in position):raise ValueError('Invalid released-object geometry')
        task['obstacles'].append(dict(id=obj['id'],box=[v for x in position for v in (x-radius,x+radius)]))
        existing.add(obj['id'])
    clean=k.check_motion(task,sub)
    if not clean['valid_output']:return dict(ok=False,reason='invalid_output')
    mech=clean['mechanism'];keys=sub['keyframes'];base=mech['base']
    pedestal=[base[0]-.125,base[0]+.125,base[1]-.125,base[1]+.125,0,.025]
    pedestal_task=copy.deepcopy(task);pedestal_task['obstacles'].append(dict(id='pedestal',box=pedestal))
    arm_with_pedestal=k.check_motion(pedestal_task,sub)
    fixed=[]
    for obstacle in task['obstacles']:
        gap=_box_gap(pedestal,obstacle['box'])
        post=k._seg_box_dist([*base,0],[*base,k.POST_H],obstacle['box'],.01)-VERIFIED_LINK_RADIUS
        fixed.append(dict(obstacle=obstacle['id'],pedestal_gap_m=gap,post_clearance_m=post,ok=min(gap,post)>=k.CLEARANCE))
    for ball in task['balls']:
        for role in ('pickup','destination'):
            point=ball[role];pointbox=[v for x in point for v in (x,x)]
            gap=_box_gap(pedestal,pointbox)-ball['radius_m']
            fixed.append(dict(ball=ball['id'],role=role,pedestal_clearance_m=gap,ok=gap>=k.CLEARANCE))
    events,error=pp._events(task,sub,len(keys))
    if error:return dict(ok=False,reason=error,fixed=fixed)
    span=pp._timeline(task,events);rows=[]
    # The connector endpoint at the wrist has the same link-point motion bound as the tip.
    lever=k._lever(mech)
    for n,(a,b) in enumerate(zip(keys,keys[1:])):
        sweep=sum(abs(math.radians(b[j]-a[j]))*lever[j] for j in range(3))
        steps=max(1,math.ceil(sweep/.005));worst=float('inf');why=''
        rest=pp._resting(task,span,n)
        held=next((ball for ball in task['balls'] if span[ball['id']][0]<=n<span[ball['id']][1]),None)
        own_contact=float('inf')
        for s in range(steps+1):
            q=[a[j]+s/steps*(b[j]-a[j]) for j in range(3)]
            pts,wrist,end,box=tool_geometry(mech,q)
            checks=[(box[4],'gripper floor'),(min(wrist[2],end[2])-CONNECTOR_RADIUS,'connector floor'),
                    (k._seg_box_dist(pts[1],pts[2],box,.01)-VERIFIED_LINK_RADIUS,'gripper link1'),
                    (k._seg_box_dist(pts[0],pts[1],box,.01)-VERIFIED_LINK_RADIUS,'gripper post'),
                    (k._seg_seg_dist(wrist,end,pts[1],pts[2])-CONNECTOR_RADIUS-VERIFIED_LINK_RADIUS,'connector link1'),
                    (k._seg_seg_dist(wrist,end,pts[0],pts[1])-CONNECTOR_RADIUS-VERIFIED_LINK_RADIUS,'connector post')]
            for obstacle in task['obstacles']+[dict(id='pedestal',box=pedestal)]:
                checks.extend([(_box_gap(box,obstacle['box']),'gripper '+obstacle['id']),
                               (k._seg_box_dist(wrist,end,obstacle['box'],.01)-CONNECTOR_RADIUS,'connector '+obstacle['id'])])
            for bid,position in rest:
                pointbox=[v for x in position for v in (x,x)]
                checks.extend([(_box_gap(box,pointbox)-pp.RADIUS,'gripper resting '+bid),
                               (k._seg_seg_dist(wrist,end,position,position)-CONNECTOR_RADIUS-pp.RADIUS,'connector resting '+bid)])
            if held:
                checks.append((k._seg_seg_dist(pts[1],pts[2],pts[3],pts[3])-VERIFIED_LINK_RADIUS-held['radius_m'],'held ball link1'))
                pointbox=[v for x in pts[3] for v in (x,x)]
                checks.append((_box_gap(pedestal,pointbox)-held['radius_m'],'held ball pedestal'))
                own_contact=min(own_contact,k._seg_seg_dist(wrist,end,pts[3],pts[3])-CONNECTOR_RADIUS-held['radius_m'])
            value,culprit=min(checks)
            if value<worst:worst,why=value,culprit
        # Tool-vs-link1 is relative motion of two moving bodies, hence twice the covering radius.
        margin=worst-sweep/steps
        contact_margin=own_contact-sweep/steps
        rows.append(dict(segment=n,certified_margin_m=margin,closest=why,held_connector_clearance_m=contact_margin if held else None,ok=margin>=k.CLEARANCE and (not held or contact_margin>=-.01)))
    onward_ok=True
    if task.get('stateful_operation'):
        onward=[i for i,w in enumerate(task['waypoints']) if w.get('role')=='onward']
        onward_ok=len(onward)==1 and onward[0]==len(task['waypoints'])-1 and sub.get('waypoint_keyframes',[])[-1:]==[len(keys)-1] and events[-1]['keyframe']<len(keys)-1
    return dict(ok=arm_with_pedestal['all_ok'] and all(r['ok'] for r in fixed+rows) and onward_ok,onward_ok=onward_ok,arm_ok=arm_with_pedestal['all_ok'],arm_report=arm_with_pedestal,fixed=fixed,segments=rows,
                scope='Conservative enclosure certificate; requires separately verified rendered mesh enclosure; no force dynamics')
