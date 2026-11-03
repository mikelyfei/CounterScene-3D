"""Grounded pick/carry/place: declared home pose, grasp/release events, one or two balls.

Ideal kinematic grasp: an attached ball's centre is the gripper tip. A resting ball (not yet grasped, or already
released) is a static sphere obstacle for the arm links and for any carried ball. Only the segment arriving at a
ball's grasp keyframe and the segment leaving its release keyframe may touch that ball (that contact is the grasp).
"""
import math,random

RADIUS=.08
CABINET_H=(.45,1.10)      # cabinet heights in every tier: sampled here, pairwise at least CABINET_GAP apart (owner, 06:18Z)
CABINET_GAP=.12


def cabinet_height(rng,lo,taken):
    """A height in [max(lo, 0.45), 1.10] at least CABINET_GAP from every height in taken, or None."""
    lo=max(lo,CABINET_H[0])
    options=[h for h in (lo+i*.01 for i in range(int((CABINET_H[1]-lo)/.01)+1)) if all(abs(h-t)>=CABINET_GAP for t in taken)]
    return round(rng.choice(options),4) if options else None

def _trig_range(a,b,cos=False):
    a,b=sorted((a,b));f=math.cos if cos else math.sin
    points=[a,b];offset=0 if cos else math.pi/2
    for k in range(math.floor((a-offset)/math.pi)-1,math.ceil((b-offset)/math.pi)+2):
        x=offset+k*math.pi
        if a<=x<=b:points.append(x)
    values=[f(x) for x in points];return min(values),max(values)


def floor_lower_bound(mech,a,b):
    from .kinematics import POST_H,fk
    qa=[math.radians(v) for v in a];qb=[math.radians(v) for v in b];l1,l2=mech['link_lengths']
    s1=_trig_range(qa[1],qb[1]);s2=_trig_range(qa[1]+qa[2],qb[1]+qb[2])
    lower=POST_H+l1*s1[0]+l2*s2[0]
    terms=[]
    for coefficient,angles in ((l1*(qb[1]-qa[1]),(qa[1],qb[1])),(l2*(qb[1]+qb[2]-qa[1]-qa[2]),(qa[1]+qa[2],qb[1]+qb[2]))):
        lo,hi=_trig_range(*angles,cos=True);terms.append(sorted((coefficient*lo,coefficient*hi)))
    if sum(t[0] for t in terms)>=-1e-10 or sum(t[1] for t in terms)<=1e-10:
        lower=max(lower,min(fk(mech,a)[3][2],fk(mech,b)[3][2]))
    return lower


def _events(task,sub,n_keys):
    """(validated events, None) or (None, problem). Each ball is grasped once, then released once; one gripper."""
    events=sub.get('events');balls={b['id'] for b in task['balls']}
    if not isinstance(events,list) or not events:return None,'events must be a non-empty list of {"keyframe", "action", "ball"}'
    state={b:'resting' for b in balls};holding=None;last=-1
    for e in events:
        if not isinstance(e,dict) or type(e.get('keyframe'))!=int or e.get('action') not in ('grasp','release') or e.get('ball') not in balls:
            return None,'each event needs an integer keyframe, action "grasp" or "release" and a known ball id'
        if not last<e['keyframe']<n_keys:return None,'event keyframes must be strictly increasing keyframe indices'
        last=e['keyframe']
        if e['action']=='grasp':
            if holding:return None,f'grasp of {e["ball"]} while still holding {holding}'
            if state[e['ball']]!='resting':return None,f'{e["ball"]} can be grasped only once'
            holding,state[e['ball']]=e['ball'],'held'
        else:
            if holding!=e['ball']:return None,f'release of {e["ball"]}, which is not held'
            holding,state[e['ball']]=None,'placed'
    if holding or any(v!='placed' for v in state.values()):return None,'every ball must be grasped once and released once'
    return events,None


def _timeline(task,events):
    """{ball: (grasp keyframe, release keyframe)} from validated events."""
    out={}
    for e in events:out.setdefault(e['ball'],[None,None])[0 if e['action']=='grasp' else 1]=e['keyframe']
    return {b:tuple(v) for b,v in out.items()}


def _resting(task,span,n):
    """Resting balls (id, centre) during segment n -> n+1, excluding the contact segments of each ball."""
    out=[]
    for b in task['balls']:
        g,r=span[b['id']]
        if g<=n<r or n+1==g or n==r:continue
        out.append((b['id'],b['pickup'] if n<g else b['destination']))
    return out


def _support_top(task,ball,p):
    """Top height of the ball's declared support under point p (floor = 0), or None if p is not over it."""
    if ball['support']=='floor':return 0.
    box=next(o['box'] for o in task['obstacles'] if o['id']==ball['support'])
    return box[5] if box[0]<=p[0]<=box[1] and box[2]<=p[1]<=box[3] else None


def check_payload(task,sub,mech,keys,wk):
    from .kinematics import fk,_lever,_seg_seg_dist,POST_H,RADIUS as ARM_RADIUS,CLEARANCE
    ARM_RADIUS=task.get('verified_link_radius_m',ARM_RADIUS)
    problems=[];home=task['home']
    if any(abs(a-b)>1e-6 for a,b in zip(keys[0],home)):problems.append('keyframes[0] must equal the declared home pose q_home')
    if task.get('return_home') and any(abs(a-b)>1e-6 for a,b in zip(keys[-1],home)):problems.append('the last keyframe must return to q_home')
    events,why=_events(task,sub,len(keys))
    if why:return dict(ok=False,problems=problems+[why],carry_segments=[],arm_vs_balls=[])
    span=_timeline(task,events);ids={w['id']:i for i,w in enumerate(task['waypoints'])};r=RADIUS
    for b in task['balls']:
        g,rel=span[b['id']]
        if g==0:problems.append(f'{b["id"]} grasped at the home pose: the arm must first move to the ball')
        if wk[ids['pickup_'+b['id']]]!=g:problems.append(f'grasp of {b["id"]} must happen at its pickup waypoint keyframe')
        if wk[ids['place_'+b['id']]]!=rel:problems.append(f'release of {b["id"]} must happen at its place waypoint keyframe')
        tip=fk(mech,keys[g])[3]
        if math.dist(tip,b['pickup'])>task['reach_tolerance_m']:problems.append(f'gripper misses {b["id"]} at its grasp keyframe')
        drop=fk(mech,keys[rel])[3];top=_support_top(task,b,drop)
        if math.dist(drop,b['destination'])>task['reach_tolerance_m'] or top is None or abs(drop[2]-top-b['radius_m'])>.005:
            problems.append(f'{b["id"]} must be released resting on its support ({b["support"]}) at its destination, within 5 mm vertically')
    carry,arm=[],[]
    lever=_lever(mech)
    for n in range(len(keys)-1):
        a,c=keys[n],keys[n+1];sweep=sum(abs(math.radians(c[j]-a[j]))*lever[j] for j in range(3));steps=max(1,math.ceil(sweep/(CLEARANCE/2)))
        held=next((b for b in task['balls'] if span[b['id']][0]<=n<span[b['id']][1]),None);rest=_resting(task,span,n)
        worst_c,why_c,worst_a,why_a=1e9,'',1e9,''
        for s in range(steps+1):
            pts=fk(mech,[a[j]+s/steps*(c[j]-a[j]) for j in range(3)]);tip=pts[3]
            for bid,p in rest:                                     # arm links vs resting balls
                for li in (1,2):
                    d=_seg_seg_dist(pts[li],pts[li+1],p,p)-ARM_RADIUS-r
                    if d<worst_a:worst_a,why_a=d,f'link{li} vs resting {bid}'
            if held:                                               # carried ball vs fixtures, post and resting balls
                for o in task['obstacles']:
                    if o['id']==held['support'] and n+1==span[held['id']][1]:continue   # setting the ball down on its cabinet
                    box=o['box'];d=math.sqrt(sum(max(box[2*k]-tip[k],0,tip[k]-box[2*k+1])**2 for k in range(3)))-r
                    if d<worst_c:worst_c,why_c=d,f'{held["id"]} vs {o["id"]}'
                base=mech['base'];d=math.sqrt((tip[0]-base[0])**2+(tip[1]-base[1])**2+max(-tip[2],0,tip[2]-POST_H)**2)-r-ARM_RADIUS
                if d<worst_c:worst_c,why_c=d,f'{held["id"]} vs fixed post'
                for bid,p in rest:
                    d=math.dist(tip,p)-2*r
                    if d<worst_c:worst_c,why_c=d,f'{held["id"]} vs resting {bid}'
        slack=sweep/steps/2
        if rest:arm.append(dict(from_keyframe=n,to_keyframe=n+1,certified_clearance_m=round(worst_a-slack,5),closest=why_a,ok=worst_a-slack>=CLEARANCE))
        if held:
            floor=floor_lower_bound(mech,a,c)-r
            carry.append(dict(from_keyframe=n,to_keyframe=n+1,ball=held['id'],certified_clearance_m=round(worst_c-slack,5),floor_margin_m=round(floor,5),closest=why_c,
                              ok=worst_c-slack>=CLEARANCE and floor>=-1e-4))
    return dict(ok=not problems and all(s['ok'] for s in carry+arm),problems=problems,carry_segments=carry,arm_vs_balls=arm,events=events,
                model='ideal kinematic grasp: an attached ball centre follows the tip; resting balls are static sphere obstacles; '
                      'only the segment into a grasp and the segment out of a release may touch that ball; no grasp-force dynamics')


TIERS={'L1':dict(plan=[('pickup','ball_1'),('place','ball_1')],obstacles=1,free_len=False,free_base=False,ret=False),
       'L2':dict(plan=[('pickup','ball_1'),('transfer',None),('place','ball_1')],obstacles=3,free_len=True,free_base=False,ret=False),
       'L3':dict(plan=[('pickup','ball_1'),('place','ball_1'),('pickup','ball_2'),('place','ball_2')],obstacles=4,free_len=True,free_base=True,ret=True,
                 drop_on_cabinet='ball_1')}


def tuck_path(p,q):
    """Fold in place, swing folded, unfold at the target yaw (the certified gold detour)."""
    return [[p[0],60.,p[2]],[p[0],60.,-150.],[q[0],60.,-150.],[q[0],60.,q[2]],q]


def make(seed,tier):
    from . import kinematics as k
    cfg=TIERS[tier];plan=cfg['plan']
    for attempt in range(2000):
        rng=random.Random(f'grounded-pick-place-v3-{seed}-{tier}-{attempt}');base=[1.8,1.8]
        lengths=[round(rng.uniform(.65,.85),2),round(rng.uniform(.55,.7),2)];mech=dict(base=base,link_lengths=lengths)
        angle=rng.uniform(-150,-60);angles=[]
        for _ in plan:angles.append(angle);angle+=rng.uniform(50,80)
        if angles[-1]>155:continue
        home_yaw=rng.uniform(-160,160)
        if min(abs(home_yaw-a) for a in angles)<45:continue
        home=[round(home_yaw,1),80.,-120.]
        targets,configs,stand=[],[],None
        for (role,ball),a in zip(plan,angles):
            radius=rng.uniform(.8,1.05);height=RADIUS if role!='transfer' else rng.uniform(.45,.8)
            point=[base[0]+radius*math.cos(math.radians(a)),base[1]+radius*math.sin(math.radians(a)),height]
            if role=='place' and ball==cfg.get('drop_on_cabinet'):        # cabinet under the drop point; the ball rests on its top
                top,half=round(rng.uniform(.62,.95),4),rng.uniform(.14,.18)      # mid/high: another cabinet must be lower
                point[2]=top+RADIUS
                stand=dict(id='cabinet_1',kind='floor_cabinet',support='floor',
                           box=[round(point[0]-half,4),round(point[0]+half,4),round(point[1]-half,4),round(point[1]+half,4),0.,top])
            q=k._ik(mech,point)
            if q is None or any(not lo+.5<=v<=hi-.5 for v,(lo,hi) in zip(q,k.HW_LIMITS)):break
            targets.append([round(v,4) for v in point]);configs.append(q)
        if len(configs)!=len(plan):continue
        keys=[home];wk=[]
        for q in configs:keys+=tuck_path(keys[-1],q);wk.append(len(keys)-1)
        if cfg['ret']:keys+=tuck_path(keys[-1],home)
        balls=[]
        for b in sorted({ball for _,ball in plan if ball}):
            pick=targets[plan.index(('pickup',b))];place=targets[plan.index(('place',b))]
            balls.append(dict(id=b,radius_m=RADIUS,pickup=pick,destination=place,support=stand['id'] if stand and b==cfg.get('drop_on_cabinet') else 'floor',
                              destination_role='intermediate drop point' if cfg['ret'] and b=='ball_1' else 'final placement'))
        events=[dict(keyframe=wk[i],action='grasp' if role=='pickup' else 'release',ball=ball) for i,(role,ball) in enumerate(plan) if role!='transfer']
        mids=[k.fk(mech,[(p[j]+q[j])/2 for j in range(3)])[3] for p,q in zip([home]+configs[:-1],configs)]
        if tier!='L1':mids.append(k.fk(mech,k.VIA['lift']((configs[0][0]+configs[1][0])/2))[3])
        obstacles=[stand] if stand else []
        if stand and (min(stand['box'][0],stand['box'][2])<.08 or max(stand['box'][1],stand['box'][3])>3.52 or
                      k._seg_box_dist([*base,0],[*base,k.POST_H],stand['box'],.02)-k.RADIUS<k.CLEARANCE):continue
        for point in mids:
            if len(obstacles)==cfg['obstacles']+bool(stand):break
            half=rng.uniform(.13,.20);tops=[o['box'][5] for o in obstacles if o['kind']=='floor_cabinet']
            height=round(min(2.2,point[2]+.15),4) if point[2]+.15>CABINET_H[1] else cabinet_height(rng,point[2]+.15,tops)   # tall enough to block the direct path
            if height is None:continue
            box=[round(point[0]-half,4),round(point[0]+half,4),round(point[1]-half,4),round(point[1]+half,4),0.,height]
            if min(box[0],box[2])<.08 or box[1]>3.52 or box[3]>3.52:continue
            if k._seg_box_dist([*base,0],[*base,k.POST_H],box,.02)-k.RADIUS<k.CLEARANCE:continue
            if any(all(min(box[2*j+1],o['box'][2*j+1])-max(box[2*j],o['box'][2*j])>0 for j in range(3)) for o in obstacles):continue
            if any(math.sqrt(sum(max(box[2*j]-p[j],0,p[j]-box[2*j+1])**2 for j in range(3)))<RADIUS+.05 for b in balls for p in (b['pickup'],b['destination'])):continue
            if stand and any(math.sqrt(sum(max(box[2*j]-p[j],0,p[j]-box[2*j+1])**2 for j in range(3)))<RADIUS+.05 for p in [[(stand['box'][0]+stand['box'][1])/2,(stand['box'][2]+stand['box'][3])/2,stand['box'][5]]]):continue
            n=len(obstacles)+1
            obstacles.append(dict(id=f'cabinet_{n}' if box[5]<=CABINET_H[1] else f'column_{n}',kind='floor_column' if box[5]>CABINET_H[1] else 'floor_cabinet',support='floor',box=box))
        if len(obstacles)!=cfg['obstacles']+bool(stand):continue
        if stand and stand['box'][5]<=min(o['box'][5] for o in obstacles if o['kind']=='floor_cabinet' and o is not stand):continue   # drop cabinet is not the lowest
        waypoints=[dict(id=f'{role}_{ball}' if ball else 'transfer_1',position=t,role=role,**({'ball':ball} if ball else {})) for (role,ball),t in zip(plan,targets)]
        task=dict(experiment='D',instance=f'D-{tier}-s{seed}',version='grounded-pick-place-3',tier=tier,room=[3.6,3.6,2.4],post_height=k.POST_H,link_radius=k.RADIUS,
                  clearance_m=k.CLEARANCE,reach_tolerance_m=k.REACH_TOL,hardware_limits_deg=k.HW_LIMITS,free_base=cfg['free_base'],free_len=cfg['free_len'],
                  base=None if cfg['free_base'] else base,link_lengths=None if cfg['free_len'] else lengths,obstacles=obstacles,waypoints=waypoints,balls=balls,
                  home=home,return_home=cfg['ret'])
        if cfg['free_len']:task['link_length_range']=[[.4,.9],[.35,.8]]
        if cfg['free_base']:task['base_region']=[[1.5,1.5],[2.1,2.1]]
        home_tip=k.fk(mech,home)[3]
        if any(math.dist(home_tip,b['pickup'])<=.3 for b in balls):continue
        gold=dict(mechanism=dict(**mech,joint_limits_deg=k.HW_LIMITS),keyframes=keys,waypoint_keyframes=wk,events=events)
        report=k.check_motion(task,gold)
        if not report.get('all_ok'):continue
        if k.check_motion(task,k.static_baseline(task,gold))['all_ok']:continue
        if tier!='L1' and k.check_motion(task,k.lift_baseline(task,gold))['all_ok']:continue
        if tier=='L3' and k.check_motion(task,k.revisit_placed_ball(task,gold))['all_ok']:continue
        if stand and (k.check_motion(task,k.release_above_support(task,gold))['all_ok'] or k.check_motion(task,k.release_at_other_cabinet_height(task,gold))['all_ok']):continue
        return task,dict(gold=gold,gold_report=report)
    raise RuntimeError('No certified grounded pick-place instance found')
