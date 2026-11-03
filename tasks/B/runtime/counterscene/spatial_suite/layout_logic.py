"""Semantic placement contract shared by all A/B/C/E indoor items."""
import random
from . import core

HOSTS=('table_green','cabinet_blue','desk_blue','sidetable_brown')
SHELVES=('shelf_orange',)
SURFACE_OBJECTS=('box_yellow','case_brown','tv_black','monitor_gray','laptop_gray','flowerpot_green','soccer_white','rugby_brown')
SHELF_ITEMS=('box_yellow','case_brown','flowerpot_green','soccer_white','rugby_brown')
SCREENS=('tv_black','monitor_gray','laptop_gray')
SEATS=('chair_cyan','deskchair_black')
WORK_SURFACES=('table_green','desk_blue')
SEAT_REACH=.5


def policy():
    return dict(version='logical-room-3',flat_supports=list(HOSTS),surface_objects=list(SURFACE_OBJECTS),
                tabletop_required=list(SCREENS),wall_storage=['shelf_orange','cabinet_blue'],wall_distance_m=.35,
                shelf_supports=list(SHELVES),shelf_items=list(SHELF_ITEMS),seats=list(SEATS),work_surfaces=list(WORK_SURFACES),
                screens=list(SCREENS),seat_reach_m=SEAT_REACH,
                rules=('Large furniture, chairs, lamps, plants and columns stand on the floor. Small items may stand on the floor or on a declared flat '
                       'table/cabinet/desk surface with their full footprint supported; shelf items may also rest on a bookcase shelf board with their '
                       'full footprint on that board. Screens (TV, monitor, laptop) must stand on such a surface. Tall storage stays near the back wall. '
                       'A seat within seat_reach_m of a table or desk is a seat at it and must face it (facing relation). A screen standing on a table or '
                       'desk that has a seat facing it must itself face that seat, so its display - not its back - is toward the person.'))


def apply_rules(obj,scene_policy):
    if not scene_policy:return obj
    obj=dict(obj);obj['room_policy']=scene_policy
    return obj


def _rests_on(b,h):
    """Full footprint of box b on the top face of box h."""
    return abs(b[4]-h[5])<=core.TOL and all(b[i]>=h[i]-1e-6 and b[i+1]<=h[i+1]+1e-6 for i in (0,2))


def issues(objs,room):
    out=[]
    for oid,o in objs.items():
        p=o.get('room_policy')
        if not p:continue
        b=core.aabb(o);base=abs(b[4])<=core.TOL
        if oid not in p['surface_objects'] and not base:out.append(dict(check='semantic_floor_only',objects=[oid]))
        if oid in p['tabletop_required'] and base:out.append(dict(check='tabletop_required',objects=[oid]))
        if oid in p['surface_objects'] and not base:
            hosts=[objs[n] for n in p['flat_supports'] if n in objs]
            if oid in p.get('shelf_items',()):hosts+=[x for x in objs.values() if x.get('group') in p.get('shelf_supports',()) and '#board' in x['id']]
            if not any(_rests_on(b,core.aabb(h)) for h in hosts):out.append(dict(check='approved_full_footprint_support',objects=[oid]))
        if oid in p['wall_storage'] and room[1]-b[3]>p['wall_distance_m']+core.TOL:out.append(dict(check='storage_near_back_wall',objects=[oid]))
    out+=workstation_issues(objs)
    return out


def seated_at(objs,host,p):
    """Seats within reach of a work surface that face it."""
    return [s for s in p.get('seats',()) if s in objs and core.gap(objs[s],objs[host])<=p['seat_reach_m'] and core.rel_facing(objs[s],objs[host])[0]]


def workstation_issues(objs):
    out=[]
    for oid,o in objs.items():
        p=o.get('room_policy')
        if not p or 'seats' not in p:continue
        if oid in p['seats']:
            for host in p['work_surfaces']:
                if host in objs and core.gap(o,objs[host])<=p['seat_reach_m'] and not core.rel_facing(o,objs[host])[0]:
                    out.append(dict(check='seat_faces_work_surface',objects=[oid,host]))
        if oid in p['screens']:
            b=core.aabb(o)
            for host in p['work_surfaces']:
                if host in objs and _rests_on(b,core.aabb(objs[host])):
                    for seat in seated_at(objs,host,p):
                        if not core.rel_facing(o,objs[seat])[0]:out.append(dict(check='screen_faces_seat',objects=[oid,seat]))
    return out


COMPOSITES={'shelf_orange':'bookcase'}
BALLS=('soccer_white','rugby_brown')
SIDE=('lamp_white','plant_pink','sidetable_brown')
SEAT_FOR={'desk_blue':'deskchair_black','table_green':'chair_cyan'}


def facing_yaw(o,target):
    """The right-angle yaw whose front direction points most nearly at target's centre."""
    cx,cy,_=core.center(o);tx,ty,_=core.center(target)
    return max((0,90,180,270),key=lambda y:core.front_vector(dict(yaw=y))[0]*(tx-cx)+core.front_vector(dict(yaw=y))[1]*(ty-cy))


def _place_composite(o,objs):
    parts=core._expand(o,COMPOSITES[o['id']]);o.update(envelope=True,boards=[q for k,q in parts.items() if '#board' in k]);objs[o['id']]=o;objs.update(parts)


def _pick(rng,n,stack,catalog,must):
    """n catalogue entries including `must`, with every dependency (screen host, item support) present."""
    by={c[0]:c for c in catalog};chosen=[by[m] for m in must]
    floor=[c for c in catalog if c[0] not in SURFACE_OBJECTS and c not in chosen];small=[c for c in catalog if c[0] in SURFACE_OBJECTS and c not in chosen]
    ns=max(0,min(len(small),max(stack,n-len(floor)-len(chosen))-sum(c[0] in SURFACE_OBJECTS for c in chosen)))
    rng.shuffle(small);chosen+=small[:ns];chosen+=rng.sample(floor,min(len(floor),max(0,n-len(chosen))));chosen+=small[ns:ns+max(0,n-len(chosen))]
    ids={c[0] for c in chosen}
    def need(alternatives):
        if ids&set(alternatives):return
        drop=next((c for c in chosen[::-1] if c[0] not in must and c[0] not in SURFACE_OBJECTS and c[0] not in HOSTS),None)
        if drop:chosen.remove(drop);ids.discard(drop[0])
        chosen.append(by[alternatives[0]]);ids.add(alternatives[0])
    if ids&set(SCREENS):need(WORK_SURFACES)
    if ids&(set(SURFACE_OBJECTS)-set(BALLS)):need(HOSTS+SHELVES)
    return chosen


def layout(rng,n,use_yaw,stack,jitter,catalog,must=(),wants=None,uniform_jitter=False):
    """Logical room: wall storage, work surfaces with seats facing them, side items, then items on hosts/shelves/floor.
    wants: optional {id: 'floor' | 'shelf' | host id} for surface objects."""
    wants=wants or {};chosen=_pick(rng,n,stack,catalog,must)
    if jitter and uniform_jitter:
        chosen=[(name,size if name in BALLS else [round(v*factor,6) for v in size],s) for name,size,s in chosen for factor in [rng.uniform(1-jitter,1+jitter)]]
    elif jitter:chosen=[(name,size if name in BALLS else [round(v*rng.uniform(1-jitter,1+jitter),2) for v in size],s) for name,size,s in chosen]
    W,D=round(2.5+.35*n,2),round(2+.25*n,2);room=[W,D,2.5];objs={};rules=policy()
    floor=[c for c in chosen if c[0] not in SURFACE_OBJECTS or (c[0] in BALLS and wants.get(c[0])=='floor')]
    items=[c for c in chosen if c not in floor]
    rank=lambda cid:0 if cid in rules['wall_storage'] else 1 if cid in WORK_SURFACES else 2 if cid in SEATS else 3 if cid in SIDE else 4
    def placed(o,margin=.12):
        b=core.aabb(o)
        if b[0]<0 or b[2]<0 or b[1]>W or b[3]>D:return False
        others=[p for p in objs.values() if not p.get('rigid')]
        return all(core.penetration(o,p)<=0 for p in others) and all(core.gap(o,p)>=margin or core.gap(o,p)<1e-6 for p in others)
    for cid,size,_ in sorted(floor,key=lambda c:rank(c[0])):
        hosts=[h for h,seat in SEAT_FOR.items() if h in objs and not any(core.gap(objs[s],objs[h])<=SEAT_REACH for s in SEATS if s in objs)]
        for attempt in range(500):
            o=dict(id=cid,size=size,position=[0,0,0],yaw=0,room_policy=rules)
            if cid in SEATS and hosts:
                host=objs[sorted(hosts,key=lambda h:SEAT_FOR[h]!=cid)[0]];h=core.aabb(host);behind=rng.random()<.5;o['yaw']=0 if behind else 180;fx,fy=core.footprint_size(o)
                x=host['position'][0]+rng.uniform(-.2,.2);y=h[3]+fy/2+rng.uniform(.15,.3) if behind else h[2]-fy/2-rng.uniform(.15,.3)
            else:
                fx,fy=core.footprint_size(o)
                if cid in rules['wall_storage']:x=rng.uniform(fx/2+.08,W-fx/2-.08);y=D-fy/2-.10
                elif cid in WORK_SURFACES:x=W*rng.uniform(.3,.7);y=D*rng.uniform(.35,.6)
                elif cid in SIDE:x=fx/2+.12 if rng.random()<.5 else W-fx/2-.12;y=rng.uniform(fy/2+.15,D-fy/2-.15)
                elif [p for p in objs.values() if not p.get('rigid')] and rng.random()<.45 and cid not in SEATS:
                    host=core.aabb(rng.choice([p for p in objs.values() if not p.get('rigid')]));x=host[1]+fx/2;y=(host[2]+host[3])/2
                else:x=rng.uniform(fx/2+.1,W-fx/2-.1);y=rng.uniform(fy/2+.1,D-fy/2-.1)
            o['position']=[round(x,4),round(y,4),0.]
            if cid in SEATS and not hosts and any(core.gap(o,objs[w])<=SEAT_REACH for w in WORK_SURFACES if w in objs):continue
            if not placed(o):continue
            if cid in COMPOSITES:_place_composite(o,objs)
            else:objs[cid]=o
            break
        else:raise RuntimeError('logical floor layout failed')
    for cid,size,_ in sorted(items,key=lambda c:0 if c[0] in SCREENS else 1):
        want=wants.get(cid)
        for attempt in range(500):
            if cid in SCREENS:options=[h for h in WORK_SURFACES if h in objs]+(['cabinet_blue'] if cid=='tv_black' and 'cabinet_blue' in objs else [])
            elif want=='shelf':options=[b['id'] for s in SHELVES if s in objs for b in objs[s]['boards'][1:3]]
            elif want and want not in ('floor','shelf'):options=[want]
            else:options=[h for h in HOSTS if h in objs]+([b['id'] for s in SHELVES if s in objs for b in objs[s]['boards'][1:3]] if cid in SHELF_ITEMS else [])
            if not options:raise RuntimeError('no support for '+cid)
            host=objs[rng.choice(options)];h=core.aabb(host)
            o=dict(id=cid,size=size,yaw=rng.choice([0,90]) if use_yaw and cid not in BALLS else 0,position=[0,0,h[5]],room_policy=rules)
            seats=seated_at(objs,host['id'],rules) if cid in SCREENS and host['id'] in WORK_SURFACES else []
            if seats:o['yaw']=facing_yaw(dict(o,position=[host['position'][0],host['position'][1],h[5]]),objs[seats[0]])
            fx,fy=core.footprint_size(o)
            if fx>h[1]-h[0] or fy>h[3]-h[2]:continue
            x=rng.uniform(h[0]+fx/2,h[1]-fx/2);y=rng.uniform(h[2]+fy/2,h[3]-fy/2)
            if seats:x=min(max(objs[seats[0]]['position'][0]+rng.uniform(-.08,.08),h[0]+fx/2),h[1]-fx/2)
            o['position']=[round(x,4),round(y,4),h[5]]
            if all(core.penetration(o,p)<=1e-8 for p in core.solids(objs).values()) and not issues(dict(objs,**{cid:o}),room):objs[cid]=o;break
        else:raise RuntimeError('logical tabletop layout failed')
    if core.global_checks(objs,room):raise RuntimeError('logical layout invalid')
    return room,objs
