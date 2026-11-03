"""Experiment D: kinematic mechanism design with continuous-motion verification.

Mechanism template (a yaw-pitch-pitch arm on a fixed post):
  joint q1 = yaw about +z at the post base, post height h0 (fixed),
  joint q2 = pitch of link 1 at the post top (0 = horizontal, +90 = straight up),
  joint q3 = pitch of link 2 relative to link 1.
The agent chooses base position (L2+), link lengths (within ranges), declared joint limits
(inside hardware limits) and a keyframe sequence. Motion between keyframes is linear in
joint space; every link must stay clear of obstacles and the floor for the whole motion.
"""
import copy
import math
import random
from functools import lru_cache

POST_H = 0.6
RADIUS = 0.04          # link capsule radius (m)
CLEARANCE = 0.02       # required free space beyond the radius (m)
REACH_TOL = 0.03       # waypoint tolerance (m)
HW_LIMITS = [[-170, 170], [-30, 110], [-150, 150]]
MAX_KEYFRAMES = 32     # two-ball tasks with a return home need ~26

D_TIERS = {'L1': dict(waypoints=2, obstacles=1, free_base=False, free_len=False, yaw_limit=170),
           'L2': dict(waypoints=3, obstacles=1, free_base=False, free_len=True, yaw_limit=170, block_lift=True),
           'L3': dict(waypoints=4, obstacles=2, free_base=True, free_len=True, yaw_limit=120, block_lift=True)}
VIA = {'lift': lambda yaw: [yaw, 85., -60.],      # raise the whole arm over low obstacles
       'tuck': lambda yaw: [yaw, 60., -150.]}     # fold link 2 back under link 1, close to the post


def fk(mech, q):
    """Returns joint points [base, post_top, elbow, tip]."""
    bx, by = mech['base']
    l1, l2 = mech['link_lengths']
    q1, q2, q3 = (math.radians(v) for v in q)
    top = (bx, by, POST_H)

    def ray(p, pitch, L):
        return (p[0] + L * math.cos(pitch) * math.cos(q1), p[1] + L * math.cos(pitch) * math.sin(q1), p[2] + L * math.sin(pitch))
    elbow = ray(top, q2, l1)
    return [(bx, by, 0.), top, elbow, ray(elbow, q2 + q3, l2)]


def _seg_box_dist(a, b, box, step):
    """Conservative segment-to-AABB distance: sampled points minus half the sample spacing."""
    L = math.dist(a, b)
    n = max(1, int(math.ceil(L / step)))
    best = 1e9
    for i in range(n + 1):
        t = i / n
        p = [a[k] + t * (b[k] - a[k]) for k in range(3)]
        d = math.sqrt(sum(max(box[2 * k] - p[k], 0., p[k] - box[2 * k + 1]) ** 2 for k in range(3)))
        best = min(best, d)
    return best - (L / n) / 2


def _seg_seg_dist(p1, q1, p2, q2, n=12):
    """Exact segment distance, including parallel and degenerate segments.

    Sampling two segments needs the SUM of both covering radii for a safe
    lower bound. The earlier max-radius correction could overestimate clearance.
    """
    u=[q1[i]-p1[i] for i in range(3)];v=[q2[i]-p2[i] for i in range(3)];w=[p1[i]-p2[i] for i in range(3)]
    dot=lambda a,b:sum(x*y for x,y in zip(a,b))
    A,B,C,D,E=dot(u,u),dot(u,v),dot(v,v),dot(u,w),dot(v,w)
    clamp=lambda x:max(0.,min(1.,x))
    if A<=1e-20 and C<=1e-20:return math.dist(p1,p2)
    if A<=1e-20:s,t=0.,clamp(E/C)
    elif C<=1e-20:s,t=clamp(-D/A),0.
    else:
        denom=A*C-B*B
        s=clamp((B*E-C*D)/denom) if denom>1e-20 else 0.
        t=(B*s+E)/C
        if t<0.:t=0.;s=clamp(-D/A)
        elif t>1.:t=1.;s=clamp((B-D)/A)
    return math.sqrt(sum((w[i]+s*u[i]-t*v[i])**2 for i in range(3)))


def pose_clearance(mech, q, obstacles, radius=RADIUS):
    """Minimum free distance (beyond link radius) to obstacles, floor and self, plus the culprit."""
    pts = fk(mech, q)
    worst, why = 1e9, ''
    for li in (1, 2):                                   # moving links: top->elbow, elbow->tip
        a, b = pts[li], pts[li + 1]
        for o in obstacles:
            d = _seg_box_dist(a, b, o['box'], radius / 2) - radius
            if d < worst:
                worst, why = d, f'link{li} vs {o["id"]}'
        d = min(a[2], b[2]) - radius
        if d < worst:
            worst, why = d, f'link{li} vs floor'
    d = _seg_seg_dist(pts[0], pts[1], pts[2], pts[3]) - 2 * radius   # link2 vs post
    if d < worst:
        worst, why = d, 'link2 vs post'
    return worst, why


def _lever(mech):
    l1, l2 = mech['link_lengths']
    return (l1 + l2, l1 + l2, l2)   # max distance of any moving point from each joint axis


def check_motion(task, sub):
    """Verify a submitted mechanism + keyframes. Returns a full report (the clean verifier)."""
    rep = dict(valid_output=True, problems=[], waypoints=[], segments=[], limits=[])
    try:
        def finite(v):
            if type(v) not in (int,float) or not math.isfinite(v):raise ValueError('finite numeric value required')
            return float(v)
        mech = dict(base=[finite(v) for v in sub['mechanism']['base']] if task['free_base'] else task['base'],
                    link_lengths=[finite(v) for v in sub['mechanism']['link_lengths']] if task['free_len'] else task['link_lengths'])
        limits = [[finite(a), finite(b)] for a, b in sub['mechanism']['joint_limits_deg']]
        keys = [[finite(v) for v in k] for k in sub['keyframes']]
        wk = sub['waypoint_keyframes']
        assert isinstance(wk,list) and all(type(i)==int for i in wk)
        assert len(mech['base'])==2 and len(mech['link_lengths'])==2 and 1<=len(keys)<=MAX_KEYFRAMES
        assert len(limits) == 3 and all(len(k) == 3 for k in keys) and len(wk) == len(task['waypoints']) and len(keys) <= MAX_KEYFRAMES
    except Exception:
        return dict(valid_output=False, problems=['submission does not match the output schema'])
    if task['free_len']:
        for L, (lo, hi) in zip(mech['link_lengths'], task['link_length_range']):
            if not lo - 1e-6 <= L <= hi + 1e-6:
                rep['problems'].append(f'link length {L} outside [{lo}, {hi}]')
    if task['free_base']:
        (x0, y0), (x1, y1) = task['base_region']
        if not (x0 <= mech['base'][0] <= x1 and y0 <= mech['base'][1] <= y1):
            rep['problems'].append('base outside the mounting region')
    for j, ((lo, hi), (hlo, hhi)) in enumerate(zip(limits, task['hardware_limits_deg'])):
        ok = hlo - 1e-6 <= lo < hi <= hhi + 1e-6
        rep['limits'].append(dict(joint=j + 1, declared=[lo, hi], hardware=[hlo, hhi], ok=ok))
    for n, k in enumerate(keys):
        for j in range(3):
            if not limits[j][0] - 1e-6 <= k[j] <= limits[j][1] + 1e-6:
                rep['problems'].append(f'keyframe {n} joint {j + 1} = {k[j]} outside declared limits')
    if wk != sorted(wk) or (wk and (wk[0] < 0 or wk[-1] >= len(keys))):
        rep['problems'].append('waypoint_keyframes must be non-decreasing keyframe indices')
    else:
        for w, i in zip(task['waypoints'], wk):
            tip = fk(mech, keys[i])[3]
            err = math.dist(tip, w['position'])
            rep['waypoints'].append(dict(id=w['id'], keyframe=i, tip=[round(v, 3) for v in tip], error_m=round(err, 4), ok=err <= REACH_TOL))
    lever = _lever(mech)
    for n in range(len(keys)):
        a = keys[n]
        b = keys[n + 1] if n + 1 < len(keys) else keys[n]
        # Lipschitz bound: any link point moves at most sum(|dq_j| * lever_j) per unit t.
        sweep = sum(abs(math.radians(b[j] - a[j])) * lever[j] for j in range(3))
        steps = max(1, int(math.ceil(sweep / (CLEARANCE / 2))))
        worst, why, at = 1e9, '', 0.
        for s in range(steps + 1):
            t = s / steps
            q = [a[j] + t * (b[j] - a[j]) for j in range(3)]
            c, w = pose_clearance(mech, q, task['obstacles'],task.get('verified_link_radius_m',RADIUS))
            if c < worst:
                worst, why, at = c, w, t
        margin = worst - (sweep / steps) / 2
        rep['segments'].append(dict(from_keyframe=n, to_keyframe=min(n + 1, len(keys) - 1), min_clearance_m=round(worst, 4),
                                    certified_margin_m=round(margin, 4), closest=why, at_t=round(at, 3), ok=margin >= CLEARANCE))
        if n + 1 >= len(keys):
            break
    rep['all_ok'] = (not rep['problems'] and all(x['ok'] for x in rep['limits']) and len(rep['waypoints']) == len(task['waypoints'])
                     and all(x['ok'] for x in rep['waypoints']) and all(x['ok'] for x in rep['segments']))
    rep['mechanism'] = mech
    if task.get('balls'):
        from .pick_place import check_payload
        rep['payload']=check_payload(task,sub,mech,keys,wk)
        rep['all_ok']=rep['all_ok'] and rep['payload']['ok']
    return rep


def _ik(mech, p, elbow_up=True):
    bx, by = mech['base']
    l1, l2 = mech['link_lengths']
    dx, dy, dz = p[0] - bx, p[1] - by, p[2] - POST_H
    q1 = math.degrees(math.atan2(dy, dx))
    r = math.hypot(dx, dy)
    d2 = r * r + dz * dz
    c3 = (d2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if abs(c3) > 1:
        return None
    q3 = -math.acos(c3) if elbow_up else math.acos(c3)
    q2 = math.atan2(dz, r) - math.atan2(l2 * math.sin(q3), l1 + l2 * math.cos(q3))
    return [round(q1, 3), round(math.degrees(q2), 3), round(math.degrees(q3), 3)]


def _make_d_legacy(seed, tier):
    cfg = D_TIERS[tier]
    for attempt in range(400):
        rng = random.Random(f'D-{seed}-{tier}-{attempt}')
        room = [3.0, 3.0, 2.2]
        base = [1.5, 1.5]
        lengths = [round(rng.uniform(0.55, 0.8), 2), round(rng.uniform(0.45, 0.7), 2)]
        mech = dict(base=base, link_lengths=lengths)
        hw = [[-cfg['yaw_limit'], cfg['yaw_limit']], HW_LIMITS[1], HW_LIMITS[2]]
        # gold joint path: waypoints at sampled configs, with an "over the top" via-point between them
        configs = []
        for _ in range(cfg['waypoints']):
            configs.append([round(rng.uniform(hw[0][0] + 15, hw[0][1] - 15), 1), round(rng.uniform(-10, 35), 1), round(rng.uniform(-110, -20), 1)])
        tips = [fk(mech, q)[3] for q in configs]
        if any(t[2] < 0.15 for t in tips) or min(math.dist(a, b) for a in tips for b in tips if a is not b) < 0.35:
            continue
        keys, wk = [configs[0]], [0]
        for q in configs[1:]:
            a = keys[-1]
            if cfg.get('block_lift'):     # fold in place, swing folded, unfold at the target yaw
                keys += [VIA['tuck'](a[0]), VIA['tuck'](q[0]), q]
            else:
                keys += [VIA['lift'](round((a[0] + q[0]) / 2, 1)), q]
            wk.append(len(keys) - 1)
        obstacles = []
        ok = True
        for n in range(cfg['obstacles']):
            # block the straight joint-space path between waypoint n and n+1 at its midpoint
            a, b = configs[n % (len(configs) - 1)], configs[n % (len(configs) - 1) + 1]
            mid = fk(mech, [(a[j] + b[j]) / 2 for j in range(3)])[3]
            s = rng.uniform(0.18, 0.26)
            box = [mid[0] - s, mid[0] + s, mid[1] - s, mid[1] + s, 0., max(mid[2] + s, 0.3)]
            obstacles.append(dict(id=f'obstacle_{n + 1}', box=[round(v, 3) for v in box]))
        task = dict(experiment='D', instance=f'D-{tier}-s{seed}', tier=tier, room=room, post_height=POST_H, link_radius=RADIUS,
                    clearance_m=CLEARANCE, reach_tolerance_m=REACH_TOL, hardware_limits_deg=hw, free_base=cfg['free_base'],
                    free_len=cfg['free_len'], base=base, link_lengths=lengths, obstacles=obstacles,
                    waypoints=[dict(id=f'w{n + 1}', position=[round(v, 3) for v in t]) for n, t in enumerate(tips)])
        if cfg['free_len']:
            task['link_length_range'] = [[0.4, 0.9], [0.35, 0.8]]
            task['link_lengths'] = None
        if cfg['free_base']:
            task['base_region'] = [[1.2, 1.2], [1.8, 1.8]]
            task['base'] = None
        if cfg.get('block_lift'):
            # an overhead block where the generic "lift over" detour would put the tip, so only folding works
            a, b = configs[0], configs[1]
            top = fk(mech, VIA['lift']((a[0] + b[0]) / 2))[3]
            obstacles.append(dict(id='overhead_block', box=[round(top[0] - .2, 3), round(top[0] + .2, 3), round(top[1] - .2, 3), round(top[1] + .2, 3), round(top[2] - .25, 3), round(min(top[2] + .2, room[2]), 3)]))
        lim = [[hw[j][0], hw[j][1]] for j in range(3)]
        gold = dict(mechanism=dict(base=base, link_lengths=lengths, joint_limits_deg=lim), keyframes=keys, waypoint_keyframes=wk)
        rep = check_motion(task, gold)
        if not rep['all_ok']:
            continue
        static = static_baseline(task, gold)
        if cfg['obstacles'] and check_motion(task, static)['all_ok']:
            continue            # obstacles must actually defeat the "pose-only" solution
        if cfg.get('block_lift') and check_motion(task, lift_baseline(task, gold))['all_ok']:
            continue            # and the generic lift-over detour
        if any(x['ok'] is False for x in check_motion(task, gold)['segments']):
            ok = False
        if ok:
            return task, dict(gold=gold, gold_report=rep)
    raise RuntimeError('D generation failed')


def _rebuild(task, gold, via):
    """Same waypoints (and home pose / events), different motion between them: via(prev, next) -> intermediate keyframes."""
    home = [gold['keyframes'][0]] if task.get('home') else []
    targets = [gold['keyframes'][i] for i in gold['waypoint_keyframes']] + ([gold['keyframes'][-1]] if task.get('return_home') else [])
    keys, wk = list(home), []
    for n, q in enumerate(targets):
        if keys:
            keys += via(keys[-1], q)
        keys.append(q)
        if n < len(gold['waypoint_keyframes']):
            wk.append(len(keys) - 1)
    out = dict(mechanism=gold['mechanism'], keyframes=keys, waypoint_keyframes=wk)
    if gold.get('events'):
        out['events'] = [dict(e, keyframe=wk[gold['waypoint_keyframes'].index(e['keyframe'])]) for e in gold['events']]
    return out


def static_baseline(task, gold):
    """Pose-only cheat: reach every waypoint but move directly between them (no motion planning)."""
    return _rebuild(task, gold, lambda p, q: [])


def lift_baseline(task, gold):
    """Generic detour: lift over the obstacle between every pair of waypoints."""
    return _rebuild(task, gold, lambda p, q: [VIA['lift'](round((p[0] + q[0]) / 2, 1))])


def _release_at(task, gold, z):
    """The gold with the first release pose moved to height z over the same destination."""
    out = copy.deepcopy(gold)
    r = next(e['keyframe'] for e in gold['events'] if e['action'] == 'release')
    ball = next(b for b in task['balls'] if b['id'] == next(e['ball'] for e in gold['events'] if e['action'] == 'release'))
    mech = dict(base=gold['mechanism']['base'], link_lengths=gold['mechanism']['link_lengths'])
    q = _ik(mech, [ball['destination'][0], ball['destination'][1], z])
    if q:
        out['keyframes'][r] = q
    return out


def release_above_support(task, gold):
    """Negative probe: ball_1 is let go 0.10 m above its cabinet top (no contact) instead of being set down on it."""
    return _release_at(task, gold, task['balls'][0]['destination'][2] + 0.10)


def release_at_other_cabinet_height(task, gold):
    """Negative probe: ball_1 is released at the resting height of a different cabinet."""
    b = task['balls'][0]
    other = min((o for o in task['obstacles'] if o['kind'] == 'floor_cabinet' and o['id'] != b['support']), key=lambda o: o['box'][5])
    return _release_at(task, gold, other['box'][5] + b['radius_m'])


def revisit_placed_ball(task, gold):
    """Negative probe: after placing ball_1 the arm goes back down into it before continuing."""
    r = next(e['keyframe'] for e in gold['events'] if e['action'] == 'release')
    keys = gold['keyframes'][:r + 2] + [gold['keyframes'][r], gold['keyframes'][r + 1]] + gold['keyframes'][r + 2:]
    shift = lambda i: i + 2 if i > r + 1 else i
    return dict(gold, keyframes=keys, waypoint_keyframes=[shift(i) for i in gold['waypoint_keyframes']],
                events=[dict(e, keyframe=shift(e['keyframe'])) for e in gold['events']])


def score_d(task, sub):
    rep = check_motion(task, sub)
    if not rep.get('valid_output'):
        return dict(success=False, reason='invalid_output', report=rep)
    poses_ok = len(rep['waypoints']) == len(task['waypoints']) and all(w['ok'] for w in rep['waypoints'])
    return dict(success=rep['all_ok'], waypoints_ok=sum(w['ok'] for w in rep['waypoints']), waypoints=len(task['waypoints']),
                segments_ok=sum(s['ok'] for s in rep['segments']), segments=len(rep['segments']),
                joint_travel_deg=round(sum(sum(abs(b[j]-a[j]) for j in range(3)) for a,b in zip(sub['keyframes'],sub['keyframes'][1:])),4),
                min_certified_clearance_m=min((s['certified_margin_m'] for s in rep['segments']),default=None),
                limits_ok=all(x['ok'] for x in rep['limits']) and not any('outside declared' in p for p in rep['problems']),
                payload_ok=rep.get('payload',{}).get('ok'),payload_report=rep.get('payload'),
                static_only=poses_ok and not rep['all_ok'], problems=rep['problems'])


@lru_cache(maxsize=64)
def _certified_d(seed,tier):
    from .pick_place import make
    return make(seed,tier)

def make_d(seed,tier):
    return copy.deepcopy(_certified_d(seed,tier))
