"""Tasks 1-3 of the iteration-3 redesign, on realistic furniture (Kenney CC0 meshes for rendering).

Task 1  multi-constraint scene generation, answered with a scene PROGRAM (execution success counts).
Task 2  iterative repair of a scene program with three injected errors (misplaced object,
        interpenetration, coupled trap); every step logs the constraint state vector.
Task 3  multi-view reconstruction with hidden-view verification and a composite table; reports
        R_visible (views the agent saw) against R_3D (hidden views, depth, centres).

Scoring always uses exact axis-aligned boxes; meshes are only drawn inside those boxes for images.
"""
import copy
import itertools
import math
import random

from . import core, raycast
from .. import program

# id: (base size [x, y, z] at yaw 0 (front faces -y), Kenney model, role)
#   role: floor = stands on the floor; small = may stand on a host; host = has a top other things may rest on
FURNITURE = {
    'sofa': ([1.80, 0.85, 0.80], 'loungeSofa', 'floor'),
    'armchair': ([0.85, 0.80, 0.80], 'loungeChair', 'floor'),
    'coffee_table': ([1.00, 0.55, 0.40], 'tableCoffee', 'host'),
    'dining_table': ([1.40, 0.80, 0.75], 'table', 'host'),
    'chair': ([0.45, 0.50, 0.90], 'chair', 'floor'),
    'bookcase': ([0.90, 0.35, 1.80], 'bookcaseOpen', 'floor'),
    'tv_cabinet': ([1.40, 0.45, 0.50], 'cabinetTelevision', 'host'),
    'floor_lamp': ([0.35, 0.35, 1.50], 'lampRoundFloor', 'floor'),
    'plant': ([0.40, 0.40, 0.90], 'pottedPlant', 'floor'),
    'fridge': ([0.70, 0.70, 1.80], 'kitchenFridge', 'floor'),
    'kitchen_cabinet': ([0.80, 0.60, 0.90], 'kitchenCabinet', 'host'),
    'tv': ([1.00, 0.15, 0.60], 'televisionModern', 'small'),
    'books': ([0.30, 0.20, 0.25], 'books', 'small'),
}
PALETTE = ['red', 'blue', 'green', 'yellow', 'white', 'orange', 'purple', 'gray', 'brown', 'black', 'pink', 'cyan']
SCALES = [0.85, 1.0, 1.15]
HEIGHT = 2.5
GRID = 0.05


def _snap(v):
    return round(round(v / GRID) * GRID, 4)


def _scaled(base, s):
    return [round(v * s, 3) for v in base]


def reference_camera(room):
    from .grounding import cameras      # lazy: grounding imports constraint_graph
    return cameras(room, ['reference'])['reference']


# ------------------------------------------------------------------ gold layouts
def furniture_layout(rng, n, density, n_small, yaw=True, contact=0.4):
    """Random valid furniture layout. density = footprint area / floor area (higher = tighter room).
    Returns (room, objs, catalogue entries with size and colour options)."""
    floor_ids = [i for i, v in FURNITURE.items() if v[2] != 'small']
    small_ids = [i for i, v in FURNITURE.items() if v[2] == 'small']
    n_small = min(n_small, len(small_ids))
    ids = rng.sample(floor_ids, n - n_small) + rng.sample(small_ids, n_small)
    hosts_needed = n_small
    if hosts_needed and not any(FURNITURE[i][2] == 'host' for i in ids[:n - n_small]):
        host = rng.choice([i for i in floor_ids if FURNITURE[i][2] == 'host'])
        ids[0] = host if host not in ids else ids[0]
    entries, sizes = {}, {}
    for i in ids:
        s = rng.choice(SCALES)
        cols = rng.sample(PALETTE, 3)
        entries[i] = dict(id=i, model=FURNITURE[i][1], size_options=[_scaled(FURNITURE[i][0], k) for k in SCALES], colors=cols)
        sizes[i] = (_scaled(FURNITURE[i][0], s), rng.choice(cols))
    area = sum(sizes[i][0][0] * sizes[i][0][1] for i in ids if FURNITURE[i][2] != 'small')
    floor = area / density
    W = _snap(max(2.6, math.sqrt(floor * 1.3)))
    D = _snap(max(2.2, floor / W))
    room = [W, D, HEIGHT]
    objs = {}
    order = sorted([i for i in ids if FURNITURE[i][2] != 'small'], key=lambda i: -sizes[i][0][0] * sizes[i][0][1])
    for i in order:
        size, col = sizes[i]
        for _ in range(600):
            y = rng.choice([0, 90, 180, 270]) if yaw else 0
            o = dict(id=i, size=size, position=[0, 0, 0], yaw=y, color=col)
            fx, fy = core.footprint_size(o)
            if objs and rng.random() < contact:
                other = objs[rng.choice(list(objs))]
                b = core.aabb(other)
                side = rng.choice('LRFB')
                cx = {'L': b[0] - fx / 2, 'R': b[1] + fx / 2}.get(side, rng.uniform(b[0], b[1]))
                cy = {'F': b[2] - fy / 2, 'B': b[3] + fy / 2}.get(side, rng.uniform(b[2], b[3]))
                o['position'] = [round(cx, 4), round(cy, 4), 0.]
            else:
                o['position'] = [_snap(rng.uniform(fx / 2, W - fx / 2)), _snap(rng.uniform(fy / 2, D - fy / 2)), 0.]
            b = core.aabb(o)
            if b[0] < 0 or b[2] < 0 or b[1] > W or b[3] > D:
                continue
            if all(core.penetration(o, p) <= 0 for p in objs.values()) and all(core.gap(o, p) >= 0.15 or core.gap(o, p) <= 1e-6 for p in objs.values()):
                objs[i] = o
                break
        else:
            raise RuntimeError('layout failed')
    hosts = [i for i in objs if FURNITURE[i][2] == 'host']
    for i in [i for i in ids if FURNITURE[i][2] == 'small']:
        size, col = sizes[i]
        rng.shuffle(hosts)
        for h in hosts:
            b = core.aabb(objs[h])
            o = dict(id=i, size=size, yaw=objs[h]['yaw'] if objs[h]['yaw'] in (0, 180) else 0, color=col,
                     position=[round((b[0] + b[1]) / 2, 4), round((b[2] + b[3]) / 2, 4), b[5]])
            fx, fy = core.footprint_size(o)
            if fx <= b[1] - b[0] + 1e-6 and fy <= b[3] - b[2] + 1e-6 and b[5] + size[2] <= HEIGHT and all(core.penetration(o, p) <= 0 for p in objs.values()):
                objs[i] = o
                break
        else:
            raise RuntimeError('no host')
    if core.global_checks(objs, room):
        raise RuntimeError('invalid layout')
    return room, objs, [entries[i] for i in ids]


# ------------------------------------------------------------------ Task 1
T1_TIERS = {
    'L1': dict(n=5, k=10, density=0.20, small=0, yaw=False, occludes=0, types=['left_of', 'right_of', 'in_front_of', 'behind', 'touching', 'on_floor', 'x_less', 'abs_dx_lt', 'disjoint', 'color_is', 'taller_than']),
    'L2': dict(n=8, k=18, density=0.27, small=1, yaw=True, occludes=1, types=['left_of', 'right_of', 'in_front_of', 'behind', 'touching', 'on_floor', 'x_less', 'abs_dx_lt', 'disjoint', 'color_is', 'taller_than', 'on_top_of', 'yaw_is', 'facing', 'occludes', 'between']),
    'L3': dict(n=11, k=30, density=0.33, small=2, yaw=True, occludes=2, types=['left_of', 'right_of', 'in_front_of', 'behind', 'touching', 'on_floor', 'x_less', 'abs_dx_lt', 'disjoint', 'color_is', 'taller_than', 'on_top_of', 'yaw_is', 'facing', 'occludes', 'between', 'distance']),
}
# minimum count per type, so every Mike-requested family is exercised
T1_QUOTA = {'abs_dx_lt': 1, 'on_floor': 1, 'disjoint': 1, 'touching': 1, 'color_is': 2, 'taller_than': 1, 'x_less': 1, 'yaw_is': 1}


def _t1_candidates(rng, objs, entries, types, ctx):
    ent = {e['id']: e for e in entries}
    out = []
    ids = list(objs)
    for a, b in itertools.permutations(ids, 2):
        A, B = objs[a], objs[b]
        for t in ('left_of', 'right_of', 'in_front_of', 'behind', 'on_top_of', 'facing', 'x_less'):
            if t in types and core.RELATIONS[t][0](A, B)[0]:
                if t == 'x_less' and abs(core.center(A)[0] - core.center(B)[0]) < 0.1:
                    continue
                out.append(dict(type=t, args=[a, b]))
        if a < b and 'touching' in types and core.rel_touching(A, B)[0]:
            out.append(dict(type='touching', args=[a, b]))
        if a < b and 'disjoint' in types and 0.01 < core.gap(A, B) < 0.6:
            out.append(dict(type='disjoint', args=[a, b]))
        if a < b and 'abs_dx_lt' in types:
            dx = abs(core.center(A)[0] - core.center(B)[0])
            if dx < 0.8:
                out.append(dict(type='abs_dx_lt', args=[a, b], range=[round(max(0.1, math.ceil((dx + rng.uniform(0.05, 0.25)) * 20) / 20), 2)]))
        if 'distance' in types and a < b:
            d = core.rel_distance(A, B, 0, 99)[1]
            out.append(dict(type='distance', args=[a, b], range=[round(max(0, d - 0.15), 2), round(d + 0.15, 2)]))
        if 'taller_than' in types and core.rel_taller_than(A, B)[0]:
            ha = [s[2] for s in ent[a]['size_options']]
            hb = [s[2] for s in ent[b]['size_options']]
            if min(ha) <= max(hb):                     # only when it actually constrains the size choice
                out.append(dict(type='taller_than', args=[a, b]))
        if 'occludes' in types and core.center(A)[1] < core.center(B)[1] and core.aabb(A)[4] <= core.aabb(B)[5]:
            ba, bb = core.aabb(A), core.aabb(B)
            if min(ba[1], bb[1]) - max(ba[0], bb[0]) > 0.1 and core.rel_occludes(A, B, ctx)[0]:
                out.append(dict(type='occludes', args=[a, b]))
    for a in ids:
        if 'on_floor' in types and core.rel_on_floor(objs[a])[0]:
            out.append(dict(type='on_floor', args=[a]))
        if 'color_is' in types:
            out.append(dict(type='color_is', args=[a], range=[objs[a]['color']]))
        if 'yaw_is' in types and objs[a]['yaw']:
            out.append(dict(type='yaw_is', args=[a], range=[objs[a]['yaw']]))
    if 'between' in types:
        for c, a, b in itertools.permutations(ids, 3):
            if a < b and core.rel_between(objs[c], objs[a], objs[b])[0] and core.gap(objs[a], objs[b]) > 0.6:
                out.append(dict(type='between', args=[c, a, b]))
    return out


def _t1_select(rng, objs, cands, k, cfg):
    chosen, keys = [], set()

    def key(c):
        return (c['type'], tuple(c['args']))

    def take(c):
        if key(c) not in keys and len(chosen) < k:
            chosen.append(c)
            keys.add(key(c))

    pool = cands[:]
    rng.shuffle(pool)
    occ = [c for c in pool if c['type'] == 'occludes']
    for c in occ[:cfg['occludes']]:
        take(c)
    for axis in (('left_of', 'right_of', 'x_less'), ('in_front_of', 'behind')):
        pick = [c for c in pool if c['type'] in axis]
        if pick:
            take(pick[0])
    for t, m in T1_QUOTA.items():
        for c in [c for c in pool if c['type'] == t][:m]:
            take(c)
    pair_seen = {}
    while len(chosen) < k:
        rest = [c for c in pool if key(c) not in keys]
        if not rest:
            break
        counts = {}
        for c in chosen:
            counts[c['type']] = counts.get(c['type'], 0) + 1
        for c in chosen:
            for o in c['args']:
                pair_seen[o] = pair_seen.get(o, 0) + 1
        take(max(rest, key=lambda c: 2 / (1 + counts.get(c['type'], 0)) + 1.5 / (1 + min(pair_seen.get(o, 0) for o in c['args'])) + rng.random()))
    for n, c in enumerate(chosen):
        c['id'] = f'c{n + 1:02d}'
    return chosen


def _mirror_breaks(task, objs, cons):
    room = task['room']
    ctx = dict(room=room, camera=task['camera'])
    for fx, fy in ((True, False), (False, True), (True, True)):
        m = {}
        for i, o in objs.items():
            p = o['position']
            m[i] = dict(o, position=[room[0] - p[0] if fx else p[0], room[1] - p[1] if fy else p[1], p[2]])
        if all(core.evaluate(c, m, ctx)[0] for c in cons):
            return False
    return True


def make_t1(seed, tier):
    """Task 1 instance. Returns (task, priv)."""
    cfg = T1_TIERS[tier]
    for attempt in range(200):
        rng = random.Random(f'T1-{seed}-{tier}-{attempt}')
        try:
            room, objs, entries = furniture_layout(rng, cfg['n'], cfg['density'], cfg['small'], cfg['yaw'])
        except RuntimeError:
            continue
        cam = reference_camera(room)
        ctx = dict(room=room, camera=cam)
        cands = _t1_candidates(rng, objs, entries, cfg['types'], ctx)
        if sum(c['type'] == 'occludes' for c in cands) < cfg['occludes']:
            continue
        cons = _t1_select(rng, objs, cands, cfg['k'], cfg)
        if len(cons) < cfg['k'] or {a for c in cons for a in c['args']} != set(objs):
            continue
        task = dict(experiment='T1', instance=f'T1-{tier}-s{seed}', tier=tier, room=room, objects=entries, camera=cam,
                    constraints=cons, relation_types=sorted({c['type'] for c in cons}), uses_yaw=cfg['yaw'])
        if not _mirror_breaks(task, objs, cons):
            continue
        gold = core.placements_of(objs, with_size=True)
        rep = core.check_scene(task, gold)
        assert rep['all_satisfied'], rep
        return task, dict(gold=gold, gold_program=program.literal_program(gold))
    raise RuntimeError('T1 generation failed')


def run_program(task, source):
    """Execute a scene program in the restricted interpreter and verify its output."""
    try:
        placements = program.execute(source, public_params=dict(room=task['room']))
    except program.ProgramError as e:
        return dict(executed=False, error=str(e)[:300]), None
    rep = core.check_scene(task, placements)
    return dict(executed=True, report=rep), placements


def state_vector(task, rep):
    """Constraint state vector: one bit per constraint plus one bit 'physically valid'."""
    if not rep or not rep.get('valid_output'):
        return None
    return [int(r['satisfied']) for r in rep['constraints']] + [int(not rep['physical'])]


def score_t1(task, history):
    """history: list of (executed, report) per program run, the last one being the submission."""
    if not history:
        return dict(success=False, reason='no_submission', execution_success=False, csr=0., collision_free=False, iterations_to_success=None)
    ex, rep = history[-1]
    its = next((n + 1 for n, (e, r) in enumerate(history) if e and r.get('all_satisfied')), None)
    if not ex or not rep.get('valid_output'):
        return dict(success=False, reason='execution_failed' if not ex else 'invalid_output', execution_success=False, csr=0.,
                    collision_free=False, iterations_to_success=its, programs_run=len(history))
    sat = sum(r['satisfied'] for r in rep['constraints'])
    return dict(success=bool(rep['all_satisfied']), execution_success=True, satisfied=sat, total=len(rep['constraints']),
                csr=round(sat / len(rep['constraints']), 4),
                collision_free=not any(p['check'] == 'no_interpenetration' for p in rep['physical']),
                physical=rep['physical'], iterations_to_success=its, programs_run=len(history),
                failed=[r['id'] for r in rep['constraints'] if not r['satisfied']])


# ------------------------------------------------------------------ Task 2
T2_TIERS = {'L1': dict(base='L1', traps=1), 'L2': dict(base='L2', traps=1), 'L3': dict(base='L3', traps=2)}
AXIS_STEPS = [round(s * 0.05, 2) for s in range(1, 61)]


def _fixed_task(task, gold):
    """T2 objects have their sizes and colours fixed to the gold choice."""
    t = copy.deepcopy(task)
    t['objects'] = [dict(id=o['id'], model=o['model'], size=gold[o['id']]['size'], color=gold[o['id']]['color']) for o in task['objects']]
    return t


def _sat(task, pl):
    rep = core.check_scene(task, pl)
    return rep, {r['id'] for r in rep['constraints'] if r['satisfied']}, rep.get('physical', [])


def _move(pl, oid, dx=0., dy=0.):
    out = copy.deepcopy(pl)
    p = out[oid]['position']
    p[0], p[1] = round(p[0] + dx, 4), round(p[1] + dy, 4)
    return out


def _dependents(task, oid):
    """Objects resting on oid move with it."""
    return [c['args'][0] for c in task['constraints'] if c['type'] == 'on_top_of' and c['args'][1] == oid]


def _move_with(task, pl, oid, dx, dy):
    out = _move(pl, oid, dx, dy)
    for d in _dependents(task, oid):
        out = _move(out, d, dx, dy)
    return out


def naive_fix(task, pl, cid):
    """The 'local' repair an agent makes first: translate the constraint's FIRST argument along one axis by the
    smallest step that satisfies the constraint without a physical violation. Returns new placements or None."""
    c = next(c for c in task['constraints'] if c['id'] == cid)
    a = c['args'][0]
    best = None
    for s in AXIS_STEPS:
        for dx, dy in ((s, 0), (-s, 0), (0, s), (0, -s)):
            cand = _move_with(task, pl, a, dx, dy)
            rep = core.check_scene(task, cand)
            if not rep['valid_output'] or rep['physical']:
                continue
            if next(r for r in rep['constraints'] if r['id'] == cid)['satisfied']:
                return cand
        if best:
            break
    return best


def make_t2(seed, tier):
    """Task 2: a scene program with three injected errors. Returns (task, priv)."""
    cfg = T2_TIERS[tier]
    for attempt in range(80):
        rng = random.Random(f'T2-{seed}-{tier}-{attempt}')
        try:
            t1, p1 = make_t1(seed * 7919 + 13 + attempt, cfg['base'])
        except RuntimeError:
            continue
        gold = p1['gold']
        task = _fixed_task(t1, gold)
        all_ids = {c['id'] for c in task['constraints']}
        scene = copy.deepcopy(gold)
        used, errors = set(), []
        movable = [i for i in gold if not _dependents(task, i) and core.aabb(dict(size=gold[i]['size'], position=gold[i]['position'], yaw=gold[i]['yaw']))[4] < 1e-6]
        ok = True
        # (3) coupled traps first: move Y, so that constraint c(A, Y) breaks; naive fix moves A and breaks another constraint
        for _ in range(cfg['traps']):
            found = None
            trap_cons = [c for c in task['constraints'] if len(c['args']) == 2 and c['type'] in
                         ('left_of', 'right_of', 'in_front_of', 'behind', 'x_less', 'abs_dx_lt', 'touching', 'distance', 'facing')]
            rng.shuffle(trap_cons)
            for c in trap_cons:
                A, Y = c['args']
                if A in used or Y in used or Y not in movable or A not in movable:
                    continue
                _, sat0, _ = _sat(task, scene)
                for _ in range(12):
                    dx, dy = rng.choice([(rng.choice([-1, 1]) * rng.choice(AXIS_STEPS[4:24]), 0), (0, rng.choice([-1, 1]) * rng.choice(AXIS_STEPS[4:24]))])
                    cand = _move(scene, Y, dx, dy)
                    rep, sat, phys = _sat(task, cand)
                    broken = sat0 - sat
                    if phys or broken != {c['id']}:
                        continue
                    fixed = naive_fix(task, cand, c['id'])
                    if fixed is None:
                        continue
                    _, sat2, _ = _sat(task, fixed)
                    regress = sat - sat2
                    if regress:
                        found = (c, Y, A, cand, sorted(regress))
                        break
                if found:
                    break
            if not found:
                ok = False
                break
            c, Y, A, scene, regress = found
            used |= {Y, A}
            errors.append(dict(kind='coupled_trap', constraint=c['id'], moved_object=Y, naive_subject=A,
                               naive_fix_breaks=regress, fix=f'move {Y} back; moving {A} breaks {", ".join(regress)}'))
        if not ok:
            continue
        # (1) misplaced object: a relational error with no physical symptom
        cands = [i for i in movable if i not in used]
        rng.shuffle(cands)
        hit = None
        _, sat0, _ = _sat(task, scene)
        for m in cands:
            for _ in range(40):
                dx, dy = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
                cand = _move(scene, m, _snap(dx), _snap(dy))
                rep, sat, phys = _sat(task, cand)
                if not phys and len(sat0 - sat) >= 1 and rep['valid_output']:
                    hit = (m, cand, sorted(sat0 - sat))
                    break
            if hit:
                break
        if not hit:
            continue
        m, scene, br = hit
        used.add(m)
        errors.append(dict(kind='misplaced', moved_object=m, breaks=br))
        # (2) interpenetration: push one object into a neighbour
        hit = None
        cands = [i for i in movable if i not in used]
        rng.shuffle(cands)
        for p in cands:
            others = sorted((j for j in gold if j != p and j not in used), key=lambda j: math.dist(gold[p]['position'][:2], gold[j]['position'][:2]))
            for q in others[:3]:
                vec = [gold[q]['position'][k] - scene[p]['position'][k] for k in (0, 1)]
                n = math.hypot(*vec) or 1.
                for f in (0.25, 0.35, 0.45, 0.55, 0.65):
                    d = n * f
                    cand = _move(scene, p, round(vec[0] / n * d, 3), round(vec[1] / n * d, 3))
                    rep, sat, phys = _sat(task, cand)
                    pen = [x for x in phys if x['check'] == 'no_interpenetration']
                    if rep['valid_output'] and pen and all(x['check'] == 'no_interpenetration' for x in phys) and max(x['depth_m'] for x in pen) >= 0.08:
                        hit = (p, q, cand)
                        break
                if hit:
                    break
            if hit:
                break
        if not hit:
            continue
        p, q, scene = hit
        used.add(p)
        errors.append(dict(kind='interpenetration', moved_object=p, into=q))
        rep, sat, phys = _sat(task, scene)
        if rep['all_satisfied']:
            continue
        # certify: gold repairs everything
        assert core.check_scene(task, gold)['all_satisfied']
        task.update(experiment='T2', instance=f'T2-{tier}-s{seed}', tier=tier)
        task['initial_program'] = scene_program(task, scene)
        priv = dict(gold=gold, initial=scene, injected_errors=errors, initial_violations=sorted(all_ids - sat),
                    initial_physical=phys, source_t1=t1['instance'])
        return task, priv
    raise RuntimeError('T2 generation failed')


def scene_program(task, placements):
    """The repair input: a Python scene file in the bpy-like style the benchmark's restricted interpreter runs."""
    lines = ['def build(public_params):',
             '    """Scene file: one entry per object (Blender-style: location = [x, y, bottom z], rotation_z = yaw in degrees).',
             '    Edit the numbers and return the full scene; the benchmark executes this function, not Blender."""',
             '    return {']
    for o in task['objects']:
        p = placements[o['id']]
        lines.append(f'        "{o["id"]}": {{"position": [{p["position"][0]}, {p["position"][1]}, {p["position"][2]}], "yaw": {int(p.get("yaw", 0))}}},')
    lines.append('    }')
    return '\n'.join(lines) + '\n'


def trajectory_metrics(vectors):
    """vectors: list of state vectors (None for a failed run) from the initial scene onward.
    Regression Rate = (# previously-satisfied entries that flip to 0) / (# previously-satisfied entries re-checked),
    summed over consecutive valid steps."""
    broken = checked = 0
    flips = []
    prev = None
    for n, v in enumerate(vectors):
        if v is None:
            continue
        if prev is not None:
            pv, pidx = prev
            ones = [k for k, b in enumerate(pv) if b]
            lost = [k for k in ones if not v[k]]
            broken += len(lost)
            checked += len(ones)
            if lost:
                flips.append(dict(step=n, lost=lost))
        prev = (v, n)
    return dict(regression_rate=round(broken / checked, 4) if checked else 0., regressions=broken, rechecked=checked,
                monotonic=not flips, regression_steps=flips)


def score_t2(task, priv, vectors, final_placements, final_executed):
    tm = trajectory_metrics(vectors)
    if not final_executed or final_placements is None:
        return dict(success=False, reason='execution_failed' if not final_executed else 'no_submission', execution_success=bool(final_executed), **tm)
    rep = core.check_scene(task, final_placements)
    if not rep['valid_output']:
        return dict(success=False, reason='invalid_output', execution_success=True, problems=rep['output_problems'], **tm)
    sat = sum(r['satisfied'] for r in rep['constraints'])
    edit = core.edit_cost(priv['initial'], final_placements)
    needed = {e['moved_object'] for e in priv['injected_errors']}
    return dict(success=bool(rep['all_satisfied']), execution_success=True, satisfied=sat, total=len(rep['constraints']),
                csr=round(sat / len(rep['constraints']), 4), physical=rep['physical'],
                collision_free=not any(p['check'] == 'no_interpenetration' for p in rep['physical']),
                moved=edit['moved'], collateral_moves=sorted(set(edit['moved']) - needed),
                failed=[r['id'] for r in rep['constraints'] if not r['satisfied']], **tm)


# ------------------------------------------------------------------ Task 3
T3_TIERS = {'L1': dict(n=3, queries=0, yaw=False, small=0, density=0.14),
            'L2': dict(n=4, queries=1, yaw=True, small=0, density=0.16),
            'L3': dict(n=6, queries=2, yaw=True, small=1, density=0.18)}
T3_PASS_MEAN, T3_PASS_MIN = 0.80, 0.60
PART_RANGE = [[0.02, 0.02, 0.02], [2.0, 2.0, 1.2]]
LEG = 0.06


def table_parts(obj):
    """Composite table: a 0.05 m top on four square legs, inside the table's box."""
    b = core.aabb(obj)
    top = 0.05
    h = b[5] - b[4] - top
    cx, cy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
    parts = [dict(size=[round(b[1] - b[0], 3), round(b[3] - b[2], 3), top], position=[round(cx, 4), round(cy, 4), round(b[4] + h, 4)])]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(dict(size=[LEG, LEG, round(h, 3)], position=[round(cx + sx * ((b[1] - b[0]) / 2 - LEG), 4), round(cy + sy * ((b[3] - b[2]) / 2 - LEG), 4), b[4]]))
    return parts


def make_t3(seed, tier):
    from .grounding import cameras, PUBLIC_VIEWS, HIDDEN_VIEWS
    cfg = T3_TIERS[tier]
    for attempt in range(80):
        rng = random.Random(f'T3-{seed}-{tier}-{attempt}')
        try:
            room, objs, entries = furniture_layout(rng, cfg['n'], cfg['density'], cfg['small'], cfg['yaw'], contact=0.25)
        except RuntimeError:
            continue
        if 'dining_table' not in objs and 'coffee_table' not in objs:
            continue
        table = 'dining_table' if 'dining_table' in objs else 'coffee_table'
        if any(FURNITURE[i][2] == 'small' and core.rel_on_top_of(objs[i], objs[table])[0] for i in objs):
            continue                                 # keep the composite table's top clear
        gold, task_objects = {}, []
        for e in entries:
            o = objs[e['id']]
            if e['id'] == table:
                gold[table] = dict(parts=table_parts(o))
                task_objects.append(dict(id=table, model=e['model'], parts=5, part_size_range=PART_RANGE, color=o['color'],
                                         note='composite: give 5 boxes (any order) that together form the table (top + legs)'))
            else:
                base = o['size']
                task_objects.append(dict(id=e['id'], model=e['model'], color=o['color'], size=base,
                                         size_range=[[round(v * 0.8, 3) for v in base], [round(v * 1.2, 3) for v in base]]))
                gold[e['id']] = dict(position=o['position'], yaw=o['yaw'], size=base)
        task = dict(experiment='T3', instance=f'T3-{tier}-s{seed}', tier=tier, room=room, objects=task_objects, constraints=[],
                    composite=table, uses_yaw=cfg['yaw'], query_budget=cfg['queries'],
                    public_cameras=cameras(room, PUBLIC_VIEWS), queryable_views=[v for v in PUBLIC_VIEWS if v != 'reference'])
        gobjs, probs = core.build_objects(task, gold)
        assert not probs, probs
        if core.global_checks(gobjs, room):
            continue
        vis = {}
        for n, cam in task['public_cameras'].items():
            img = raycast.render(gobjs, room, cam, scale=2)
            for i in img['ids']:
                if i >= 0:
                    k = raycast.base_name(img['names'][i])
                    vis[k] = vis.get(k, 0) + 1
        if len(vis) < len(entries) or min(vis.values()) < 40:
            continue
        return task, dict(gold=gold, hidden_cameras=cameras(room, HIDDEN_VIEWS), public_visibility_px=vis)
    raise RuntimeError('T3 generation failed')


def _views(task, gobjs, objs, cams):
    from .grounding import SCALE_EVAL
    return {n: raycast.compare(raycast.render(gobjs, task['room'], c, SCALE_EVAL), raycast.render(objs, task['room'], c, SCALE_EVAL)) for n, c in cams.items()}


def score_t3(task, priv, placements, seen_views=('reference',)):
    """R_visible: silhouette agreement on views the agent saw (what a single-view judge rewards).
    R_3D: hidden-view agreement, depth error and centre error (what the scene actually is)."""
    objs, problems = core.build_objects(task, placements)
    if problems:
        return dict(success=False, reason='invalid_output', problems=problems)
    gobjs, _ = core.build_objects(task, priv['gold'])
    physical = core.global_checks(objs, task['room'])
    hidden = _views(task, gobjs, objs, priv['hidden_cameras'])
    seen = _views(task, gobjs, objs, {n: task['public_cameras'][n] for n in seen_views})
    names = [o['id'] for o in task['objects']]
    per_obj_hidden = {}
    for n in names:         # views where the object is invisible in both renders carry no evidence
        vals = [v['object_iou'][n] for v in hidden.values() if n in v['object_iou']]
        per_obj_hidden[n] = round(sum(vals) / len(vals), 4) if vals else 0.
    r3d = sum(per_obj_hidden.values()) / len(names)
    rvis = sum(v['mean_object_iou'] for v in seen.values()) / len(seen)

    def centre(oid, os):
        bs = [core.aabb(o) for k, o in os.items() if raycast.base_name(k) == oid]
        return [(min(b[0] for b in bs) + max(b[1] for b in bs)) / 2, (min(b[2] for b in bs) + max(b[3] for b in bs)) / 2, (min(b[4] for b in bs) + max(b[5] for b in bs)) / 2]
    err = {n: round(math.dist(centre(n, objs), centre(n, gobjs)), 4) for n in names}
    depth = [v['depth_abs_err'] for v in hidden.values() if v['depth_abs_err'] is not None]
    ok = not physical and r3d >= T3_PASS_MEAN and min(per_obj_hidden.values()) >= T3_PASS_MIN
    return dict(success=ok, physical=physical, R_visible=round(rvis, 4), R_3D=round(r3d, 4), hack_gap=round(rvis - r3d, 4),
                min_object_hidden_iou=min(per_obj_hidden.values()), per_object_hidden_iou=per_obj_hidden,
                hidden_depth_err_m=round(sum(depth) / len(depth), 4) if depth else None,
                centre_err_m=err, mean_centre_err_m=round(sum(err.values()) / len(err), 4),
                per_view={n: round(v['mean_object_iou'], 4) for n, v in {**seen, **hidden}.items()})


def billboard(task, box, eye):
    """A 2 cm sheet on the camera-facing side of a box, tall and wide enough to cover the box's whole
    silhouette from `eye` (top face included). Looks right from the reference view only."""
    b = box
    yf = b[2]
    zs, xs = [], []
    for x in (b[0], b[1]):
        for y in (b[2], b[3]):
            for z in (b[4], b[5]):
                t = (yf - eye[1]) / (y - eye[1]) if abs(y - eye[1]) > 1e-9 else 1.
                xs.append(eye[0] + t * (x - eye[0]))
                zs.append(eye[2] + t * (z - eye[2]))
    x0, x1 = max(0.01, min(xs)), min(task['room'][0] - 0.01, max(xs))
    z1 = min(task['room'][2], max(zs))
    return dict(size=[round(x1 - x0, 3), 0.02, round(z1, 3)], position=[round((x0 + x1) / 2, 4), round(yf + 0.01, 4), 0.])


def facade_table_baseline(task, priv):
    """Mike's hack: the table top becomes a vertical plane and the legs thin front sheets, all on the
    camera-facing side; other objects are left exactly at gold."""
    eye = task['public_cameras']['reference']['eye']
    out = copy.deepcopy(priv['gold'])
    tid = task['composite']
    gobjs, _ = core.build_objects(task, priv['gold'])
    parts = [o for k, o in gobjs.items() if raycast.base_name(k) == tid]
    top = max(parts, key=lambda o: o['position'][2])
    legs = [o for o in parts if o is not top]
    tb = core.aabb(top)
    yf = min(core.aabb(o)[2] for o in parts)
    sheet = billboard(task, [tb[0], tb[1], yf, tb[3], tb[4], tb[5]], eye)
    lo = tb[4]
    sheet = dict(size=[sheet['size'][0], 0.02, round(max(0.05, sheet['size'][2] - lo), 3)], position=[sheet['position'][0], round(yf + 0.01, 4), round(lo, 4)])
    new = [sheet]
    front = sorted(legs, key=lambda o: o['position'][1])[:2]
    for leg in legs:      # back-leg sheets sit 1 cm behind the front ones, so the sheets never overlap
        lb = core.aabb(leg)
        y = yf + 0.01 if leg in front else yf + 0.04
        new.append(dict(size=[LEG, 0.02, round(lo, 3)], position=[round((lb[0] + lb[1]) / 2, 4), round(y, 4), 0.]))
    # the floating sheet needs support: rest it on the front legs' sheets (same group, xy overlap)
    out[tid] = dict(parts=new)
    return out


def billboard_all_baseline(task, priv):
    """Every object replaced by its reference-view billboard (composite as one sheet + 4 slivers)."""
    eye = task['public_cameras']['reference']['eye']
    out = facade_table_baseline(task, priv)
    gobjs, _ = core.build_objects(task, priv['gold'])
    for o in task['objects']:
        if o['id'] == task['composite']:
            continue
        sh = billboard(task, core.aabb(gobjs[o['id']]), eye)
        lo, hi = o['size_range']
        size = [min(max(sh['size'][0], lo[0]), hi[0]), lo[1], min(max(sh['size'][2], lo[2]), hi[2])]
        out[o['id']] = dict(position=[sh['position'][0], round(sh['position'][1] - 0.01 + lo[1] / 2, 4), 0.], yaw=0, size=size)
    return out
