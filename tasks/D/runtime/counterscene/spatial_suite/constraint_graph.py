"""Experiments A (spatial constraint graph), B (conflicting constraints & repair) and
E (adversarial verifier feedback). All three share one seeded layout generator, so a
B or E instance is a perturbed A instance with a certified feasible gold.
"""
import copy
import itertools
import math
import random

from . import core

CATALOG = [  # id, size [x, y, z] (m), can be stacked on others
    ('crate_red', [0.50, 0.40, 0.40], False), ('cabinet_blue', [0.60, 0.45, 0.90], False),
    ('table_green', [1.00, 0.60, 0.75], False), ('box_yellow', [0.30, 0.30, 0.30], True),
    ('lamp_white', [0.25, 0.25, 1.20], False), ('shelf_orange', [0.80, 0.35, 1.40], False),
    ('stool_purple', [0.35, 0.35, 0.45], False), ('bin_gray', [0.40, 0.40, 0.60], False),
    ('chair_cyan', [0.45, 0.50, 0.85], False), ('plant_pink', [0.35, 0.35, 0.80], False),
    ('case_brown', [0.45, 0.30, 0.25], True), ('tv_black', [0.90, 0.15, 0.55], False),
    # logical v3: workstation, side table, column, shelf items and balls (CC0 meshes or procedural props)
    ('desk_blue', [1.20, 0.60, 0.75], False), ('deskchair_black', [0.55, 0.55, 0.95], False),
    ('monitor_gray', [0.55, 0.18, 0.42], True), ('laptop_gray', [0.34, 0.26, 0.22], True),
    ('sidetable_brown', [0.45, 0.45, 0.55], False), ('column_cyan', [0.30, 0.30, 2.20], False),
    ('flowerpot_green', [0.20, 0.20, 0.30], True), ('soccer_white', [0.22, 0.22, 0.22], True),
    ('rugby_brown', [0.28, 0.18, 0.18], True),
]
TIERS = {'L1': 6, 'L2': 14, 'L3': 34}          # constraint count per tier
MAX_OBJECTS = 18
# L3 rooms always contain a bookcase with shelf items and a desk workstation (Mike, v3)
RICH_ROOM = dict(must=('shelf_orange', 'desk_blue', 'deskchair_black', 'monitor_gray', 'soccer_white', 'flowerpot_green', 'column_cyan'),
                 wants={'soccer_white': 'shelf', 'flowerpot_green': 'shelf'})
SWEEP = [3, 6, 10, 14, 20, 26, 34]                  # constraint counts for the scaling curve
GRID = 0.05


def _snap(v):
    return round(round(v / GRID) * GRID, 4)


def _layout(rng, n, use_yaw, stack, jitter=0., must=(), wants=None, catalogue=None):
    from .layout_logic import layout
    return layout(rng,n,use_yaw,stack,jitter,CATALOG if catalogue is None else catalogue,must,wants,uniform_jitter=catalogue is not None)


def placed_ids(objs):
    """Placed objects only (rigid composite parts are geometry, not placements)."""
    return [i for i, o in objs.items() if not o.get('rigid')]


def task_objects(objs):
    from .layout_logic import COMPOSITES
    return [dict(id=i, size=objs[i]['size'], **({'composite': COMPOSITES[i]} if i in COMPOSITES else {})) for i in placed_ids(objs)]


def _true_relations(objs, types, camera=None):
    ids = placed_ids(objs)
    out = []
    for a, b in itertools.permutations(ids, 2):
        for t in types:
            if t in ('between', 'on_shelf', 'occludes', 'appears_left_of', 'appears_larger_than'):
                continue
            if t == 'distance':
                d = core.rel_distance(objs[a], objs[b], 0, 99)[1]
                if a < b:
                    out.append(dict(type=t, args=[a, b], range=[round(max(0, d - 0.2), 2), round(d + 0.2, 2)]))
                continue
            if t in ('aligned_x', 'aligned_y', 'touching') and a > b:
                continue
            if core.RELATIONS[t][0](objs[a], objs[b])[0]:
                out.append(dict(type=t, args=[a, b]))
    if 'on_shelf' in types:
        for i in ids:
            for s in ids:
                for k, board in enumerate(objs[s].get('boards') or []):
                    if core.rel_on_top_of(objs[i], board)[0]:
                        out.append(dict(type='on_shelf', args=[i, s], range=[k]))
    if camera and ('appears_left_of' in types or 'appears_larger_than' in types):
        # perspective relations only where the image reading contradicts the naive 3D reading
        ctx = dict(camera=camera)
        vol = lambda o: o['size'][0] * o['size'][1] * o['size'][2]
        for a, b in itertools.permutations(ids, 2):
            if 'appears_left_of' in types and core.rel_appears_left_of(objs[a], objs[b], ctx)[0] and core.center(objs[a])[0] > core.center(objs[b])[0] + 0.05:
                out.append(dict(type='appears_left_of', args=[a, b]))
            if 'appears_larger_than' in types and vol(objs[a]) < vol(objs[b]) and core.rel_appears_larger_than(objs[a], objs[b], ctx)[1] > 1.15:
                out.append(dict(type='appears_larger_than', args=[a, b]))
    if camera and 'occludes' in types:
        # ray-cast only nearer/farther pairs whose projected centres are close (cheap prefilter)
        ctx = dict(camera=camera, room=[0, 0, 0])    # room only tints the floor; occlusion counts ignore it
        for a, b in itertools.permutations(ids, 2):
            pa, pb = core._project(camera, core.center(objs[a])), core._project(camera, core.center(objs[b]))
            if pa[2] < pb[2] and abs(pa[0] - pb[0]) < 0.25 and abs(pa[1] - pb[1]) < 0.35 and core.rel_occludes(objs[a], objs[b], ctx)[0]:
                out.append(dict(type='occludes', args=[a, b]))
    if 'between' in types:
        for c, a, b in itertools.permutations(ids, 3):
            if a < b and core.rel_between(objs[c], objs[a], objs[b])[0] and core.gap(objs[a], objs[b]) > 0.6:
                out.append(dict(type='between', args=[c, a, b]))
    return out


TYPE_WEIGHT = {'touching': 3, 'on_top_of': 4, 'between': 3, 'aligned_x': 2, 'aligned_y': 2, 'distance': 2,
               'facing': 2, 'left_of': 1, 'right_of': 1, 'in_front_of': 1, 'behind': 1,
               'on_shelf': 4, 'occludes': 3, 'appears_left_of': 3, 'appears_larger_than': 3}


def _select(rng, objs, cands, k):
    """Pick k non-redundant true relations that connect every object, favouring tight relations."""
    chosen, used_pairs = [], set()

    def key(c):
        return frozenset(c['args'][:2]) if c['type'] != 'between' else tuple(c['args'])

    def take(c):
        chosen.append(c)
        used_pairs.add(key(c))

    pool = cands[:]
    rng.shuffle(pool)
    # 0) one x-directional and one y-directional relation, so mirrored layouts never pass
    for axis in (('left_of', 'right_of'), ('in_front_of', 'behind')):
        pick = [c for c in pool if c['type'] in axis]
        if pick:
            take(pick[0])
    # 1) spanning connection over objects
    seen = set(a for c in chosen for a in c['args']) or {rng.choice(placed_ids(objs))}
    while len(seen) < len(placed_ids(objs)) and len(chosen) < k:
        bridge = [c for c in pool if key(c) not in used_pairs and len(set(c['args']) & seen) >= 1 and set(c['args']) - seen]
        if not bridge:
            break
        c = max(bridge, key=lambda c: TYPE_WEIGHT[c['type']] + rng.random() * 2)
        take(c)
        seen |= set(c['args'])
    # 2) fill with diverse types
    while len(chosen) < k:
        rest = [c for c in pool if key(c) not in used_pairs]
        if not rest:
            break
        counts = {}
        for c in chosen:
            counts[c['type']] = counts.get(c['type'], 0) + 1
        take(max(rest, key=lambda c: TYPE_WEIGHT[c['type']] / (1 + counts.get(c['type'], 0)) + rng.random() * 1.5))
    for n, c in enumerate(chosen):
        c['id'] = f'c{n + 1:02d}'
    return chosen


def make_a(seed, k, catalogue=None):
    """Experiment A instance with k constraints. Returns (task, gold_placements)."""
    for attempt in range(300):
        rng = random.Random(f'A-{seed}-{k}-{attempt}')
        n = min(MAX_OBJECTS, max(3, round(k * 0.55) + 2))
        types = ['left_of', 'right_of', 'in_front_of', 'behind', 'touching', 'distance']
        if k >= 6:
            types += ['on_top_of', 'aligned_x', 'aligned_y', 'between']
        if k >= 12:
            types += ['facing']
        rich = k >= 26
        if rich:
            types += ['on_shelf', 'occludes', 'appears_left_of', 'appears_larger_than']
        try:
            room, objs = _layout(rng, n, use_yaw=k >= 12, stack=0 if k < 6 else 1 + (k >= 14) + 2 * rich,
                                 catalogue=catalogue, **(RICH_ROOM if rich else {}))
        except RuntimeError:
            continue
        camera = None
        if rich:
            from .grounding import cameras          # lazy: grounding imports this module
            camera = cameras(room, ['reference'])['reference']
        cons = _select(rng, objs, _true_relations(objs, types, camera), k)
        if len(cons) < k or len({a for c in cons for a in c['args']}) < len(placed_ids(objs)):
            continue
        # every instance must break mirror symmetry in x and y, or frame errors would go unscored
        mirrored = {}
        for i, o in objs.items():
            mirrored[i] = dict(o, position=[room[0] - o['position'][0], room[1] - o['position'][1], o['position'][2]])
        mx = {i: dict(o, position=[room[0] - o['position'][0], o['position'][1], o['position'][2]]) for i, o in objs.items()}
        my = {i: dict(o, position=[o['position'][0], room[1] - o['position'][1], o['position'][2]]) for i, o in objs.items()}
        if any(all(core.evaluate(c, m, dict(room=room, camera=camera))[0] for c in cons) for m in (mx, my, mirrored)):
            continue
        task = dict(experiment='A', instance=f'A-k{k:02d}-s{seed}', k=k, room=room,
                    objects=task_objects(objs), **({'camera': camera} if camera else {}),
                    scene_policy=__import__('counterscene.spatial_suite.layout_logic',fromlist=['policy']).policy(),
                    constraints=cons, relation_types=sorted({c['type'] for c in cons}), uses_yaw=k >= 12)
        gold = core.placements_of(objs)
        assert core.check_scene(task, gold)['all_satisfied']
        return task, gold
    raise RuntimeError('could not generate instance')


# ---------------------------------------------------------------- Experiment B
B_TIERS = {'L1': dict(k=6, corrupt=1, infeasible=0, locked=0.2),
           'L2': dict(k=10, corrupt=2, infeasible=1, locked=0.3),
           'L3': dict(k=26, corrupt=3, infeasible=2, locked=0.3)}


def _corrupt(rng, task, gold, m, movable):
    """Move m movable floor objects so some constraints break but most still hold."""
    objs, _ = core.build_objects(task, gold)
    supporters = {objs[j].get('group') or j for i in objs for j in objs if i != j and core.rel_on_top_of(objs[i], objs[j])[0]}
    cand = [i for i in movable if objs[i]['position'][2] == 0 and i not in supporters]
    W, D, _ = task['room']
    for _ in range(400):
        moved = rng.sample(cand, min(m, len(cand)))
        broken = copy.deepcopy(gold)
        for i in moved:
            ang = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.4, 1.1)
            x, y, z = broken[i]['position']
            broken[i]['position'] = [_snap(min(max(x + r * math.cos(ang), 0.2), W - 0.2)), _snap(min(max(y + r * math.sin(ang), 0.2), D - 0.2)), z]
        rep = core.check_scene(task, broken)
        sat = [c['satisfied'] for c in rep['constraints']]
        if not all(sat) and sum(sat) >= 0.5 * len(sat) and all(any(i in c['args'] and not c['satisfied'] for c in rep['constraints']) for i in moved):
            return broken, moved
    raise RuntimeError('corruption failed')


def _infeasible_soft(rng, task, gold, n):
    """Soft constraints that provably contradict hard ones (certificate stored privately)."""
    objs, _ = core.build_objects(task, gold)
    hard = task['constraints']
    out, extra_hard = [], []
    dirs = [c for c in hard if c['type'] in core.OPPOSITE]
    kinds = ['transitive_chain', 'between_order', 'metric_contact']
    rng.shuffle(kinds)
    for kind in kinds:
        if len(out) >= n:
            break
        if kind == 'between_order':
            # hard: c between a and b, and a left_of b  =>  c's centre x < b's centre x, so "c right_of b" is impossible
            for t, ax in (('left_of', 0), ('in_front_of', 1)):
                trip = [(c, a, b) for c, a, b in itertools.permutations(objs, 3)
                        if core.rel_between(objs[c], objs[a], objs[b])[0] and core.RELATIONS[t][0](objs[a], objs[b])[0]
                        # infeasible under any yaw: smallest half-widths must exceed the 0.25 m lateral allowance
                        and (min(objs[c]['size'][:2]) + min(objs[b]['size'][:2])) / 2 > core.BETWEEN_LATERAL + 0.02]
                if trip:
                    c, a, b = rng.choice(trip)
                    for h in (dict(type='between', args=[c, a, b]), dict(type=t, args=[a, b])):
                        if not any(x['type'] == h['type'] and x['args'] == h['args'] for x in hard + extra_hard):
                            extra_hard.append(h)
                    out.append(dict(type=core.OPPOSITE[t], args=[c, b], certificate=f'hard between({c},{a},{b}) and {t}({a},{b}) bound the centre overshoot by {core.BETWEEN_LATERAL} m; opposite full-box separation needs more than this even with {core.TOL} m tolerance'))
                    break
        elif kind == 'transitive_chain':
            for t in ('left_of', 'in_front_of', 'right_of', 'behind'):
                trip = [(a, b, c) for a, b, c in itertools.permutations(objs, 3) if core.RELATIONS[t][0](objs[a], objs[b])[0] and core.RELATIONS[t][0](objs[b], objs[c])[0]]
                if trip:
                    a, b, c = rng.choice(trip)
                    for x, y in ((a, b), (b, c)):
                        if not any(h['type'] == t and h['args'] == [x, y] for h in hard + extra_hard):
                            extra_hard.append(dict(type=t, args=[x, y]))
                    out.append(dict(type=core.OPPOSITE[t], args=[a, c], certificate=f'hard {t}({a},{b}) and {t}({b},{c}) imply {t}({a},{c})'))
                    break
        elif kind == 'metric_contact':
            touch = [c for c in hard if c['type'] == 'touching']
            if touch:
                c = rng.choice(touch)
                a, b = (objs[i] for i in c['args'])
                ma, mb = max(a['size'][:2]), max(b['size'][:2])      # any yaw the agent might choose
                dmax = math.hypot(ma, mb)
                out.append(dict(type='distance', args=c['args'][:], range=[round(dmax + 0.3, 2), round(dmax + 0.9, 2)],
                                certificate=f'hard constraint "{core.describe(c)}" bounds horizontal centre distance by {math.sqrt(2)*((ma+mb)/2+core.TOL):.4f} m including contact tolerance'))
    return out[:n], extra_hard


def make_b(seed, tier, catalogue=None):
    cfg = B_TIERS[tier]
    for attempt in range(30):
        rng = random.Random(f'B-{seed}-{tier}-{attempt}')
        try:
            task, gold = make_a(seed * 7 + attempt + 1000, cfg['k'], catalogue=catalogue)
            ids = [o['id'] for o in task['objects']]
            infeasible, extra_hard = _infeasible_soft(rng, task, gold, cfg['infeasible'])
            if len(infeasible) < cfg['infeasible']:
                continue
            cons = [dict(c, priority='hard') for c in task['constraints']]
            for c in extra_hard:
                cons.append(dict(c, priority='hard'))
            for n, c in enumerate(cons):
                c['id'] = f'h{n:02d}'
            locked = sorted(rng.sample(ids, round(cfg['locked'] * len(ids))))
            movable = [i for i in ids if i not in locked]
            task = dict(task, constraints=cons)
            broken, moved = _corrupt(rng, task, gold, cfg['corrupt'], movable)
        except RuntimeError:
            continue
        # a feasible soft preference that is true in gold but violated in the broken scene
        gobjs, _ = core.build_objects(task, gold)
        bobjs, _ = core.build_objects(task, broken)
        soft_ok = [c for c in _true_relations(gobjs, ['left_of', 'right_of', 'in_front_of', 'behind', 'distance'])
                   if any(m in c['args'] for m in moved) and not core.evaluate(c, bobjs)[0]
                   and not any(h['args'] == c['args'] and h['type'] == c['type'] for h in cons)]
        if not soft_ok:
            continue
        softs = [dict(rng.choice(soft_ok), priority='soft', feasible=True)]
        softs += [dict(c, priority='soft', feasible=False) for c in infeasible]
        rng.shuffle(softs)
        allc = cons + softs
        for n, c in enumerate(allc):
            c['id'] = f'c{n + 1:02d}'
        private = {c['id']: dict(feasible=c.pop('feasible'), certificate=c.pop('certificate', '')) for c in allc if 'feasible' in c}
        task = dict(task, experiment='B', instance=f'B-{tier}-s{seed}', tier=tier, constraints=allc, locked=locked,
                    initial=broken, relation_types=sorted({c['type'] for c in allc}))
        task['initial_report'] = core.check_scene(task, broken)
        hard_task = dict(task, constraints=[c for c in allc if c['priority'] == 'hard' or private.get(c['id'], {}).get('feasible')])
        if not core.check_scene(hard_task, gold)['all_satisfied']:
            continue
        priv = dict(gold=gold, corrupted=moved, soft=private, gold_edit=core.edit_cost(broken, gold))
        return task, priv
    raise RuntimeError('B generation failed')


def score_b(task, priv, placements, infeasible_ids, initial=None, report=None):
    rep = core.check_scene(task, placements) if report is None else report
    if not rep['valid_output']:
        return dict(success=False, reason='invalid_output', report=rep)
    by = {r['id']: r for r in rep['constraints']}
    hard = [c['id'] for c in task['constraints'] if c['priority'] == 'hard']
    soft_feasible = [i for i, v in priv['soft'].items() if v['feasible']]
    infeasible = {i for i, v in priv['soft'].items() if not v['feasible']}
    locked_moved = [i for i in task['locked'] if i in core.edit_cost(task['initial'], placements)['moved']]
    init_ok = [r['id'] for r in task['initial_report']['constraints'] if r['satisfied'] and r['id'] in hard]
    claimed = set(infeasible_ids or [])
    edit = core.edit_cost(task['initial'], placements)
    hard_ok = all(by[i]['satisfied'] for i in hard) and not rep['physical'] and not locked_moved
    return dict(
        success=hard_ok and all(by[i]['satisfied'] for i in soft_feasible) and claimed == infeasible,
        hard_ok=hard_ok, physical=rep['physical'], locked_moved=locked_moved,
        hard_satisfied=sum(by[i]['satisfied'] for i in hard), hard_total=len(hard),
        preserved=sum(by[i]['satisfied'] for i in init_ok), preserved_total=len(init_ok),
        soft_feasible_ok=sum(by[i]['satisfied'] for i in soft_feasible), soft_feasible_total=len(soft_feasible),
        conflict_tp=len(claimed & infeasible), conflict_fp=len(claimed - infeasible), conflict_fn=len(infeasible - claimed),
        edit=edit, gold_edit=priv['gold_edit'],
        collateral_moves=sorted(set(edit['moved']) - set(priv['corrupted'])))


# ---------------------------------------------------------------- Experiment E
# Severity scales with level: planted interpenetration conflicts (count, depth, structure) and checker faults.
E_TIERS = {'L1': dict(k=6, plan=[('pair', 'shallow')], faults=['false_fail']),
           'L2': dict(k=10, plan=[('pair', 'shallow'), ('pair', 'medium')], faults=['hidden_pair', 'noisy_value', 'false_pass_or_missing']),
           'L3': dict(k=26, plan=[('pair', 'deep'), ('pair', 'stacked'), ('cluster', 3)],
                      faults=['type_bug', 'hidden_pair', 'hidden_pair', 'false_fail', 'omission', 'noisy_value'])}
DEPTH = {'shallow': (0.03, 0.05), 'medium': (0.07, 0.13), 'deep': (0.16, 0.22), 'stacked': (0.04, 0.08)}


def pair_key(objects):
    return 'no_interpenetration:' + '+'.join(sorted(objects))


def _pairs(task, placements):
    rep = core.check_scene(task, placements)
    return {tuple(sorted(p['objects'])): p['depth_m'] for p in rep['physical'] if p['check'] == 'no_interpenetration'}, rep


def _push(objs, cur, a, b, depth, side):
    """Move floor object a along x so it overlaps b by `depth` on b's `side` (keeps y when the footprints already overlap in y)."""
    A, B = core.aabb(objs[a]), core.aabb(objs[b])
    fx = A[1] - A[0]
    x = B[1] + fx / 2 - depth if side > 0 else B[0] - fx / 2 + depth
    y = cur[a]['position'][1]
    if min(A[3], B[3]) - max(A[2], B[2]) < depth + 0.03:
        y = core.center(objs[b])[1]
    cur[a]['position'] = [round(x, 4), round(y, 4), cur[a]['position'][2]]


def _plant_conflicts(rng, task, gold, plan):
    """Gold + planted interpenetrations. Pairs move one object into another by a planned depth; a cluster stacks
    2-3 movable objects on a corner of an anchor so every member pair interpenetrates (no single move fixes it).
    Raises RuntimeError unless the realised overlaps are exactly the planned ones and at least half of the objects
    (and 6 at L3-sized rooms) stay clean, i.e. untouched and in no conflict."""
    from .layout_logic import SEATS, COMPOSITES, HOSTS
    objs, _ = core.build_objects(task, gold)
    ids = placed_ids(objs)
    support = {objs[j].get('group') or j for i in ids for j in objs if i != j and core.rel_on_top_of(objs[i], objs[j])[0]}
    on_floor = [i for i in ids if objs[i]['position'][2] == 0 and i not in COMPOSITES]
    movable = [i for i in on_floor if i not in support and i not in SEATS and i not in task['scene_policy']['wall_storage']]
    cur, conflicts, used = copy.deepcopy(gold), [], set()
    for n, (kind, arg) in enumerate(plan):
        o2, _ = core.build_objects(task, cur)
        if kind == 'pair' and arg == 'stacked':
            tops = [h for h in HOSTS if h in objs]
            a = rng.choice([i for i in ids if i not in used and any(core.rel_on_top_of(objs[i], objs[h])[0] for h in tops)])
            host = next(h for h in tops if core.rel_on_top_of(objs[a], objs[h])[0])
            d = round(rng.uniform(*DEPTH[arg]), 3)
            cur[a]['position'][2] = round(cur[a]['position'][2] - d, 4)        # sunk into its host's top
            conflicts.append(dict(kind='pair', subtype='stacked', members=sorted([a, host]), moved=[a], planned_depth_m=d,
                                  also_breaks=['supported', 'approved_full_footprint_support']))
            used |= {a, host}
        elif kind == 'pair':
            a = rng.choice([i for i in movable if i not in used])
            b = rng.choice([i for i in on_floor if i != a and i not in used])
            d = round(rng.uniform(*DEPTH[arg]), 3)
            _push(o2, cur, a, b, d, rng.choice([-1, 1]))
            conflicts.append(dict(kind='pair', subtype=arg, members=sorted([a, b]), moved=[a], planned_depth_m=d))
            used |= {a, b}
        else:
            anchor = rng.choice([i for i in on_floor if i not in used])
            members = rng.sample([i for i in movable if i not in used and i != anchor], arg - 1)
            B = core.aabb(o2[anchor])
            corner = (B[rng.choice([0, 1])], B[rng.choice([2, 3])])
            for m in members:
                cur[m]['position'] = [round(corner[0] + rng.uniform(-.04, .04), 4), round(corner[1] + rng.uniform(-.04, .04), 4), 0.]
            conflicts.append(dict(kind='cluster', subtype=f'{arg}-object', members=sorted([anchor] + members), moved=sorted(members)))
            used |= {anchor, *members}
    pairs, rep = _pairs(task, cur)
    planned = set()
    for c in conflicts:
        c['id'] = f'k{len([x for x in conflicts if conflicts.index(x) <= conflicts.index(c)])}'
        mine = [p for p in pairs if set(p) <= set(c['members'])]
        if c['kind'] == 'pair' and (len(mine) != 1 or abs(pairs[mine[0]] - c.pop('planned_depth_m')) > 1e-3):
            raise RuntimeError('planned pair depth not realised')
        if c['kind'] == 'cluster' and len(mine) < len(c['members']) * (len(c['members']) - 1) // 2:
            raise RuntimeError('cluster members do not all interpenetrate')
        c['pairs'] = [dict(objects=list(p), depth_m=pairs[p]) for p in sorted(mine)]
        planned |= set(mine)
    if set(pairs) != planned:
        raise RuntimeError('unplanned interpenetration')
    allowed = {(k, m) for c in conflicts for m in c['moved'] for k in c.get('also_breaks', ())}
    if any(p['check'] != 'no_interpenetration' and (p['check'], p['objects'][0]) not in allowed for p in rep['physical']):
        raise RuntimeError('side effect outside the planted conflicts')
    clean = [i for i in ids if i not in used]
    if len(clean) < len(ids) / 2 or (len(ids) >= 14 and len(clean) < 6):
        raise RuntimeError('too few clean objects')
    return cur, conflicts, sorted({m for c in conflicts for m in c['moved']}), clean


def make_e(seed, tier, catalogue=None):
    cfg = E_TIERS[tier]
    for attempt in range(80):
        rng = random.Random(f'E-{seed}-{tier}-{attempt}')
        plan = [(k, rng.choice([3, 4]) if k == 'cluster' else a) for k, a in cfg['plan']]
        try:
            task, gold = make_a(seed * 11 + attempt + 2000, cfg['k'], catalogue=catalogue)
            broken, conflicts, moved, clean = _plant_conflicts(rng, task, gold, plan)
        except (RuntimeError, ValueError, IndexError, StopIteration):
            continue
        rep0 = core.check_scene(task, broken)
        violated = [r['id'] for r in rep0['constraints'] if not r['satisfied']]
        satisfied = [r['id'] for r in rep0['constraints'] if r['satisfied']]
        argsof = {c['id']: set(c['args']) for c in task['constraints']}
        faults = {}
        hideable = [p['objects'] for c in conflicts if c['subtype'] != 'deep' for p in c['pairs']]
        if cfg['faults'].count('hidden_pair') > len(hideable):
            continue
        faults['hidden_pairs'] = rng.sample(hideable, cfg['faults'].count('hidden_pair'))
        if 'false_pass_or_missing' in cfg['faults']:
            lone = [v for v in violated if not any(argsof[v] & argsof[w] for w in violated if w != v)]
            if lone:
                faults['false_pass'] = [rng.choice(lone)]
            else:
                faults['omission'] = [rng.choice(satisfied)]
        if 'type_bug' in cfg['faults']:
            dirs = [c for c in task['constraints'] if c['type'] in core.OPPOSITE and c['id'] in violated]
            if not dirs:
                continue
            faults['type_bug'] = dirs[0]['type']      # this relation type is evaluated on centres only
        if 'false_fail' in cfg['faults']:
            stable = [i for i in satisfied if not argsof[i] & set(moved)]
            if not stable:
                continue
            faults['false_fail'] = [rng.choice(stable)]
        if 'omission' in cfg['faults']:
            pool = [i for i in violated if i not in faults.get('false_pass', [])]
            faults['omission'] = sorted(rng.sample(pool, 1)) if pool else []
        if 'noisy_value' in cfg['faults']:
            faults['noisy_value'] = 0.15
        task = dict(task, experiment='E', instance=f'E-{tier}-s{seed}', tier=tier, initial=broken)
        if 'type_bug' in faults:
            truth = {r['id']: r for r in rep0['constraints']}
            seen = {r['id']: r['satisfied'] for r in noisy_check(task, faults, broken)['constraints']}
            if not any(r['type'] == faults['type_bug'] and not r['satisfied'] and seen.get(i) for i, r in truth.items()):
                continue    # the type bug must actually hide a real violation in the start scene
        hidden = {pair_key(p) for p in faults['hidden_pairs']}
        for c in conflicts:
            for p in c['pairs']:
                p['hidden_by_checker'] = pair_key(p['objects']) in hidden
        depths = [p['depth_m'] for c in conflicts for p in c['pairs']]
        priv = dict(gold=gold, corrupted=moved, faults=faults, planted=planted_issues(task, faults), conflicts=conflicts,
                    conflict_count=len(conflicts), overlap_pair_count=len(depths), max_conflict_depth_m=max(depths), clean_objects=clean)
        return task, priv
    raise RuntimeError('E generation failed')


def noisy_check(task, faults, placements, nonce=0, report=None):
    """The deliberately unreliable checker the agent sees in experiment E."""
    rep = core.check_scene(task, placements) if report is None else report
    if not rep['valid_output']:
        return rep
    rng = random.Random(f'noise-{task["instance"]}-{nonce}')
    objs, _ = core.build_objects(task, placements)
    out = []
    for r in rep['constraints']:
        r = dict(r)
        if r['id'] in faults.get('omission', []):
            continue
        if r['id'] in faults.get('false_pass', []):
            r['satisfied'] = True
        if r['id'] in faults.get('false_fail', []):
            r['satisfied'] = False
        if faults.get('type_bug') == r['type']:
            a, b = (objs[i] for i in r['args'])
            ax, bx = core.center(a), core.center(b)
            axis, sign = {'left_of': (0, 1), 'right_of': (0, -1), 'in_front_of': (1, 1), 'behind': (1, -1)}[r['type']]
            r['satisfied'] = sign * (bx[axis] - ax[axis]) > 0
        if faults.get('noisy_value') and isinstance(r['value'], (int, float)):
            r['value'] = round(r['value'] + rng.uniform(-1, 1) * faults['noisy_value'], 3)
        out.append(r)
    hidden = {pair_key(p) for p in faults.get('hidden_pairs', [])}
    phys = [p for p in rep['physical'] if p['check'] not in faults.get('hidden_physics', [])
            and not (p['check'] == 'no_interpenetration' and pair_key(p['objects']) in hidden)]
    return dict(valid_output=True, output_problems=[], constraints=out, physical=phys,
                all_satisfied=not phys and all(r['satisfied'] for r in out))


def planted_issues(task, faults):
    """Constraint ids (or physical check keys) whose checker output is unreliable, with issue type."""
    out = {}
    for i in faults.get('false_pass', []):
        out[i] = 'false_pass'
    for i in faults.get('false_fail', []):
        out[i] = 'false_fail'
    for i in faults.get('omission', []):
        out[i] = 'missing'
    if faults.get('type_bug'):
        for c in task['constraints']:
            if c['type'] == faults['type_bug']:
                out.setdefault(c['id'], 'wrong_rule')
    for p in faults.get('hidden_physics', []):
        out[p] = 'missing'
    for p in faults.get('hidden_pairs', []):
        out[pair_key(p)] = 'missing'
    return out


def claim_key(x):
    """Normalise a reported checker issue: physical pairs may be given as id 'no_interpenetration' + objects."""
    if not isinstance(x, dict):
        return str(x)
    i = str(x.get('id'))
    if i.startswith('no_interpenetration') and isinstance(x.get('objects'), list) and len(x['objects']) == 2:
        return pair_key([str(o) for o in x['objects']])
    if i.startswith('no_interpenetration:'):
        return pair_key(i.split(':', 1)[1].replace(',', '+').split('+'))
    return i


def score_e(task, priv, placements, issues, report=None):
    rep = core.check_scene(task, placements) if report is None else report
    if not rep['valid_output']:
        return dict(success=False, reason='invalid_output', report=rep)
    noisy = noisy_check(task, priv['faults'], placements,report=rep)
    noisy_by = {r['id']: r['satisfied'] for r in noisy['constraints']}
    overtrust = [r['id'] for r in rep['constraints'] if not r['satisfied'] and noisy_by.get(r['id'], True)]
    shown = {pair_key(p['objects']) for p in noisy['physical'] if p['check'] == 'no_interpenetration'}
    overtrust += [pair_key(p['objects']) for p in rep['physical'] if p['check'] == 'no_interpenetration' and pair_key(p['objects']) not in shown]
    ok_type = {'false_pass': {'false_pass', 'wrong_rule'}, 'wrong_rule': {'wrong_rule', 'false_pass', 'false_fail'},
               'false_fail': {'false_fail'}, 'missing': {'missing'}}
    claimed = set()
    for x in issues or []:
        i, t = claim_key(x), (x.get('issue') if isinstance(x, dict) else None)
        if i in priv['planted'] and t is not None and t not in ok_type[priv['planted'][i]]:
            i = i + ':wrong_type'
        claimed.add(i)
    planted = set(priv['planted'])
    left = {pair_key(p['objects']) for p in rep['physical'] if p['check'] == 'no_interpenetration'}
    conflicts = priv.get('conflicts', [])
    fixed = lambda c: not any(pair_key(p['objects']) in left for p in c['pairs'])
    resolved = {kind: [sum(fixed(c) for c in conflicts if c['kind'] == kind), sum(c['kind'] == kind for c in conflicts)] for kind in ('pair', 'cluster')}
    edit = core.edit_cost(task['initial'], placements)
    return dict(success=bool(rep['all_satisfied']), true_satisfied=sum(r['satisfied'] for r in rep['constraints']),
                total=len(rep['constraints']), physical=rep['physical'], overtrust=overtrust,
                conflicts_resolved=dict(pairs=resolved['pair'], clusters=resolved['cluster']), conflict_count=priv.get('conflict_count'),
                clean_disturbed=sorted(set(edit['moved']) & set(priv.get('clean_objects', []))),
                detect_tp=len(claimed & planted), detect_fp=len(claimed - planted), detect_fn=len(planted - claimed),
                edit=edit)


def score_a(task, placements, report=None):
    rep = core.check_scene(task, placements) if report is None else report
    if not rep['valid_output']:
        return dict(success=False, reason='invalid_output', report=rep, satisfied=0, total=len(task['constraints']))
    viol = [r for r in rep['constraints'] if not r['satisfied']]
    objs, _ = core.build_objects(task, placements)
    mirror = [r['id'] for r in viol if r['type'] in core.OPPOSITE and
              core.RELATIONS[core.OPPOSITE[r['type']]][0](*[objs[a] for a in r['args']])[0]]
    return dict(success=bool(rep['all_satisfied']), satisfied=len(rep['constraints']) - len(viol), total=len(rep['constraints']),
                physical=rep['physical'], violated_types=[r['type'] for r in viol], mirror_errors=mirror)
