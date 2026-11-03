"""Shared box geometry, spatial relations and scene checks for the spatial suite.

Frame: x = right, y = back (away from the default camera), z = up; metres.
An object is {"id", "size": [sx, sy, sz], "position": [cx, cy, z_bottom], "yaw": 0|90|180|270}.
Yaw is restricted to right angles so every object is an exact axis-aligned box.
"""
import math

TOL = 0.01          # contact / ordering tolerance (m)
ALIGN_TOL = 0.03    # alignment tolerance (m)
BETWEEN_LATERAL = 0.25


def footprint_size(obj):
    sx, sy, _ = obj['size']
    return (sy, sx) if int(obj.get('yaw', 0)) % 180 == 90 else (sx, sy)


def aabb(obj):
    (cx, cy, z0), (fx, fy), sz = obj['position'], footprint_size(obj), obj['size'][2]
    return (cx - fx / 2, cx + fx / 2, cy - fy / 2, cy + fy / 2, z0, z0 + sz)


def center(obj):
    b = aabb(obj)
    return ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)


def gap(a, b):
    """Euclidean separation of two boxes (0 when touching or overlapping)."""
    A, B = aabb(a), aabb(b)
    d = [max(A[2 * i] - B[2 * i + 1], B[2 * i] - A[2 * i + 1], 0.) for i in range(3)]
    return math.sqrt(sum(v * v for v in d))


def penetration(a, b):
    """Smallest overlap depth over the three axes; > 0 only when interiors intersect."""
    A, B = aabb(a), aabb(b)
    return min(min(A[2 * i + 1], B[2 * i + 1]) - max(A[2 * i], B[2 * i]) for i in range(3))


def front_vector(obj):
    t = math.radians(int(obj.get('yaw', 0)))
    return (math.sin(t), -math.cos(t))   # yaw 0 faces -y (toward the default camera)


def _overlap_xy(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) > TOL and min(a[3], b[3]) - max(a[2], b[2]) > TOL


def _inside_xy(p, b, tol=TOL):
    return b[0] - tol <= p[0] <= b[1] + tol and b[2] - tol <= p[1] <= b[3] + tol


# Each relation returns (holds, measured_value_for_feedback).
def rel_left_of(a, b): v = aabb(b)[0] - aabb(a)[1]; return v >= -TOL, v
def rel_right_of(a, b): v = aabb(a)[0] - aabb(b)[1]; return v >= -TOL, v
def rel_in_front_of(a, b): v = aabb(b)[2] - aabb(a)[3]; return v >= -TOL, v
def rel_behind(a, b): v = aabb(a)[2] - aabb(b)[3]; return v >= -TOL, v


def rel_on_top_of(a, b):
    A, B = aabb(a), aabb(b)
    dz = A[4] - B[5]
    return abs(dz) <= TOL and _inside_xy(center(a), B, 0.), dz


def rel_touching(a, b):
    g = gap(a, b)
    return g <= TOL and penetration(a, b) <= TOL, g


def rel_between(c, a, b):
    (ax, ay, _), (bx, by, _), (cx, cy, _) = center(a), center(b), center(c)
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return False, 99.
    t = ((cx - ax) * dx + (cy - ay) * dy) / L2
    lateral = abs((cx - ax) * dy - (cy - ay) * dx) / math.sqrt(L2)
    return 0 < t < 1 and lateral <= BETWEEN_LATERAL, lateral


def rel_distance(a, b, lo, hi):
    (ax, ay, _), (bx, by, _) = center(a), center(b)
    d = math.hypot(ax - bx, ay - by)
    return lo <= d <= hi, d


def rel_aligned_x(a, b): v = abs(center(a)[0] - center(b)[0]); return v <= ALIGN_TOL, v
def rel_aligned_y(a, b): v = abs(center(a)[1] - center(b)[1]); return v <= ALIGN_TOL, v


def rel_facing(a, b):
    (ax, ay, _), (bx, by, _) = center(a), center(b)
    fx, fy = front_vector(a)
    dx, dy = bx - ax, by - ay
    n = math.hypot(dx, dy) or 1e-9
    ang = math.degrees(math.acos(max(-1., min(1., (fx * dx + fy * dy) / n))))
    return ang <= 45., ang


def rel_on_shelf(a, b, level):
    """A rests on shelf board `level` of composite B (board 0 is the bottom board)."""
    boards = b.get('boards') or []
    if not 0 <= int(level) < len(boards):
        return False, None
    return rel_on_top_of(a, boards[int(level)])


def _project(cam, p):
    """Normalised image coordinates (x right, y up, both in [-1, 1] inside the frame) and depth of a world point."""
    from .raycast import _basis
    f, r, u = _basis(cam)
    d = [p[k] - cam['eye'][k] for k in range(3)]
    z = sum(d[k] * f[k] for k in range(3))
    th = math.tan(math.radians(cam['fov_deg']) / 2)
    tw = th * cam['width'] / cam['height']
    return sum(d[k] * r[k] for k in range(3)) / (z * tw), sum(d[k] * u[k] for k in range(3)) / (z * th), z


def projected_area(obj, cam):
    """Area (in normalised image units) of the convex silhouette of the box, ignoring occlusion."""
    b = aabb(obj)
    pts = sorted(_project(cam, (x, y, z))[:2] for x in b[0:2] for y in b[2:4] for z in b[4:6])
    cross = lambda o, a, c: (a[0] - o[0]) * (c[1] - o[1]) - (a[1] - o[1]) * (c[0] - o[0])
    hull = []
    for seq in (pts, pts[::-1]):
        part = []
        for p in seq:
            while len(part) >= 2 and cross(part[-2], part[-1], p) <= 0:
                part.pop()
            part.append(p)
        hull += part[:-1]
    return abs(sum(hull[i][0] * hull[i - 1][1] - hull[i - 1][0] * hull[i][1] for i in range(len(hull)))) / 2


def rel_appears_left_of(a, b, ctx):
    """In the reference image, A's projected centre is left of B's projected centre."""
    v = _project(ctx['camera'], center(b))[0] - _project(ctx['camera'], center(a))[0]
    return v > 0, v


def rel_appears_larger_than(a, b, ctx):
    """In the reference image, A's projected silhouette area (unoccluded) exceeds B's."""
    v = projected_area(a, ctx['camera']) / max(projected_area(b, ctx['camera']), 1e-12)
    return v > 1, v


def rel_x_less(a, b): v = center(b)[0] - center(a)[0]; return v > 0, v
def rel_abs_dx_lt(a, b, d): v = abs(center(a)[0] - center(b)[0]); return v < d, v
def rel_on_floor(a): v = aabb(a)[4]; return abs(v) <= TOL, v
def rel_disjoint(a, b): v = gap(a, b); return v > TOL, v
def rel_color_is(a, c): return a.get('color') == c, a.get('color')
def rel_taller_than(a, b): v = aabb(a)[5] - aabb(a)[4] - (aabb(b)[5] - aabb(b)[4]); return v > TOL, v
def rel_yaw_is(a, y): v = int(a.get('yaw', 0)) % 360; return v == int(y) % 360, v


_OCC_CACHE = {}


def rel_occludes(a, b, ctx):
    """Fraction of B's pixels (reference camera, 80x60) hidden when A is present; holds at >= 0.5."""
    key = (tuple(round(v, 4) for v in aabb(a)), tuple(round(v, 4) for v in aabb(b)), repr(ctx['camera']), tuple(ctx['room']))
    if key not in _OCC_CACHE:
        if len(_OCC_CACHE) > 20000:
            _OCC_CACHE.clear()
        _OCC_CACHE[key] = _occludes(a, b, ctx)
    return _OCC_CACHE[key]


def _occludes(a, b, ctx):
    from . import raycast       # lazy: raycast imports core
    alone = raycast.render({b['id']: b}, ctx['room'], ctx['camera'], 4)
    both = raycast.render({a['id']: a, b['id']: b}, ctx['room'], ctx['camera'], 4)
    n0 = sum(1 for i in alone['ids'] if i >= 0)
    n1 = sum(1 for i in both['ids'] if i >= 0 and both['names'][i] == b['id'])
    v = 1 - n1 / n0 if n0 else 0.
    return n0 > 0 and v >= 0.5, v


RELATIONS = {
    'left_of': (rel_left_of, 2, 'A is entirely left of B: A.max_x <= B.min_x (tolerance 0.01 m).'),
    'right_of': (rel_right_of, 2, 'A is entirely right of B: A.min_x >= B.max_x.'),
    'in_front_of': (rel_in_front_of, 2, 'A is entirely in front of B (closer to -y): A.max_y <= B.min_y.'),
    'behind': (rel_behind, 2, 'A is entirely behind B (toward +y): A.min_y >= B.max_y.'),
    'on_top_of': (rel_on_top_of, 2, "A rests on B: A.min_z == B.max_z (+-0.01 m) and A's footprint centre lies inside B's top face."),
    'touching': (rel_touching, 2, 'A and B touch: their boxes are within 0.01 m and do not interpenetrate.'),
    'between': (rel_between, 3, "A is between B and C on the floor plan: A's centre projects strictly inside segment B-C (centres) and lies within 0.25 m of that segment."),
    'distance': (rel_distance, 2, 'Floor-plan distance between the centres of A and B is within [lo, hi] metres.'),
    'aligned_x': (rel_aligned_x, 2, 'Centres of A and B have the same x (within 0.03 m).'),
    'aligned_y': (rel_aligned_y, 2, 'Centres of A and B have the same y (within 0.03 m).'),
    'x_less': (rel_x_less, 2, 'centre_x(A) < centre_x(B).'),
    'abs_dx_lt': (rel_abs_dx_lt, 2, '|centre_x(A) - centre_x(B)| < d metres (d given as range[0]).'),
    'on_floor': (rel_on_floor, 1, 'A rests directly on the floor: min_z(A) = 0 (within 0.01 m).'),
    'disjoint': (rel_disjoint, 2, 'A and B do not intersect or touch: their boxes are more than 0.01 m apart.'),
    'color_is': (rel_color_is, 1, 'A has the given colour (range[0]).'),
    'taller_than': (rel_taller_than, 2, 'height(A) > height(B) by more than 0.01 m (heights of the chosen sizes).'),
    'yaw_is': (rel_yaw_is, 1, 'yaw(A) equals the given angle (range[0], degrees).'),
    'occludes': (rel_occludes, 2, 'Seen from the reference camera, A hides at least 50% of the pixels B would cover if A were absent.'),
    'on_shelf': (rel_on_shelf, 2, "A rests on shelf board range[0] of bookcase B (board 0 is the bottom board): A.min_z equals that board's top (+-0.01 m) and A's footprint centre lies over the board."),
    'appears_left_of': (rel_appears_left_of, 2, "In the reference camera image, the projection of A's box centre lies left of the projection of B's box centre (perspective, not world x)."),
    'appears_larger_than': (rel_appears_larger_than, 2, "In the reference camera image, the projected silhouette of A's box (ignoring occlusion) has a larger area than B's (perspective, not physical size)."),
    'facing': (rel_facing, 2, "A's front faces B: the angle between A's front direction and the direction from A to B is <= 45 deg. At yaw 0 the front points to -y; yaw rotates counter-clockwise seen from above."),
}
OPPOSITE = {'left_of': 'right_of', 'right_of': 'left_of', 'in_front_of': 'behind', 'behind': 'in_front_of'}
CAMERA_RELATIONS = ('occludes', 'appears_left_of', 'appears_larger_than')


def evaluate(constraint, objs, ctx=None):
    """constraint = {"id", "type", "args": [ids...], optional "range": [lo, hi] or [value]}"""
    fn = RELATIONS[constraint['type']][0]
    try:
        args = [objs[i] for i in constraint['args']]
    except KeyError as e:
        return False, None, f'unknown object {e.args[0]}'
    extra = [ctx] if constraint['type'] in CAMERA_RELATIONS else []
    ok, value = fn(*args, *constraint.get('range', []), *extra)
    return bool(ok), (round(value, 4) if isinstance(value, (int, float)) else value), ''


def describe(c):
    a = c['args']
    t = c['type']
    if t == 'between':
        return f'{a[0]} is between {a[1]} and {a[2]}'
    if t == 'distance':
        return f'distance({a[0]}, {a[1]}) in [{c["range"][0]:.2f}, {c["range"][1]:.2f}] m'
    if t == 'x_less':
        return f'x_{a[0]} < x_{a[1]}'
    if t == 'abs_dx_lt':
        return f'|x_{a[0]} - x_{a[1]}| < {c["range"][0]}'
    if t == 'on_floor':
        return f'z_min({a[0]}) = floor'
    if t in ('color_is', 'yaw_is'):
        return f'{t.split("_")[0]}({a[0]}) = {c["range"][0]}'
    if t == 'disjoint':
        return f'{a[0]} and {a[1]} do not intersect'
    if t == 'on_shelf':
        return f'{a[0]} rests on shelf board {c["range"][0]} of {a[1]}'
    return f'{a[0]} {t.replace("_", " ")} {a[1]}'


# Rigid composites: scored as their parts (local yaw-0 frame, front faces -y). One definition per kind.
SHELF_T, BACK_T = 0.03, 0.02


def composite_parts(kind, size):
    """[(name, offset [dx, dy, bottom_z], size)] of a rigid composite of the given outer size."""
    W, D, H = size
    if kind != 'bookcase':
        raise ValueError(kind)
    inner = [W - 2 * SHELF_T, D - BACK_T, SHELF_T]
    parts = [('side_l', [-W / 2 + SHELF_T / 2, 0, 0], [SHELF_T, D, H]), ('side_r', [W / 2 - SHELF_T / 2, 0, 0], [SHELF_T, D, H]),
             ('back', [0, D / 2 - BACK_T / 2, 0], [W - 2 * SHELF_T, BACK_T, H])]
    return parts + [(f'board{k}', [0, -BACK_T / 2, k * (H - SHELF_T) / 3], inner) for k in range(4)]


def _expand(obj, kind):
    """World-space part boxes of a placed rigid composite (yaw 90/270 swaps footprints)."""
    yaw = int(obj['yaw']) % 360
    c, s = round(math.cos(math.radians(yaw))), round(math.sin(math.radians(yaw)))
    out = {}
    for name, (dx, dy, z0), size in composite_parts(kind, obj['size']):
        size = [size[1], size[0], size[2]] if yaw % 180 else list(size)
        pos = [obj['position'][0] + dx * c - dy * s, obj['position'][1] + dx * s + dy * c, obj['position'][2] + z0]
        pid = f'{obj["id"]}#{name}'
        out[pid] = dict(id=pid, group=obj['id'], rigid=True, size=size, position=pos, yaw=0)
    return out


def solids(objs):
    """The physical boxes: rigid composites contribute their parts, never their outer envelope
    (an envelope passed without its parts, e.g. to a pairwise occlusion test, stands in for them)."""
    expanded = {o['group'] for o in objs.values() if o.get('rigid')}
    return {i: o for i, o in objs.items() if not (o.get('envelope') and i in expanded)}


def global_checks(objs, room, fixed=None):
    """Hard physical checks: inside room, no interpenetration, every object supported, sizes untouched."""
    issues = []
    W, D, H = room
    ids = list(solids(objs))
    for i in objs:
        b = aabb(objs[i])
        if b[0] < -TOL or b[2] < -TOL or b[1] > W + TOL or b[3] > D + TOL or b[4] < -TOL or b[5] > H + TOL:
            issues.append(dict(check='inside_room', objects=[i]))
        if int(objs[i].get('yaw', 0)) not in (0, 90, 180, 270):
            issues.append(dict(check='yaw_right_angle', objects=[i]))
    for i in objs:
        b = aabb(objs[i])
        if objs[i].get('rigid'):
            continue        # parts are fixed to their composite; the composite (envelope) is checked as a whole
        if abs(b[4]) > TOL and not any(j != i and objs[j].get('group') != i and abs(b[4] - aabb(objs[j])[5]) <= TOL and
                                       (_inside_xy(center(objs[i]), aabb(objs[j]), 0.) or
                                        (objs[i].get('group') and objs[i].get('group') == objs[j].get('group') and _overlap_xy(b, aabb(objs[j]))))
                                       for j in ids):
            issues.append(dict(check='supported', objects=[i]))
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            if objs[ids[x]].get('rigid') and objs[ids[x]].get('group') == objs[ids[y]].get('group'):
                continue
            p = penetration(objs[ids[x]], objs[ids[y]])
            if p > TOL:
                issues.append(dict(check='no_interpenetration', objects=[ids[x], ids[y]], depth_m=round(p, 4)))
    for i, ref in (fixed or {}).items():
        if ref.get('size_options') or ref.get('size_range') or ref.get('parts'):
            continue
        o = objs.get(i)
        if o is None or [round(v, 4) for v in o['size']] != [round(v, 4) for v in ref['size']]:
            issues.append(dict(check='size_unchanged', objects=[i]))
    from .layout_logic import issues as semantic_issues
    issues.extend(semantic_issues(objs,room))
    return issues


def check_scene(task, placements):
    """Clean verifier. placements: {id: {"position": [...], "yaw": int}}. Returns a full report."""
    objs, problems = build_objects(task, placements)
    report = dict(valid_output=not problems, output_problems=problems, constraints=[], physical=[])
    if problems:
        return report
    report['physical'] = global_checks(objs, task['room'], {o['id']: o for o in task['objects']})
    ctx = dict(room=task['room'], camera=task.get('camera'))
    for c in task['constraints']:
        ok, value, err = evaluate(c, objs, ctx)
        report['constraints'].append(dict(id=c['id'], type=c['type'], args=c['args'], satisfied=ok, value=value, **({'error': err} if err else {})))
    report['all_satisfied'] = not report['physical'] and all(r['satisfied'] for r in report['constraints'])
    return report


def build_objects(task, placements):
    """Merge catalogue entries with agent placements (the trust boundary).

    Catalogue entry options: fixed "size"; "size_range" [lo, hi] or "size_options" [..] (agent picks "size");
    "colors" (agent picks "color"); "parts": n (agent gives "parts": [{"size", "position"}] boxes, yaw 0).
    """
    problems, objs = [], {}
    if not isinstance(placements, dict):
        return {}, ['placements must be an object keyed by object id']
    for o in task['objects']:
        p = placements.get(o['id'])
        if p is None:
            problems.append(f'missing placement for {o["id"]}')
            continue
        try:
            if o.get('parts'):
                parts = p['parts']
                assert isinstance(parts, list) and len(parts) == o['parts']
                lo, hi = o['part_size_range']
                for k, q in enumerate(parts):
                    size = [float(v) for v in q['size']]
                    pos = [float(v) for v in q['position']]
                    assert len(size) == 3 and len(pos) == 3 and all(math.isfinite(v) for v in size + pos)
                    if any(not (lo[n] - 1e-6 <= size[n] <= hi[n] + 1e-6) for n in range(3)):
                        problems.append(f'part {k} of {o["id"]} outside the allowed part size range')
                    objs[f'{o["id"]}#{k}'] = dict(id=f'{o["id"]}#{k}', group=o['id'], size=size, position=pos, yaw=0)
                continue
            pos = [float(v) for v in p['position']]
            yaw = int(round(float(p.get('yaw', 0)))) % 360
            assert len(pos) == 3 and all(math.isfinite(v) for v in pos)
            size = o.get('size')
            if o.get('size_range') or o.get('size_options'):
                size = [float(v) for v in p['size']]
                assert len(size) == 3
        except Exception:
            problems.append(f'bad placement for {o["id"]}')
            continue
        if o.get('size_range'):
            lo, hi = o['size_range']
            if any(not (lo[k] - 1e-6 <= size[k] <= hi[k] + 1e-6) for k in range(3)):
                problems.append(f'size of {o["id"]} outside allowed range')
        if o.get('size_options') and not any(all(abs(size[k] - opt[k]) < 1e-6 for k in range(3)) for opt in o['size_options']):
            problems.append(f'size of {o["id"]} is not one of its size options')
        obj = dict(id=o['id'], size=size, position=pos, yaw=yaw)
        if o.get('colors'):
            if p.get('color') not in o['colors']:
                problems.append(f'color of {o["id"]} must be one of {o["colors"]}')
            obj['color'] = p.get('color')
        elif o.get('color'):
            obj['color'] = o['color']
        if task.get('scene_policy'):obj['room_policy']=task['scene_policy']
        objs[o['id']] = obj
        if o.get('composite'):
            parts = _expand(obj, o['composite'])
            obj.update(envelope=True, boards=[p for k, p in parts.items() if '#board' in k])
            objs.update(parts)
    extra = set(placements) - {o['id'] for o in task['objects']}
    if extra:
        problems.append(f'unknown object ids: {sorted(extra)}')
    return objs, problems


def edit_cost(before, after):
    """Moved objects and total centre displacement between two placement dicts (same ids)."""
    moved, dist = [], 0.
    for i, p in after.items():
        q = before.get(i)
        if q is None:
            continue
        d = math.dist(p['position'], q['position'])
        if d > 1e-3 or int(p.get('yaw', 0)) % 360 != int(q.get('yaw', 0)) % 360:
            moved.append(i)
            dist += d
    return dict(moved=sorted(moved), n_moved=len(moved), displacement_m=round(dist, 4))


def placements_of(objs, with_size=False):
    out = {}
    for i, o in objs.items():
        if o.get('rigid'):
            continue
        out[i] = dict(position=[round(v, 4) for v in o['position']], yaw=int(o.get('yaw', 0)))
        if with_size:
            out[i]['size'] = list(o['size'])
        if 'color' in o:
            out[i]['color'] = o['color']
    return out
