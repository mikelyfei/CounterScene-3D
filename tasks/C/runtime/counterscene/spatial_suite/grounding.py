"""Experiment C: image-to-3D grounding with held-out verification views.

The agent sees one calibrated reference render (plus a small budget of extra public views),
rebuilds the box scene, and is scored only from hidden cameras it never sees.
"""
import math
import random

from . import core, raycast
from .constraint_graph import _layout

# v3 retier: L1 = former L2, L2 = former L3, L3 = rich room (bookcase items, workstation, balls, column)
C_TIERS = {'L1': dict(n=4, jitter=0.2, yaw=True, stack=0, queries=1),
           'L2': dict(n=6, jitter=0.2, yaw=True, stack=1, queries=2),
           'L3': dict(n=10, jitter=0.2, yaw=True, stack=3, queries=3,
                      must=('shelf_orange', 'desk_blue', 'deskchair_black', 'monitor_gray', 'soccer_white', 'rugby_brown',
                            'flowerpot_green', 'column_cyan'),
                      wants={'soccer_white': 'floor', 'rugby_brown': 'shelf', 'flowerpot_green': 'shelf'})}
TALL_M = 1.2              # objects at least this tall may not hide more than MAX_HIDE of the reference furniture pixels
MAX_HIDE = 0.6
MIN_VISIBLE = 0.3         # every object shows at least this fraction of itself in some public view
VIS_SCALE = 4             # 80x60 visibility renders
PUBLIC_VIEWS = {'reference': (-90, 32), 'front_left': (-140, 30), 'front_right': (-40, 30), 'left': (180, 25), 'right': (0, 25), 'top': (-90, 78)}
HIDDEN_VIEWS = {'back': (90, 30), 'back_left': (135, 22), 'back_right': (45, 40), 'low_front_right': (-62, 12)}
PASS_IOU = 0.80          # hidden-view mean per-object mask IoU required to pass (calibrated in iteration log)
SCALE_EVAL = 2           # hidden views scored at 160x120


def cameras(room, names):
    W, D, _ = room
    c = (W / 2, D / 2, 0.35)
    R = 0.78 * math.hypot(W, D) + 0.7
    out = {}
    for n in names:
        az, el = {**PUBLIC_VIEWS, **HIDDEN_VIEWS}[n]
        a, e = math.radians(az), math.radians(el)
        eye = (c[0] + R * math.cos(e) * math.cos(a), c[1] + R * math.cos(e) * math.sin(a), c[2] + R * math.sin(e))
        out[n] = raycast.camera(eye, c, fov_deg=50)
    return out


def _pixels(objs, room, cam):
    img = raycast.render(objs, room, cam, VIS_SCALE)
    out = {}
    for i in img['ids']:
        if i >= 0:
            n = raycast.base_name(img['names'][i])
            out[n] = out.get(n, 0) + 1
    return out


def visibility(objs, room, cams):
    """{object: {view: visible fraction}} = pixels seen in the scene / pixels the object covers when alone."""
    from .constraint_graph import placed_ids
    ids = placed_ids(objs)
    alone = {i: {n: _pixels({k: o for k, o in objs.items() if raycast.base_name(k) == i}, room, cam).get(i, 0) for n, cam in cams.items()} for i in ids}
    out = {i: {} for i in ids}
    for n, cam in cams.items():
        seen = _pixels(objs, room, cam)
        for i in ids:
            out[i][n] = round(seen.get(i, 0) / alone[i][n], 3) if alone[i][n] else 0.
    return out


def tall_occluders(objs, room, cam):
    """Fraction of the other objects' reference pixels that each tall object hides."""
    from .constraint_graph import placed_ids
    full = _pixels(objs, room, cam)
    out = {}
    for t in placed_ids(objs):
        if core.aabb(objs[t])[5] - core.aabb(objs[t])[4] < TALL_M:
            continue
        rest = {k: o for k, o in objs.items() if raycast.base_name(k) != t}
        without = sum(_pixels(rest, room, cam).values())
        out[t] = round(1 - (sum(full.values()) - full.get(t, 0)) / without, 3) if without else 0.
    return out


def make_c(seed, tier, catalogue=None):
    from .constraint_graph import placed_ids, _true_relations
    from .layout_logic import COMPOSITES, BALLS
    cfg = C_TIERS[tier]
    for attempt in range(120):
        rng = random.Random(f'C-{seed}-{tier}-{attempt}')
        try:
            room, objs = _layout(rng, cfg['n'], cfg['yaw'], cfg['stack'], cfg['jitter'], cfg.get('must', ()), cfg.get('wants'), catalogue=catalogue)
        except RuntimeError:
            continue
        ids = placed_ids(objs)
        if len({i.split('_')[-1] for i in ids}) < len(ids):
            continue                                   # colour is the object's identity in every image
        pub = cameras(room, PUBLIC_VIEWS)
        hides = tall_occluders(objs, room, pub['reference'])
        if any(f > MAX_HIDE for f in hides.values()):
            continue                                   # no single tall object may dominate the reference view
        vis = visibility(objs, room, pub)
        if min(max(v.values()) for v in vis.values()) < MIN_VISIBLE:
            continue                                   # identifiability across the public pool
        witness = perspective = None
        if tier == 'L3':
            occluded = [i for i in ids if vis[i]['reference'] < 0.5 and any(vis[i][q] >= 0.5 for q in pub if q != 'reference')]
            perspective = [c for c in _true_relations(objs, ['appears_left_of', 'appears_larger_than'], pub['reference'])]
            if not occluded or not perspective:
                continue
            witness = rng.choice(occluded)
        objects = []
        for i in ids:
            o = objs[i]
            entry = dict(id=i, color=i.split('_')[-1], **({'composite': COMPOSITES[i]} if i in COMPOSITES else {}))
            if cfg['jitter'] and i not in COMPOSITES and i not in BALLS:
                base = next(c[1] for c in (catalogue if catalogue is not None else __import__('counterscene.spatial_suite.constraint_graph', fromlist=['CATALOG']).CATALOG) if c[0] == i)
                entry['size_range'] = [[round(v * (1 - cfg['jitter']), 3) for v in base], [round(v * (1 + cfg['jitter']), 3) for v in base]]
                entry['size'] = [round(v, 3) for v in base]   # nominal; true size is private
            else:
                entry['size'] = o['size']
            objects.append(entry)
        task = dict(experiment='C', instance=f'C-{tier}-s{seed}', tier=tier, room=room, objects=objects,
                    constraints=[], scene_policy=__import__('counterscene.spatial_suite.layout_logic',fromlist=['policy']).policy(), uses_yaw=cfg['yaw'], query_budget=cfg['queries'],
                    public_cameras=pub, queryable_views=[v for v in PUBLIC_VIEWS if v != 'reference'])
        gold = core.placements_of(objs)
        if cfg['jitter']:
            for i in gold:
                gold[i]['size'] = objs[i]['size']
        priv = dict(gold=gold, hidden_cameras=cameras(room, HIDDEN_VIEWS), visibility=vis, tall_occluder_hides=hides,
                    occlusion_witness=witness, perspective_witness=perspective)
        return task, priv
    raise RuntimeError('C generation failed')


def view_png(task, placements, view, cams=None):
    objs, problems = core.build_objects(task, placements)
    if problems:
        raise ValueError('; '.join(problems))
    cam = (cams or task['public_cameras'])[view]
    return raycast.png(raycast.render(objs, task['room'], cam))


def score_c(task, priv, placements, held_views=('reference',)):
    objs, problems = core.build_objects(task, placements)
    if problems:
        return dict(success=False, reason='invalid_output', problems=problems)
    gobjs, _ = core.build_objects(task, priv['gold'])
    physical = core.global_checks(objs, task['room'])
    hidden = {}
    for n, cam in priv['hidden_cameras'].items():
        hidden[n] = raycast.compare(raycast.render(gobjs, task['room'], cam, SCALE_EVAL), raycast.render(objs, task['room'], cam, SCALE_EVAL))
    seen = {}
    for n in held_views:
        cam = task['public_cameras'][n]
        seen[n] = raycast.compare(raycast.render(gobjs, task['room'], cam, SCALE_EVAL), raycast.render(objs, task['room'], cam, SCALE_EVAL))
    h_iou = sum(v['mean_object_iou'] for v in hidden.values()) / len(hidden)
    s_iou = sum(v['mean_object_iou'] for v in seen.values()) / len(seen)
    err = {i: round(math.dist(core.center(objs[i]), core.center(gobjs[i])), 4) for i in objs}
    return dict(success=not physical and h_iou >= PASS_IOU, physical=physical,
                hidden_mean_iou=round(h_iou, 4), seen_mean_iou=round(s_iou, 4), overfit_gap=round(s_iou - h_iou, 4),
                hidden_depth_err=[None if v['depth_abs_err'] is None else round(v['depth_abs_err'], 4) for v in hidden.values()],
                centre_err_m=err, mean_centre_err_m=round(sum(err.values()) / len(err), 4),
                per_view={n: round(v['mean_object_iou'], 4) for n, v in {**seen, **hidden}.items()})


def held_view_iou(task, priv, placements, views):
    """What the `compare` tool returns: silhouette agreement only on views the agent already holds."""
    objs, problems = core.build_objects(task, placements)
    if problems:
        return dict(error='; '.join(problems))
    gobjs, _ = core.build_objects(task, priv['gold'])
    out = {}
    for n in views:
        cam = task['public_cameras'][n]
        c = raycast.compare(raycast.render(gobjs, task['room'], cam, SCALE_EVAL), raycast.render(objs, task['room'], cam, SCALE_EVAL))
        # shortcut: per-view mean only; per-object IoU would be a hill-climbing oracle (found in pilot 1)
        out[n] = dict(mean_object_iou=round(c['mean_object_iou'], 3))
    return out


def overfit_baseline(task, priv, k=1.12):
    """Projective cheat: scale the gold scene about the reference camera. The reference image is
    unchanged, but objects float/sink and hidden views disagree."""
    eye = task['public_cameras']['reference']['eye']
    out = {}
    for o in task['objects']:
        g = priv['gold'][o['id']]
        size = g.get('size', o['size'])
        objs, _ = core.build_objects(task, priv['gold'])
        c = core.center(objs[o['id']])
        c2 = [eye[i] + k * (c[i] - eye[i]) for i in range(3)]
        s2 = [v * k for v in size]
        if o.get('size_range'):
            s2 = [min(max(s2[i], o['size_range'][0][i]), o['size_range'][1][i]) for i in range(3)]
        out[o['id']] = dict(position=[round(c2[0], 4), round(c2[1], 4), round(c2[2] - s2[2] / 2, 4)], yaw=g['yaw'], **({'size': s2} if o.get('size_range') else {}))
    return out
