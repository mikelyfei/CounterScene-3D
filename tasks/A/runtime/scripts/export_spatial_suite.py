"""Export every spatial-suite instance as four parts plus a manifest:

  public.json            what the model sees (English prompt incl. success criterion, images listed separately)
  private.json           evaluator-only data (gold, hidden cameras, planted faults, conflict certificates)
  gold_verification.json gold replay + scripted shortcut/attack baselines through the same scorer
  *.png                  model-visible reference image(s) (C) and review-only hidden views (C, prefixed hidden_)

  blender_scene_*.json   (T1-T3) realistic-render input for counterscene/spatial_suite/blender_render.py

  python -m scripts.export_spatial_suite --out results/spatial_suite/instances --seeds 0 1 2 --exps T1 T2 T3
  # after rendering the blender_scene files into <renders>/<instance>/<view>.png, re-export with
  # --render-root <renders> so Task 3 shows the model the realistic renders.
"""
import argparse
import base64
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from counterscene.session import write_json
from counterscene.spatial_suite import protocol, baselines, constraint_graph as cg, core, grounding, kinematics, scenes


def scene_boxes(task, placements):
    objs, _ = core.build_objects(task, placements)
    models = {o['id']: o.get('model') for o in task['objects']}
    out = []
    for i, o in objs.items():
        support='floor' if abs(core.aabb(o)[4])<=core.TOL else next((name for name,host in objs.items() if name!=i and abs(core.aabb(o)[4]-core.aabb(host)[5])<=core.TOL and core.rel_on_top_of(o,host)[0]),'unsupported')
        b = dict(id=i, aabb=[round(v, 4) for v in core.aabb(o)], yaw=o['yaw'],support=support)
        if o.get('envelope'):
            b['envelope'] = True          # drawn through its parts
        if o.get('rigid'):
            b['part_of'] = o['group']
        if o.get('color'):
            b['color'] = o['color']
        if models.get(i):
            b.update(model=models[i], size=o['size'], position=o['position'])
        out.append(b)
    return out


def blender_scene(task, placements, cams):
    """Input for blender_render.py: meshes fitted into the scored boxes; composite parts drawn as wood boxes."""
    objs, problems = core.build_objects(task, placements)
    assert not problems, problems
    rows = []
    for o in task['objects']:
        if o.get('parts'):
            parts = [dict(size=objs[k]['size'], position=objs[k]['position']) for k in objs if k.split('#')[0] == o['id']]
            rows.append(dict(id=o['id'], model=o.get('model'), parts=parts, color=o.get('color')))
        else:
            x = objs[o['id']]
            rows.append(dict(id=o['id'], model=o['model'], box=dict(size=x['size'], position=x['position'], yaw=x['yaw']), color=x.get('color')))
    views = {n: dict(c, width=640, height=480) for n, c in cams.items()}
    return dict(room=task['room'], camera_views=views, objects=rows)


def t_cameras(env):
    t = env.task
    if env.experiment == 'T3':
        return {**t['public_cameras'], **env.priv['hidden_cameras']}
    return grounding.cameras(t['room'], ['reference', 'front_left', 'front_right', 'top'])


def review_geometry(env):
    """Compact geometry for the review page's 3D viewer (not model input)."""
    t, p = env.task, env.priv
    if env.experiment == 'D':
        def path(sub):
            mech = dict(base=sub['mechanism']['base'], link_lengths=sub['mechanism']['link_lengths'])
            pts, keys = [], sub['keyframes']
            for n in range(len(keys) - 1):
                for s in range(12):
                    q = [keys[n][j] + s / 12 * (keys[n + 1][j] - keys[n][j]) for j in range(3)]
                    pts.append([[round(v, 3) for v in x] for x in kinematics.fk(mech, q)])
            pts.append([[round(v, 3) for v in x] for x in kinematics.fk(mech, keys[-1])])
            return pts
        return dict(kind='arm', room=t['room'], obstacles=t['obstacles'], waypoints=t['waypoints'], radius=t['link_radius'], balls=t['balls'],
                    home=t['home'], return_home=t['return_home'], events=p['gold']['events'], frames_per_segment=12,
                    gold=path(p['gold']), static=path(kinematics.static_baseline(t, p['gold'])))
    g = dict(kind='boxes', room=t['room'], gold=scene_boxes(t, p['gold']))
    if env.experiment == 'E':
        g['conflicts'] = p['conflicts']
        g['clean_objects'] = p['clean_objects']
    if 'initial' in t:
        g['initial'] = scene_boxes(t, t['initial'])
    if env.experiment == 'T2':
        g['initial'] = scene_boxes(t, p['initial'])
    if env.experiment == 'T3':
        g['facade'] = scene_boxes(t, scenes.facade_table_baseline(t, p))
        g['cameras'] = {**{k: dict(v, public=True) for k, v in t['public_cameras'].items()}, **{k: dict(v, public=False) for k, v in p['hidden_cameras'].items()}}
    if env.experiment == 'C':
        g['overfit'] = [b for b in scene_boxes(t, grounding.overfit_baseline(t, p))]
        g['cameras'] = {**{k: dict(v, public=True) for k, v in t['public_cameras'].items()}, **{k: dict(v, public=False) for k, v in p['hidden_cameras'].items()}}
    return g


def export(env, dest):
    dest.mkdir(parents=True, exist_ok=True)
    text, images = env.prompt()
    names = []
    for n, img in enumerate(images):
        names.append(f'model_input_{n}.png')
        (dest / names[-1]).write_bytes(img)
    public = dict(instance=env.task['instance'], experiment=env.experiment, prompt=text, images=names, max_turns=env.max_turns)
    if env.experiment == 'C':
        for v, cam in env.priv['hidden_cameras'].items():
            (dest / f'hidden_{v}.png').write_bytes(grounding.view_png(env.task, env.priv['gold'], v, env.priv['hidden_cameras']))
        for v in env.task['queryable_views']:
            (dest / f'query_{v}.png').write_bytes(grounding.view_png(env.task, env.priv['gold'], v))
    if env.experiment in ('T1', 'T2', 'T3'):
        cams = t_cameras(env)
        write_json(dest / 'blender_scene_gold.json', blender_scene(env.task, env.priv['gold'], cams))
        if env.experiment == 'T2':
            write_json(dest / 'blender_scene_initial.json', blender_scene(env.task, env.priv['initial'], cams))
        if env.experiment == 'T3':
            write_json(dest / 'blender_scene_facade.json', blender_scene(env.task, scenes.facade_table_baseline(env.task, env.priv), cams))
            for v, cam in env.priv['hidden_cameras'].items():      # box renders for review only (never model input)
                objs, _ = core.build_objects(env.task, env.priv['gold'])
                (dest / f'hidden_{v}.png').write_bytes(scenes.raycast.png(scenes.raycast.render(objs, env.task['room'], cam)))
    write_json(dest / 'public.json', public)
    write_json(dest / 'gold_answer.json',dict(placements=env.priv['gold'],scope='private evaluator answer; may not be unique'))
    write_json(dest / 'private.json', dict(env.priv, task_internal={k: v for k, v in env.task.items() if k not in ('public_cameras',)}))
    agent=dest/'agent_input';agent.mkdir(exist_ok=True)
    write_json(agent/'public.json',public)
    for name in names:(agent/name).write_bytes((dest/name).read_bytes())
    ver = {}
    for name, pol in baselines.POLICIES[env.experiment].items():
        e2 = protocol.make_env(env.experiment, env._seed, env._level, render_root=getattr(env, '_render_root', None))
        acts = pol(e2, random.Random(env._seed))
        if not acts:                      # probe does not apply at this tier (e.g. two-ball probes below L3)
            ver[name] = dict(success=None, role='not_applicable')
            continue
        res = baselines.run(e2, acts)
        ver[name] = dict(success=bool(res.get('success')), metrics={k: v for k, v in res.items() if k not in ('report',)})
        if env.experiment in ('T1', 'T2'):
            ver[name]['runs'] = len(getattr(e2, 'runs', []))
    references={'oracle_search','greedy_fixer','greedy_local','generic_lift_detour','near_miss_0.10m','trusting'}
    for name,row in ver.items():
        if row.get('role')=='not_applicable':continue
        row['role']='gold' if name.startswith('gold') else 'reference_baseline' if name in references else 'attack_probe'
    gold_key = next(k for k in ver if k.startswith('gold'))
    write_json(dest / 'gold_verification.json', dict(gold_passes=ver[gold_key]['success'],
               shortcuts_blocked=all(not v['success'] for k, v in ver.items() if v['role']=='attack_probe'), policies=ver))
    write_json(dest / 'review_geometry.json', review_geometry(env))
    return dict(instance=env.task['instance'], experiment=env.experiment, level=str(env._level), seed=env._seed,
                gold_passes=ver[gold_key]['success'], shortcuts=ver)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', default='results/spatial_suite/instances')
    a.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    a.add_argument('--exps', nargs='+', default=['T1', 'T2', 'T3'])
    a.add_argument('--no-sweep',action='store_true',help='export tier cells only; omit the optional A constraint-count sweep')
    a.add_argument('--render-root', default=None, help='folder with <instance>/<view>.png realistic renders (Task 3 model input)')
    args = a.parse_args()
    exps = [e for x in args.exps for e in (list(x) if x in ('ABCDE', 'ABC', 'DE') else [x])]
    out = ROOT / args.out
    rows = []
    for exp in exps:
        levels = ['L1', 'L2', 'L3'] + ([k for k in cg.SWEEP] if exp == 'A' and not args.no_sweep else [])
        for lv in levels:
            for s in args.seeds:
                rr = str(ROOT / args.render_root) if args.render_root else None
                env = protocol.make_env(exp, s, lv, render_root=rr)
                env._seed, env._level, env._render_root = s, lv, rr
                rows.append(export(env, out / env.task['instance']))
                print(rows[-1]['instance'], 'gold ok' if rows[-1]['gold_passes'] else 'GOLD FAIL', flush=True)
    write_json(out / 'manifest.json', dict(version='spatial-suite-logical-rooms-pick-place-v2', seeds=args.seeds, split='development (seeds 0-99); test seeds >= 1000 are generated only at evaluation time',
                                           instances=rows))


if __name__ == '__main__':
    main()
