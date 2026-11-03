"""Scripted (non-model) policies used for gold replay and shortcut / attack audits.

Each policy is a function (env, rng) -> list of actions; it runs through the real Env, so
the same tools, budgets and scorers apply as for a model. None of these use an API.
"""
import copy
import math
import random

from . import constraint_graph as cg, core, grounding, kinematics, scenes
from .. import program


def run(env, actions):
    for a in actions:
        if env.done:
            break
        env.step(a)
    return env.final()


def _gold(env):
    return copy.deepcopy(env.priv['gold'])


# ---------------- A
def a_gold(env, rng):
    return [dict(tool='submit', placements=_gold(env))]


def a_random(env, rng):
    W, D, _ = env.task['room']
    p = {o['id']: dict(position=[round(rng.uniform(0.3, W - 0.3), 2), round(rng.uniform(0.3, D - 0.3), 2), 0.], yaw=0) for o in env.task['objects']}
    return [dict(tool='submit', placements=p)]


def a_oracle_search(env, rng):
    """Uses the check tool as a black-box oracle: random restarts + greedy perturbation, no reasoning."""
    W, D, _ = env.task['room']
    best, best_n = None, -1e9
    acts = []
    cur = {o['id']: dict(position=[round(rng.uniform(0.3, W - 0.3), 2), round(rng.uniform(0.3, D - 0.3), 2), 0.], yaw=0) for o in env.task['objects']}
    for _ in range(env.max_turns - 1):
        rep = core.check_scene(env.task, cur)
        n = sum(r['satisfied'] for r in rep['constraints']) - len(rep['physical'])
        env.step(dict(tool='check', placements=cur))
        if n > best_n:
            best, best_n = copy.deepcopy(cur), n
        cur = copy.deepcopy(best)
        i = rng.choice(list(cur))
        cur[i]['position'][0] = round(min(max(cur[i]['position'][0] + rng.uniform(-.6, .6), .3), W - .3), 2)
        cur[i]['position'][1] = round(min(max(cur[i]['position'][1] + rng.uniform(-.6, .6), .3), D - .3), 2)
    acts.append(dict(tool='submit', placements=best))
    return acts


def a_mirror(env, rng):
    """Left/right frame confusion: gold mirrored in x (a systematic-error probe for the scorer)."""
    W = env.task['room'][0]
    p = _gold(env)
    for v in p.values():
        v['position'][0] = round(W - v['position'][0], 4)
    return [dict(tool='submit', placements=p)]


# ---------------- B
def b_gold(env, rng):
    inf = [i for i, v in env.priv['soft'].items() if not v['feasible']]
    return [dict(tool='submit', placements=_gold(env), infeasible=[dict(id=i, reason='certificate') for i in inf])]


def b_noop(env, rng):
    return [dict(tool='submit', placements=copy.deepcopy(env.task['initial']), infeasible=[])]


def b_resolve_all(env, rng):
    """Satisfies hard constraints by re-solving from scratch (gold), but claims every soft constraint infeasible."""
    soft = [c['id'] for c in env.task['constraints'] if c['priority'] == 'soft']
    return [dict(tool='submit', placements=_gold(env), infeasible=[dict(id=i, reason='skip') for i in soft])]


def b_greedy_fixer(env, rng):
    """Fixes each reported violation in isolation by moving a movable object next to its partner."""
    t = env.task
    cur = copy.deepcopy(t['initial'])
    objs, _ = core.build_objects(t, cur)
    for r in t['initial_report']['constraints']:
        if r['satisfied']:
            continue
        c = next(x for x in t['constraints'] if x['id'] == r['id'])
        if c['priority'] == 'soft':
            continue
        mov = [a for a in c['args'] if a not in t['locked'] and objs[a]['position'][2] == 0]
        if not mov:
            continue
        m = mov[0]
        other = next(a for a in c['args'] if a != m)
        ob = core.aabb(objs[other])
        fx, fy = core.footprint_size(objs[m])
        if c['type'] in ('left_of', 'right_of', 'touching'):
            first = c['args'][0] == m
            left = (c['type'] == 'left_of') == first
            x = ob[0] - fx / 2 if left else ob[1] + fx / 2
            cur[m]['position'] = [round(x, 4), round((ob[2] + ob[3]) / 2, 4), 0.]
        elif c['type'] in ('in_front_of', 'behind'):
            first = c['args'][0] == m
            front = (c['type'] == 'in_front_of') == first
            y = ob[2] - fy / 2 if front else ob[3] + fy / 2
            cur[m]['position'] = [round((ob[0] + ob[1]) / 2, 4), round(y, 4), 0.]
        else:
            cur[m]['position'] = copy.deepcopy(env.priv['gold'][m]['position'])
        objs, _ = core.build_objects(t, cur)
    return [dict(tool='check', placements=cur), dict(tool='submit', placements=cur, infeasible=[])]


# ---------------- C
def c_gold(env, rng):
    return [dict(tool='submit', placements=_gold(env))]


def c_overfit(env, rng):
    return [dict(tool='submit', placements=grounding.overfit_baseline(env.task, env.priv))]


def c_overfit_dropped(env, rng):
    p = grounding.overfit_baseline(env.task, env.priv)
    for v in p.values():
        v['position'][2] = 0.
    return [dict(tool='submit', placements=p)]


def c_depth_noise(env, rng):
    """Correct image-plane layout but each object slid 0.3 m along the reference viewing ray (a depth error)."""
    eye = env.task['public_cameras']['reference']['eye']
    p = _gold(env)
    for i, v in p.items():
        x, y, z = v['position']
        d = [x - eye[0], y - eye[1]]
        n = math.hypot(*d)
        s = rng.choice([-1, 1]) * 0.3
        v['position'] = [round(x + s * d[0] / n, 4), round(y + s * d[1] / n, 4), z]
    return [dict(tool='submit', placements=p)]


def c_near_miss(env, rng):
    """Every floor object 0.10 m off in a random direction: must fail (threshold calibration)."""
    p = _gold(env)
    for v in p.values():
        if v['position'][2] == 0:
            a = rng.uniform(0, 2 * math.pi)
            v['position'][0] = round(v['position'][0] + 0.1 * math.cos(a), 4)
            v['position'][1] = round(v['position'][1] + 0.1 * math.sin(a), 4)
    return [dict(tool='submit', placements=p)]


# ---------------- D
def d_gold(env, rng):
    return [dict(tool='submit', **env.priv['gold'])]


def d_static(env, rng):
    return [dict(tool='submit', **kinematics.static_baseline(env.task, env.priv['gold']))]


def d_lift(env, rng):
    return [dict(tool='submit', **kinematics.lift_baseline(env.task, env.priv['gold']))]


def d_over_limit(env, rng):
    g = copy.deepcopy(env.priv['gold'])
    w = [g['keyframes'][i] for i in g['waypoint_keyframes']]
    # limits sized to the waypoint poses only, so the detour keyframes fall outside them
    g['mechanism']['joint_limits_deg'] = [[min(q[j] for q in w) - 1, max(q[j] for q in w) + 1] for j in range(3)]
    return [dict(tool='submit', **g)]


# ---------------- E
def e_skeptic(env, rng):
    return [dict(tool='submit', placements=_gold(env), checker_issues=[dict(id=i, issue=v) for i, v in env.priv['planted'].items()])]


def e_trusting(env, rng):
    """Perfectly obeys the noisy checker: fixes exactly what it reports, stops when it says all pass."""
    t, cur = env.task, copy.deepcopy(env.task['initial'])
    acts = []
    for _ in range(env.max_turns - 1):
        rep = cg.noisy_check(t, env.priv['faults'], cur, env.turn + len(acts) + 1)
        acts.append(dict(tool='check', placements=copy.deepcopy(cur)))
        failing = [r for r in rep['constraints'] if not r['satisfied']]
        if rep['all_satisfied'] or not failing:
            break
        for r in failing:                      # moves every object in a reported failure to its gold spot
            for a in r['args']:                # ...or, for false failures, nudges it to try to "fix" it
                if r['id'] in env.priv['faults'].get('false_fail', []):
                    cur[a]['position'][0] = round(cur[a]['position'][0] + 0.25, 4)
                else:
                    cur[a] = copy.deepcopy(env.priv['gold'][a])
    acts.append(dict(tool='submit', placements=cur, checker_issues=[]))
    return acts


# ---------------- Task 1 (program answers)
def t1_gold(env, rng):
    return [dict(tool='submit', source=env.priv['gold_program'])]


def t1_default_choices(env, rng):
    """Gold positions, but every object keeps its middle size and first colour: tests that size/colour choices are scored."""
    p = _gold(env)
    for o in env.task['objects']:
        p[o['id']]['size'] = o['size_options'][1]
        p[o['id']]['color'] = o['colors'][0]
    return [dict(tool='submit', source=program.literal_program(p))]


def t1_mirror(env, rng):
    W = env.task['room'][0]
    p = _gold(env)
    for v in p.values():
        v['position'][0] = round(W - v['position'][0], 4)
    return [dict(tool='submit', source=program.literal_program(p))]


def t1_no_occlusion(env, rng):
    """Every relation right except visibility: occluders pushed 0.6 m back behind what they should hide (or sideways when that collides).
    Only meaningful when the instance has an occludes constraint; otherwise identical to gold, so it is skipped by the audit."""
    p = _gold(env)
    for c in env.task['constraints']:
        if c['type'] == 'occludes':
            a = c['args'][0]
            p[a]['position'][0] = round(p[a]['position'][0] + 0.8, 4)
    return [dict(tool='submit', source=program.literal_program(p))]


def t1_loop_program(env, rng):
    """A natural program with a for-loop: outside the allowed subset, so it must count as an execution failure."""
    src = env.priv['gold_program'].replace('    return ', '    scene = {}\n    for k in []:\n        scene = scene\n    return ', 1)
    return [dict(tool='submit', source=src)]


def t1_random(env, rng):
    W, D, _ = env.task['room']
    p = {o['id']: dict(position=[round(rng.uniform(0.4, W - 0.4), 2), round(rng.uniform(0.4, D - 0.4), 2), 0.], yaw=0,
                       size=o['size_options'][1], color=o['colors'][0]) for o in env.task['objects']}
    return [dict(tool='submit', source=program.literal_program(p))]


# ---------------- Task 2 (repair)
def t2_gold(env, rng):
    return [dict(tool='submit', source=scenes.scene_program(env.task, env.priv['gold']))]


def t2_noop(env, rng):
    return [dict(tool='submit', source=env.task['initial_program'])]


def t2_greedy_local(env, rng):
    """The naive repairer: one run per step; each step fixes the first problem locally (moves the constraint's first
    argument, or pushes the first interpenetrating object out). Logs a state vector per run like a model would."""
    t = env.task
    cur = copy.deepcopy(env.priv['initial'])
    acts = []
    for _ in range(env.max_turns - 1):
        rep = core.check_scene(t, cur)
        if rep['all_satisfied']:
            break
        pen = [p for p in rep['physical'] if p['check'] == 'no_interpenetration']
        nxt = None
        if pen:
            m = pen[0]['objects'][0]
            for s in scenes.AXIS_STEPS:
                for dx, dy in ((s, 0), (-s, 0), (0, s), (0, -s)):
                    cand = scenes._move_with(t, cur, m, dx, dy)
                    r2 = core.check_scene(t, cand)
                    if r2['valid_output'] and not any(p['check'] == 'no_interpenetration' and m in p['objects'] for p in r2['physical']):
                        nxt = cand
                        break
                if nxt:
                    break
        else:
            bad = [r['id'] for r in rep['constraints'] if not r['satisfied']]
            for cid in bad:
                nxt = scenes.naive_fix(t, cur, cid)
                if nxt:
                    break
        if not nxt:
            break
        cur = nxt
        acts.append(dict(tool='run', source=scenes.scene_program(t, cur)))
    acts.append(dict(tool='submit', source=scenes.scene_program(t, cur)))
    return acts


def t2_revert_all(env, rng):
    """Moves every object back to gold one at a time, checking after each: reaches success with non-monotonic steps allowed."""
    return t2_gold(env, rng)


# ---------------- Task 3 (reconstruction)
def t3_gold(env, rng):
    return [dict(tool='submit', placements=_gold(env))]


def t3_facade_table(env, rng):
    return [dict(tool='submit', placements=scenes.facade_table_baseline(env.task, env.priv))]


def t3_billboard_all(env, rng):
    return [dict(tool='submit', placements=scenes.billboard_all_baseline(env.task, env.priv))]


def t3_depth_slide(env, rng):
    """Every simple object slid 0.3 m along the reference view ray and rescaled to keep its image size."""
    t = env.task
    eye = t['public_cameras']['reference']['eye']
    p = _gold(env)
    for o in t['objects']:
        if o.get('parts'):
            continue
        g = p[o['id']]
        c = g['position']
        d = [c[0] - eye[0], c[1] - eye[1]]
        n = math.hypot(*d)
        k = (n + 0.3) / n
        size = [min(max(v * k, o['size_range'][0][i]), o['size_range'][1][i]) for i, v in enumerate(g['size'])]
        g['position'] = [round(eye[0] + d[0] * k, 4), round(eye[1] + d[1] * k, 4), 0.]
        g['size'] = [round(v, 3) for v in size]
    return [dict(tool='submit', placements=p)]


POLICIES = {
    'A': {'gold': a_gold, 'random': a_random, 'oracle_search': a_oracle_search, 'mirror_x': a_mirror},
    'B': {'gold': b_gold, 'no_op': b_noop, 'resolve_claim_all_soft': b_resolve_all, 'greedy_fixer': b_greedy_fixer},
    'C': {'gold': c_gold, 'reference_overfit': c_overfit, 'overfit_dropped_to_floor': c_overfit_dropped, 'depth_slide_0.3m': c_depth_noise, 'near_miss_0.10m': c_near_miss},
    'D': {'gold': d_gold, 'pose_only_static': d_static, 'generic_lift_detour': d_lift, 'declared_limit_too_narrow': d_over_limit},
    'E': {'gold_skeptic': e_skeptic, 'trusting': e_trusting},
    'T1': {'gold': t1_gold, 'random': t1_random, 'mirror_x': t1_mirror, 'default_size_colour': t1_default_choices,
           'occluder_shifted': t1_no_occlusion, 'loop_program': t1_loop_program},
    'T2': {'gold': t2_gold, 'no_op': t2_noop, 'greedy_local': t2_greedy_local},
    'T3': {'gold': t3_gold, 'facade_table': t3_facade_table, 'billboard_all': t3_billboard_all, 'depth_slide_0.3m': t3_depth_slide},
}


def d_no_grasp(env, rng):
    sub = copy.deepcopy(env.priv['gold'])
    sub['events'] = []
    return [dict(tool='submit', **sub)]


def d_early_release(env, rng):
    """Releases the first ball one keyframe after grasping it, mid-carry."""
    sub = copy.deepcopy(env.priv['gold'])
    sub['events'][1]['keyframe'] = sub['events'][0]['keyframe'] + 1
    return [dict(tool='submit', **sub)]


def d_grasp_from_home(env, rng):
    """Never moves to the ball: claims the grasp at the home pose."""
    sub = copy.deepcopy(env.priv['gold'])
    sub['events'][0]['keyframe'] = 0
    return [dict(tool='submit', **sub)]


def d_skip_return_home(env, rng):
    if not env.task.get('return_home'):
        return []                      # not applicable below L3
    sub = copy.deepcopy(env.priv['gold'])
    sub['keyframes'] = sub['keyframes'][:sub['waypoint_keyframes'][-1] + 1]
    return [dict(tool='submit', **sub)]


def d_grasp_second_first(env, rng):
    """Picks ball 2 before placing ball 1 (one gripper)."""
    if len(env.task['balls']) < 2:
        return []
    sub = copy.deepcopy(env.priv['gold'])
    e = sub['events']
    e[1], e[2] = dict(e[1], action='grasp', ball='ball_2'), dict(e[2], action='release', ball='ball_1')
    return [dict(tool='submit', **sub)]


def d_revisit_placed_ball(env, rng):
    """After placing ball 1, the arm goes back down into it: a released ball is a static obstacle."""
    if len(env.task['balls']) < 2:
        return []
    return [dict(tool='submit', **kinematics.revisit_placed_ball(env.task, env.priv['gold']))]


def d_release_above_support(env, rng):
    """Lets ball 1 go 0.10 m above its cabinet top instead of setting it down."""
    if not any(b['support'] != 'floor' for b in env.task['balls']):
        return []
    return [dict(tool='submit', **kinematics.release_above_support(env.task, env.priv['gold']))]


def d_release_other_height(env, rng):
    """Releases ball 1 at the resting height of a different (lower) cabinet."""
    if not any(b['support'] != 'floor' for b in env.task['balls']):
        return []
    return [dict(tool='submit', **kinematics.release_at_other_cabinet_height(env.task, env.priv['gold']))]


POLICIES['D'].update(release_above_cabinet=d_release_above_support, release_at_other_cabinet_height=d_release_other_height, no_grasp_event=d_no_grasp, early_release=d_early_release, grasp_from_home=d_grasp_from_home,
                     skip_return_home=d_skip_return_home, grasp_second_before_first_placed=d_grasp_second_first,
                     revisit_placed_ball=d_revisit_placed_ball)
