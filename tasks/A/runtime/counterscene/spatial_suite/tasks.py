"""Model-facing environments for Tasks 1-3 (iteration 3). Same episode contract as protocol.Env."""
import json
import os

from . import core, raycast, scenes
from .protocol import ACTION_RULE, FRAME, Env, relation_glossary, constraint_lines

PHYSICS_T = ('Physical rules (always checked): every object stays inside the room; no two boxes interpenetrate by more than 0.01 m; '
             'every box rests on the floor (bottom_z = 0) or on top of another box (bottom_z equals that box\'s top within 0.01 m and its '
             'footprint centre lies over it; a part of a composite object may also rest on another part of the same object it overlaps).')
PROGRAM_RULE = ('The program language is a restricted Python subset: exactly one function `def build(public_params):` whose body uses only '
                'assignments to plain variable names, if/else, return, list comprehensions, arithmetic, comparisons, list/dict literals, '
                'subscripting, and the builtins range, len, min, max, abs, round. Not allowed: import, for/while loops, attribute access, '
                'item assignment (d[k] = v), nested function definitions. public_params = {"room": [W, D, H]}. build must return the '
                'placements dict.')
MESH_NOTE = 'Objects are furniture models; each object\'s box is the tight axis-aligned bounding box of its model.'


def furniture_block(task, choices=True):
    rows = []
    for o in task['objects']:
        if o.get('size_options') and choices:
            rows.append(f'- {o["id"]}: size options {o["size_options"]}; colour options {o["colors"]}')
        elif o.get('parts'):
            rows.append(f'- {o["id"]}: COMPOSITE of {o["parts"]} boxes (table top + legs), each part size within {o["part_size_range"]}; colour {o["color"]}')
        elif o.get('size_range'):
            rows.append(f'- {o["id"]}: size between {o["size_range"][0]} and {o["size_range"][1]} (nominal {o["size"]}); colour {o["color"]}')
        else:
            rows.append(f'- {o["id"]}: size {o["size"]}, colour {o["color"]}')
    return '\n'.join(rows)


def program_report(res):
    if not res['executed']:
        return dict(executed=False, error=res['error'])
    rep = res['report']
    if not rep['valid_output']:
        return dict(executed=True, valid_output=False, problems=rep['output_problems'])
    return dict(executed=True, all_satisfied=rep['all_satisfied'], physical=rep['physical'],
                constraints=[dict(id=r['id'], satisfied=r['satisfied'], value=r['value']) for r in rep['constraints']])


class ProgramEnv(Env):
    """Shared by T1 and T2: actions carry a program source; every run is recorded."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.runs = []          # (executed, report, placements)

    def _run(self, source):
        res, placements = scenes.run_program(self.task, source)
        self.runs.append((res['executed'], res.get('report'), placements))
        return res

    def step(self, action):
        if isinstance(action, dict) and action.get('tool') == 'submit' and 'source' in action:
            self._run(action['source'])
        return super().step(action)

    def tool(self, a):
        if a['tool'] == 'run':
            return program_report(self._run(a.get('source')))
        return dict(error=f'Unknown tool {a["tool"]}')


class T1Env(ProgramEnv):
    experiment = 'T1'

    def prompt(self):
        t = self.task
        text = f"""TASK (Task 1, multi-constraint scene generation, instance {t['instance']})
Write a scene program that furnishes the room so that ALL constraints below hold at once. For every object you also choose
one of its sizes and one of its colours.

{FRAME}
{MESH_NOTE}
Room size W x D x H = {t['room']}.
{PHYSICS_T}

Objects (sizes are [x, y, z] at yaw 0; the front of every object faces -y at yaw 0):
{furniture_block(t)}

Reference camera (used by "occludes"): {json.dumps(t['camera'])}. Pinhole, looks from eye to target, up = +z, vertical fov fov_deg.

Relation definitions:
{relation_glossary(t['relation_types'])}

Constraints ({len(t['constraints'])}):
{constraint_lines(t['constraints'])}

Answer format: a program. {PROGRAM_RULE}
Each placement is {{"position": [x, y, bottom_z], "yaw": 0|90|180|270, "size": <one of the size options>, "color": <one of the colour options>}}.

Tools:
- {{"tool": "run", "source": "<program>"}}: executes the program and returns the verifier report (every constraint with satisfied/value, physical issues) or the execution error.
- {{"tool": "submit", "source": "<program>"}}: final answer; ends the episode.

Success criterion: the submitted program executes and its scene satisfies every constraint and every physical rule.
Reported metrics: constraint satisfaction rate, collision-free (no interpenetration), execution success, and the number of program runs until the first fully correct one (iterations to success).
{self._budget()}
{ACTION_RULE}"""
        return text, []

    def final(self):
        if not self.submission or 'source' not in self.submission:
            return dict(success=False, reason='no_submission', execution_success=False, csr=0., collision_free=False,
                        iterations_to_success=next((n + 1 for n, r in enumerate(self.runs) if r[0] and r[1].get('all_satisfied')), None))
        return scenes.score_t1(self.task, [(e, r) for e, r, _ in self.runs])


class T2Env(ProgramEnv):
    experiment = 'T2'

    def prompt(self):
        t = self.task
        text = f"""TASK (Task 2, iterative scene repair, instance {t['instance']})
The scene program below was written by a previous agent. Its scene violates some constraints and physical rules. Repair it:
make every constraint and physical rule hold, changing as little as necessary. Watch out: a fix that looks local can break a
constraint that currently holds.

{FRAME}
{MESH_NOTE}
Room size W x D x H = {t['room']}.
{PHYSICS_T}

Objects (sizes and colours are fixed):
{furniture_block(t)}

Reference camera (used by "occludes"): {json.dumps(t['camera'])}.

Relation definitions:
{relation_glossary(t['relation_types'])}

Constraints ({len(t['constraints'])}):
{constraint_lines(t['constraints'])}

Current scene program:
```python
{t['initial_program']}```
{PROGRAM_RULE}
Each placement is {{"position": [x, y, bottom_z], "yaw": 0|90|180|270}}.

Tools:
- {{"tool": "run", "source": "<program>"}}: executes your edited program and returns the verifier report or the execution error.
- {{"tool": "submit", "source": "<program>"}}: final answer; ends the episode.

Success criterion: the submitted program executes and its scene satisfies every constraint and physical rule.
Also reported: the constraint state vector after every run, the Regression Rate (previously satisfied constraints that a later
run breaks, divided by previously satisfied constraints re-checked), whether the repair was monotonic, and which objects you moved.
{self._budget()}
{ACTION_RULE}"""
        return text, []

    def vectors(self):
        init = scenes.state_vector(self.task, core.check_scene(self.task, self.priv['initial']))
        return [init] + [scenes.state_vector(self.task, r) if e else None for e, r, _ in self.runs]

    def final(self):
        vec = self.vectors()
        if not self.submission or 'source' not in self.submission:
            return dict(success=False, reason='no_submission', **scenes.trajectory_metrics(vec), state_vectors=vec)
        e, r, pl = self.runs[-1]
        out = scenes.score_t2(self.task, self.priv, vec, pl, e)
        out['state_vectors'] = vec
        return out


class T3Env(Env):
    experiment = 'T3'

    def __init__(self, task, priv, max_turns, oneshot=False, images=None):
        super().__init__(task, priv, max_turns, oneshot)
        self.held = ['reference']
        self.images = images or {}

    def view(self, v):
        if v in self.images:
            return self.images[v]
        objs, _ = core.build_objects(self.task, self.priv['gold'])
        return raycast.png(raycast.render(objs, self.task['room'], self.task['public_cameras'][v]))

    def prompt(self):
        t = self.task
        realistic = bool(self.images)
        text = f"""TASK (Task 3, multi-view 3D reconstruction, instance {t['instance']})
The attached image shows a hidden room from the "reference" camera. Reconstruct the 3D scene as boxes.
Your reconstruction is verified from OTHER cameras you never see (behind and beside the scene) and must agree with the hidden
scene there. A reconstruction that only looks right from the reference view (flat facades, sheets, objects slid along the view
ray) will fail.

{FRAME}
{MESH_NOTE if realistic else 'Each object is drawn as a single-colour box.'} The room has a wooden floor; only the back and left walls are drawn.
Room size W x D x H = {t['room']}.
{PHYSICS_T}

Objects (true sizes lie inside the given ranges; include "size" for each simple object):
{furniture_block(t, choices=False)}

Camera model: pinhole, looks from "eye" to "target", up = +z, vertical field of view fov_deg, image width x height pixels.
Reference camera: {json.dumps(t['public_cameras']['reference'])}
Other public cameras you may request (budget {t['query_budget']} extra views): {json.dumps({n: t['public_cameras'][n] for n in t['queryable_views']})}

Tools:
- {{"tool": "query_view", "view": "<name>"}}: the hidden scene from that public camera (counts against the budget).
- {{"tool": "render", "placements": {{...}}, "view": "<public camera name>"}}: renders YOUR candidate (as coloured boxes) from any public camera.
- {{"tool": "compare", "placements": {{...}}}}: mean per-object silhouette IoU (one number per view) between your candidate and the hidden scene, only on views you already hold.
- {{"tool": "submit", "placements": {{...}}}}
"placements": simple objects {{"position": [x, y, bottom_z], "yaw": 0, "size": [x, y, z]}}; the composite {t['composite']} is {{"parts": [{{"size": [x, y, z], "position": [x, y, bottom_z]}}, ...]}} with exactly 5 axis-aligned parts.

Success criterion: physical rules hold, the mean per-object silhouette IoU over the hidden cameras is at least {scenes.T3_PASS_MEAN}, and every object reaches at least {scenes.T3_PASS_MIN} there.
Reported: R_visible (IoU on views you saw), R_3D (hidden-view IoU), their gap, hidden-view depth error and 3D centre error per object.
{self._budget()}
{ACTION_RULE}"""
        return text, [self.view('reference')]

    def tool(self, a):
        t = self.task
        if a['tool'] == 'query_view':
            v = a.get('view')
            if v not in t['queryable_views']:
                return dict(error=f'unknown view {v}')
            if v not in self.held:
                if len(self.held) - 1 >= t['query_budget']:
                    return dict(error='query budget exhausted')
                self.held.append(v)
            return dict(result=f'hidden scene from {v}', _images=[self.view(v)])
        if a['tool'] == 'render':
            v = a.get('view')
            if v not in t['public_cameras']:
                return dict(error=f'unknown view {v}')
            objs, problems = core.build_objects(t, a.get('placements'))
            if problems:
                return dict(error='; '.join(problems))
            return dict(result=f'your candidate from {v}', _images=[raycast.png(raycast.render(objs, t['room'], t['public_cameras'][v]))])
        if a['tool'] == 'compare':
            objs, problems = core.build_objects(t, a.get('placements'))
            if problems:
                return dict(error='; '.join(problems))
            gobjs, _ = core.build_objects(t, self.priv['gold'])
            out = {}
            for n in self.held:
                c = raycast.compare(raycast.render(gobjs, t['room'], t['public_cameras'][n], 2), raycast.render(objs, t['room'], t['public_cameras'][n], 2))
                out[n] = dict(mean_object_iou=round(c['mean_object_iou'], 3))
            return out
        return dict(error=f'Unknown tool {a["tool"]}')

    def final(self):
        if not self.submission:
            return dict(success=False, reason='no_submission')
        return scenes.score_t3(self.task, self.priv, self.submission.get('placements'), self.held)


TURNS_T = {'L1': 6, 'L2': 8, 'L3': 10}


def load_images(render_root, instance):
    d = os.path.join(render_root or '', instance)
    if not render_root or not os.path.isdir(d):
        return {}
    return {f[:-4]: open(os.path.join(d, f), 'rb').read() for f in os.listdir(d) if f.endswith('.png')}


def make_task_env(exp, seed, level, oneshot=False, render_root=None):
    if exp == 'T1':
        task, priv = scenes.make_t1(seed, level)
        return T1Env(task, priv, 1 if oneshot else TURNS_T[level], oneshot)
    if exp == 'T2':
        task, priv = scenes.make_t2(seed, level)
        return T2Env(task, priv, TURNS_T[level])
    if exp == 'T3':
        task, priv = scenes.make_t3(seed, level)
        imgs = {k: v for k, v in load_images(render_root, task['instance']).items() if k in task['public_cameras']}
        return T3Env(task, priv, TURNS_T[level] + task['query_budget'], images=imgs)
    raise ValueError(exp)
