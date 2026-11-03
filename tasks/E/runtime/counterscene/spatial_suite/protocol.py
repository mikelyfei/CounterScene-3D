"""Model-facing protocol for experiments A-E: English prompts, tools, episode loop, scoring.

An episode is a sequence of JSON actions {"tool": ..., ...}. Every environment exposes
`prompt()` (text + images the model sees), `step(action)` and `final()`. Private data
(gold, hidden cameras, planted faults, conflict certificates) never enters observations.
"""
import base64
import copy
import json
from ..errors import ArtifactExecutionError

from . import constraint_graph as cg, core, grounding, kinematics

FRAME = ('Coordinate frame: x points right, y points away from the default viewer ("back"), z points up; units are metres. '
         'Every object is an axis-aligned box or, where its entry says so, made of axis-aligned boxes. "position" is [centre_x, centre_y, bottom_z]. "yaw" must be 0, 90, 180 or 270 '
         '(degrees, counter-clockwise seen from above); yaw 90/270 swaps the box footprint. The floor is z = 0 and the room spans '
         '[0, W] x [0, D] x [0, H].')
PHYSICS = ('Physical rules (always checked): every object stays inside the room; no two objects interpenetrate by more than 0.01 m; '
           'every object rests on the floor (bottom_z = 0) or on top of another object (bottom_z equals that object\'s top within 0.01 m '
           'and its footprint centre lies over that object); catalogue sizes cannot be changed.')
ACTION_RULE = 'Reply with exactly one JSON object per turn and nothing else. Only the first JSON object of a reply is executed; each observation tells you how many actions remain.'


def relation_glossary(types):
    return '\n'.join(f'- {t}: {core.RELATIONS[t][2]}' for t in types)


def constraint_lines(cons, with_priority=False):
    out = []
    for c in cons:
        extra = f' [{c["priority"]}]' if with_priority else ''
        rng = f', range={c["range"]}' if 'range' in c else ''
        out.append(f'- {c["id"]}{extra}: {c["type"]}({", ".join(c["args"])}{rng})  # {core.describe(c)}')
    return '\n'.join(out)


ROUND_ITEMS = {'soccer_white': 'a soccer ball (sphere)', 'rugby_brown': 'a rugby ball (prolate spheroid lying on its side, long axis along the box x at yaw 0)'}


def shape_note(o):
    """Exact scored geometry of objects that are not a single solid box."""
    if o.get('composite') == 'bookcase':
        parts = core.composite_parts('bookcase', o['size'])
        boards = [round(off[2], 4) for name, off, _ in parts if name.startswith('board')]
        return (f'; open bookcase scored as its parts, not its envelope: two side panels {core.SHELF_T} m thick, a back panel {core.BACK_T} m '
                f'thick at the back (+y at yaw 0), and 4 shelf boards {core.SHELF_T} m thick with bottoms at {boards} m above the bookcase bottom '
                f'(board 0 = bottom board, boards 1-2 = middle shelves); items rest on a board when their bottom equals that board\'s top')
    if o['id'] in ROUND_ITEMS:
        return f'; {ROUND_ITEMS[o["id"]]}, scored by its axis-aligned bounding box'
    return ''


def objects_block(task):
    rows = []
    for o in task['objects']:
        if o.get('size_range'):
            rows.append(f'- {o["id"]}: size between {o["size_range"][0]} and {o["size_range"][1]} (nominal {o["size"]}), colour {o["color"]}' + shape_note(o))
        else:
            rows.append(f'- {o["id"]}: size {o["size"]}' + (f', colour {o["color"]}' if 'color' in o else '') + shape_note(o))
    return '\n'.join(rows)


def camera_block(task):
    """Reference camera for image-plane relations (occludes, appears_left_of, appears_larger_than)."""
    if not task.get('camera'):
        return ''
    return ('Reference camera for the image-plane relations (pinhole; looks from eye to target, up = +z, vertical fov in degrees, '
            f'image width x height): {json.dumps(task["camera"])}\n')


class Env:
    experiment = '?'

    def __init__(self, task, priv, max_turns, oneshot=False):
        self.task, self.priv, self.max_turns, self.oneshot = task, priv, max_turns, oneshot
        self.turn, self.done, self.submission, self.log = 0, False, None, []

    def _budget(self):
        return f'Turn budget: {self.max_turns} actions in total including submit. ' + ('This is a ONE-SHOT run: only "submit" is available.' if self.oneshot else '')

    def step(self, action):
        if self.done:
            return dict(error='Episode has ended; actions are not accepted.'), []
        self.turn += 1
        if not isinstance(action, dict) or 'tool' not in action:
            obs = dict(error='Invalid action: reply with one JSON object containing "tool".')
        elif action['tool'] == 'submit':
            self.done, self.submission = True, action
            obs = dict(result='submitted')
        elif self.oneshot:
            obs = dict(error='Only submit is available in this run.')
        else:
            try:
                obs = self.tool(action)
            except ArtifactExecutionError:
                raise
            except Exception as error:      # tool errors are observations, never crashes
                obs = dict(error=f'{type(error).__name__}: {error}'[:400])
        images = obs.pop('_images', []) if isinstance(obs, dict) else []
        left = self.max_turns - self.turn
        if left <= 0 and not self.done:
            self.done = True
            obs['notice'] = 'Turn budget exhausted; episode ended without a submission.'
        elif not self.done:
            obs['turns_left'] = left
            if left == 1:
                obs['notice'] = 'Only one action left: it must be "submit" or the episode ends with no answer.'
        self.log.append(dict(turn=self.turn, action=action, observation=obs, n_images=len(images)))
        return obs, images


class ConstraintEnv(Env):
    """Experiment A."""
    experiment = 'A'

    def prompt(self):
        t = self.task
        text = f"""TASK (Experiment A, spatial constraint graph, instance {t['instance']})
Place every object so that ALL constraints below hold at the same time.

{FRAME}
Room size W x D x H = {t['room']}.
{PHYSICS}
Scene placement policy (additional mandatory rules): {json.dumps(t.get('scene_policy',{}))}

Objects (sizes are [x, y, z] at yaw 0):
{objects_block(t)}

Relation definitions:
{relation_glossary(t['relation_types'])}
{camera_block(t)}
Constraints ({len(t['constraints'])}):
{constraint_lines(t['constraints'])}

Tools:
- {{"tool": "check", "placements": {{...}}}}: the verifier evaluates a candidate and returns every constraint with satisfied/value plus physical issues.
- {{"tool": "submit", "placements": {{...}}}}: final answer; ends the episode.
"placements" maps every object id to {{"position": [x, y, bottom_z], "yaw": 0}}.

Success criterion: the submitted placements satisfy every constraint and every physical rule. Partial credit (fraction of constraints satisfied) is reported but success is binary.
{self._budget()}
{ACTION_RULE}"""
        return text, []

    def tool(self, a):
        if a['tool'] == 'check':
            return core.check_scene(self.task, a.get('placements'))
        return dict(error=f'Unknown tool {a["tool"]}')

    def final(self):
        if not self.submission:
            return dict(success=False, reason='no_submission', satisfied=0, total=len(self.task['constraints']))
        return cg.score_a(self.task, self.submission.get('placements'))


class RepairEnv(Env):
    """Experiment B."""
    experiment = 'B'

    def prompt(self):
        t = self.task
        text = f"""TASK (Experiment B, conflicting constraints and repair, instance {t['instance']})
A previous agent produced the scene below, but it has problems. Repair it.

{FRAME}
Room size W x D x H = {t['room']}.
{PHYSICS}
Scene placement policy (additional mandatory rules): {json.dumps(t.get('scene_policy',{}))}

Objects:
{objects_block(t)}
Locked objects (must not move at all): {t['locked'] or 'none'}

Relation definitions:
{relation_glossary(t['relation_types'])}
{camera_block(t)}
Constraints. [hard] constraints must all hold. [soft] constraints should hold when possible; some soft constraints may be impossible to satisfy together with the hard ones. Any soft constraint that is impossible must be listed in "infeasible" with a one-line reason.
{constraint_lines(t['constraints'], True)}

Current scene (placements):
{json.dumps(t['initial'])}

Verifier report for the current scene:
{json.dumps(compact_report(t['initial_report']))}

Tools:
- {{"tool": "check", "placements": {{...}}}}: verifier report for a candidate.
- {{"tool": "submit", "placements": {{...}}, "infeasible": [{{"id": "cNN", "reason": "..."}}]}}: final answer.

Success criterion: all hard constraints and physical rules hold, locked objects are unchanged, every feasible soft constraint holds, and "infeasible" lists exactly the impossible soft constraints.
Also reported (secondary): which originally satisfied constraints you preserved, how many objects you moved and how far, compared with a known feasible repair. Keep correct parts of the scene unchanged where you can.
{self._budget()}
{ACTION_RULE}"""
        return text, []

    def tool(self, a):
        if a['tool'] == 'check':
            return compact_report(core.check_scene(self.task, a.get('placements')))
        return dict(error=f'Unknown tool {a["tool"]}')

    def final(self):
        if not self.submission:
            return dict(success=False, reason='no_submission')
        inf = [x.get('id') if isinstance(x, dict) else x for x in (self.submission.get('infeasible') or [])]
        out = cg.score_b(self.task, self.priv, self.submission.get('placements'), inf)
        # regressions along the trajectory: satisfied-at-start hard constraints broken in any checked candidate
        start = {r['id'] for r in self.task['initial_report']['constraints'] if r['satisfied']}
        broke = set()
        for row in self.log:
            for r in (row['observation'].get('constraints') or []) if isinstance(row['observation'], dict) else []:
                if r['id'] in start and not r['satisfied']:
                    broke.add(r['id'])
        out['trajectory_regressions'] = sorted(broke)
        return out


class FeedbackEnv(Env):
    """Experiment E: the check tool is deliberately unreliable; measure is exact."""
    experiment = 'E'

    def prompt(self):
        t = self.task
        text = f"""TASK (Experiment E, verifier feedback under uncertainty, instance {t['instance']})
Repair the scene so that every constraint and physical rule holds. The official grade applies the definitions in this prompt exactly.
WARNING: the "check" tool is an unvalidated development checker. Its report may be incomplete (constraints or physical checks missing), its numbers may be noisy, and some of its verdicts may be wrong. Do not assume it is correct. "measure" returns exact box extents and can be used to verify anything yourself.

{FRAME}
Room size W x D x H = {t['room']}.
{PHYSICS}
Scene placement policy (additional mandatory rules): {json.dumps(t.get('scene_policy',{}))}

Objects:
{objects_block(t)}

Relation definitions:
{relation_glossary(t['relation_types'])}
{camera_block(t)}
Constraints ({len(t['constraints'])}):
{constraint_lines(t['constraints'])}

Current scene (placements):
{json.dumps(t['initial'])}

Tools:
- {{"tool": "check", "placements": {{...}}}}: development checker report (may be unreliable).
- {{"tool": "measure", "placements": {{...}}}}: exact axis-aligned extents [min_x, max_x, min_y, max_y, min_z, max_z] and centres of every object.
- {{"tool": "submit", "placements": {{...}}, "checker_issues": [{{"id": "cNN", "issue": "false_pass|false_fail|missing|wrong_rule", "evidence": "..."}}, {{"id": "no_interpenetration", "objects": ["<id>", "<id>"], "issue": "missing", "evidence": "..."}}]}}
Checker issues to report (verdict problems only): false_pass (reports satisfied but the constraint is violated), false_fail (reports violated but it holds), missing (a constraint, or an interpenetrating object pair, never appears in the report; give the pair in "objects"), wrong_rule (a relation type is evaluated with a rule different from its definition). Small numeric differences in reported values are expected noise and must NOT be reported.
Interpenetrations may involve two objects or several objects that all overlap one another; objects that are not involved in any problem should stay where they are.

Success criterion: the submitted placements satisfy every constraint and physical rule under the definitions above. Also scored: each checker problem you correctly report in "checker_issues" (precision and recall against the problems that were actually present).
{self._budget()}
{ACTION_RULE}"""
        return text, []

    def tool(self, a):
        if a['tool'] == 'check':
            return compact_report(cg.noisy_check(self.task, self.priv['faults'], a.get('placements'), self.turn))
        if a['tool'] == 'measure':
            objs, problems = core.build_objects(self.task, a.get('placements'))
            if problems:
                return dict(error='; '.join(problems))
            return {i: dict(extent=[round(v, 4) for v in core.aabb(o)], centre=[round(v, 4) for v in core.center(o)]) for i, o in objs.items()}
        return dict(error=f'Unknown tool {a["tool"]}')

    def final(self):
        if not self.submission:
            return dict(success=False, reason='no_submission')
        issues = self.submission.get('checker_issues') or []
        return cg.score_e(self.task, self.priv, self.submission.get('placements'), issues)


class GroundingEnv(Env):
    """Experiment C."""
    experiment = 'C'

    def __init__(self, *a, images=None, **k):
        super().__init__(*a, **k)
        self.held = ['reference']
        self.images = images or {}

    def reference_images(self,view):
        diagram=grounding.view_png(self.task,self.priv['gold'],view)
        return [diagram]+([self.images[view]] if view in self.images else [])

    def prompt(self):
        t = self.task
        cams = {n: t['public_cameras'][n] for n in ['reference']}
        size_rule = ('Sizes are not exact: each object\'s true size lies inside its range; include "size": [x, y, z] in each placement.'
                     if any(o.get('size_range') for o in t['objects']) else 'Sizes are exact and fixed.')
        text = f"""TASK (Experiment C, image-to-3D grounding, instance {t['instance']})
The first attached image is the exact scored box-envelope diagram from the "reference" camera. Reconstruct the 3D scene. If a second image is attached, it is the furniture appearance of the SAME scene and camera at twice the diagram resolution; it is context, not a different target. Use the diagram for exact scoring silhouettes. The furniture models are fitted inside these scored boxes.
Your reconstruction will be rendered from OTHER cameras you never see and compared with the hidden scene there, so recover real 3D positions, not just a picture that matches this one view.

{FRAME}
Room size W x D x H = {t['room']}. The floor has a 0.5 m checkerboard aligned with the axes starting at (0, 0). Outside the room is dark grey.
{PHYSICS}
Scene placement policy (additional mandatory rules): {json.dumps(t.get('scene_policy',{}))}
Each object is a single-colour box unless its entry below says otherwise (the bookcase is scored as its panels and boards; balls are scored by their bounding boxes). {size_rule}
Objects:
{objects_block(t)}

Camera model: pinhole, looks from "eye" to "target", up = +z, vertical field of view fov_deg, image width x height pixels.
Reference camera: {json.dumps(cams['reference'])}
Other public cameras you may request (budget {t['query_budget']} extra views): {json.dumps({n: t['public_cameras'][n] for n in t['queryable_views']})}

Tools:
- {{"tool": "query_view", "view": "<name>"}}: returns the hidden scene rendered from that public camera (counts against the query budget).
- {{"tool": "render", "placements": {{...}}, "view": "<public camera name>"}}: renders YOUR candidate from any public camera.
- {{"tool": "compare", "placements": {{...}}}}: mean per-object silhouette IoU (one number per view) between your candidate and the hidden scene, only for views you already hold.
- {{"tool": "submit", "placements": {{...}}}}
"placements" maps each object id to {{"position": [x, y, bottom_z], "yaw": 0{', "size": [x, y, z]' if any(o.get('size_range') for o in t['objects']) else ''}}}.

Success criterion: physical rules hold and the mean per-object silhouette IoU over the hidden verification cameras is at least {grounding.PASS_IOU}. Also reported: depth error, 3D centre error per object, and the gap between IoU on views you saw and hidden views.
{self._budget()}
{ACTION_RULE}"""
        return text, self.reference_images('reference')

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
            return dict(result=f'hidden scene from {v}', _images=self.reference_images(v))
        if a['tool'] == 'render':
            v = a.get('view')
            if v not in t['public_cameras']:
                return dict(error=f'unknown view {v}')
            return dict(result=f'your candidate from {v}', _images=[grounding.view_png(t, a.get('placements'), v)])
        if a['tool'] == 'compare':
            return grounding.held_view_iou(t, self.priv, a.get('placements'), self.held)
        return dict(error=f'Unknown tool {a["tool"]}')

    def final(self):
        if not self.submission:
            return dict(success=False, reason='no_submission')
        return grounding.score_c(self.task, self.priv, self.submission.get('placements'), self.held)


class MechanismEnv(Env):
    """Experiment D."""
    experiment = 'D'

    def prompt(self):
        t = self.task
        free = []
        free.append(f'link lengths within {t["link_length_range"]}' if t['free_len'] else f'link lengths fixed at {t["link_lengths"]}')
        free.append(f'base position anywhere in the rectangle {t["base_region"]} (x/y corners)' if t['free_base'] else f'base fixed at {t["base"]}')
        balls = t['balls']
        seq = (f'pick up ball_1, set it down on top of {balls[0]["support"]} (its intermediate drop point), move to ball_2, pick it up, place it at its final point on the floor, then return to q_home'
               if len(balls) > 1 else 'move from q_home to the ball, pick it up, carry it' + (' through every transfer checkpoint in order' if any(w['role'] == 'transfer' for w in t['waypoints']) else '') + ' and place it at its destination')
        text = f"""TASK (Experiment D, kinematic mechanism, instance {t['instance']})
Starting from the declared home pose, {seq}. Design and program a 3-joint arm with collision-free continuous motion. Obstacles are grounded cabinets or columns; none float.

{FRAME}
Mechanism template: a vertical post of height {t['post_height']} m stands on the floor at the base (x, y). Joint 1 (q1) rotates everything about +z at the base; q1 = 0 points the arm along +x, positive is counter-clockwise. Joint 2 (q2) at the post top pitches link 1: q2 = 0 is horizontal, +90 is straight up. Joint 3 (q3) pitches link 2 relative to link 1 (q3 = 0 continues straight; negative bends downward). The tip is the far end of link 2. Angles in degrees.
Links are capsules of radius {t['link_radius']} m. Motion between consecutive keyframes is linear interpolation in joint space, and the whole motion is checked continuously (not only at keyframes): each moving link must stay at least {t['clearance_m']} m (beyond its radius) away from every obstacle, from the floor, and link 2 from the post.
Design freedom: {'; '.join(free)}. Declared joint limits must lie inside the hardware limits {t['hardware_limits_deg']} and every keyframe must lie inside your declared limits.
Home pose q_home (degrees): {json.dumps(t['home'])}. keyframes[0] must equal q_home exactly; the arm starts there, away from the balls.{' The last keyframe must equal q_home again.' if t.get('return_home') else ''}
Obstacles (axis-aligned boxes [min_x, max_x, min_y, max_y, min_z, max_z]): {json.dumps(t['obstacles']) if t['obstacles'] else 'none'}
Obstacle sizes [x, y, height] in metres (cabinets all differ in height): {json.dumps({o['id']: [round(o['box'][1] - o['box'][0], 4), round(o['box'][3] - o['box'][2], 4), o['box'][5]] for o in t['obstacles']})}
Balls (centre positions): {json.dumps(balls)}. Each ball starts on the floor; "support" names what it must rest on at its destination (the floor, or the top face of that cabinet: the ball's centre is then the cabinet top + its radius, over the top face).
Grasp model: one gripper; an attached ball's centre is the gripper tip (ideal kinematic grasp, no force simulation). Each ball must be grasped once at its pickup waypoint and released once at its place waypoint, and you may hold only one ball at a time. A ball resting on the floor (not yet picked, or already placed) is a fixed sphere obstacle: arm links and any carried ball must keep {t['clearance_m']} m clearance from it, except during the segment that arrives at that ball's grasp keyframe and the segment that leaves its release keyframe. A carried ball must keep the same clearance from obstacles and the post and stay above the floor; at release its centre must be within 0.005 m of its resting height on its support, and a ball set down on a cabinet must be over that cabinet's top face. The carried ball may touch its support cabinet only in the segment that sets it down. Transfer checkpoints, if any, are virtual guides.
Waypoints (tip must be within {t['reach_tolerance_m']} m, in this order): {json.dumps(t['waypoints'])}

Submission format:
{{"mechanism": {{"base": [x, y], "link_lengths": [l1, l2], "joint_limits_deg": [[lo1, hi1], [lo2, hi2], [lo3, hi3]]}},
 "keyframes": [[q1, q2, q3], ...] (at most {kinematics.MAX_KEYFRAMES}), "waypoint_keyframes": [index of the keyframe that reaches each listed waypoint (non-decreasing)],
 "events": [{{"keyframe": <index>, "action": "grasp" | "release", "ball": "<ball id>"}}, ...] (strictly increasing keyframes)}}
(base / link_lengths are ignored when fixed.)

Tools:
- {{"tool": "simulate", ...submission fields...}}: runs the continuous-motion verifier and returns tip errors, per-segment minimum clearance with the closest pair, limit problems, and grasp/carry/ball-clearance results.
- {{"tool": "forward", "mechanism": {{...}}, "q": [q1, q2, q3]}}: joint positions [base, post_top, elbow, tip] for one pose.
- {{"tool": "submit", ...submission fields...}}

Success criterion: the sequence above is completed with valid events; every waypoint is reached in order; limits, full arm clearance, carried-ball clearance and clearance from resting balls all pass{'; the motion ends at q_home' if t.get('return_home') else ''}. Cabinets and columns are fixed and cannot be deleted or moved.
{self._budget()}
{ACTION_RULE}"""
        return text, []

    def tool(self, a):
        if a['tool'] == 'simulate':
            rep = kinematics.check_motion(self.task, a)
            rep.pop('mechanism', None)
            return rep
        if a['tool'] == 'forward':
            m = a['mechanism']
            mech = dict(base=m.get('base') if self.task['free_base'] else self.task['base'],
                        link_lengths=m.get('link_lengths') if self.task['free_len'] else self.task['link_lengths'])
            return dict(points=[[round(v, 4) for v in p] for p in kinematics.fk(mech, [float(v) for v in a['q']])])
        return dict(error=f'Unknown tool {a["tool"]}')

    def final(self):
        if not self.submission:
            return dict(success=False, reason='no_submission')
        return kinematics.score_d(self.task, self.submission)


def parse_action(text):
    """First JSON object in a reply. Extra objects are ignored (and flagged) instead of voiding the turn."""
    text = (text or '').strip()
    start = text.find('{')
    if start < 0:
        return None, False
    try:
        obj, end = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None, False
    return obj, bool(text[start + end:].strip())


def compact_report(rep):
    if not rep.get('valid_output', True):
        return dict(valid_output=False, problems=rep.get('output_problems'))
    return dict(all_satisfied=rep.get('all_satisfied'), physical=rep.get('physical'),
                constraints=[dict(id=r['id'], satisfied=r['satisfied'], value=r['value']) for r in rep['constraints']])


TURNS = {'L1': 6, 'L2': 8, 'L3': 10}


def make_env(exp, seed, level, oneshot=False, render_root=None):
    """level: tier name, or for A an integer constraint count (scaling sweep)."""
    if exp in ('T1', 'T2', 'T3'):
        from .tasks import make_task_env
        env = make_task_env(exp, seed, level, oneshot, render_root=render_root)
        env._seed, env._level = seed, level
        return env
    if exp == 'A':
        k = level if isinstance(level, int) else cg.TIERS[level]
        task, gold = cg.make_a(seed, k)
        return ConstraintEnv(task, dict(gold=gold), 1 if oneshot else 4 + k // 3, oneshot)
    if exp == 'B':
        task, priv = cg.make_b(seed, level)
        return RepairEnv(task, priv, 1 if oneshot else TURNS[level], oneshot)
    if exp == 'C':
        task, priv = grounding.make_c(seed, level)
        images={}
        if render_root:
            from pathlib import Path
            folder=Path(render_root)/task['instance']
            images={name:(folder/('gold_assets_'+name+'.png')).read_bytes() for name in task['public_cameras'] if (folder/('gold_assets_'+name+'.png')).exists()}
        return GroundingEnv(task, priv, 1 if oneshot else TURNS[level] + task['query_budget'], oneshot, images=images)
    if exp == 'D':
        task, priv = kinematics.make_d(seed, level)
        return MechanismEnv(task, priv, 1 if oneshot else TURNS[level], oneshot)
    if exp == 'E':
        task, priv = cg.make_e(seed, level)
        return FeedbackEnv(task, priv, 1 if oneshot else TURNS[level], oneshot)
    raise ValueError(exp)


def public_view(env):
    """Everything the model can see at the start, for the review bundle."""
    text, images = env.prompt()
    return dict(prompt=text, images_png_base64=[base64.b64encode(i).decode() for i in images])


def private_view(env):
    p = copy.deepcopy(env.priv)
    return p
