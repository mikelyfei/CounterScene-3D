"""Carry measured robot and released-object state across distinct operations."""
import copy,math
from .native_mechanism import executed_ball_timeline


def carried_motion_state(task,execution,previous_resting=()):
    payload=execution['payload'];keys=payload['keyframes']
    if not keys or len(keys[-1])!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in keys[-1]):raise ValueError('Actual terminal physical joints required')
    timeline=executed_ball_timeline(task,payload)
    if not timeline:raise ValueError('Actual execution timeline required')
    if any(position is None for position in timeline[-1].values()):raise ValueError('Release the held object before continuing to the next operation')
    resting={row['id']:copy.deepcopy(row) for row in previous_resting}
    for ball in task['balls']:
        if ball['id'] in resting:raise ValueError('A new stage cannot replace an existing released object')
        resting[ball['id']]=dict(id=ball['id'],position=copy.deepcopy(timeline[-1][ball['id']]),radius_m=ball['radius_m'])
    return dict(physical_q=copy.deepcopy(keys[-1]),mechanism=copy.deepcopy(payload['mechanism']),resting_objects=list(resting.values()),holding=None,scope='Actual executed terminal state; no home reset or target-position substitution')


def apply_motion_state(task,state):
    task=copy.deepcopy(task)
    if {ball['id'] for ball in task['balls']}&{obj['id'] for obj in state['resting_objects']}:raise ValueError('New active payload must have a distinct ID')
    task['home']=copy.deepcopy(state['physical_q'])
    task['base']=copy.deepcopy(state['mechanism']['base']);task['link_lengths']=copy.deepcopy(state['mechanism']['link_lengths'])
    task['free_base']=False;task['free_len']=False
    task['resting_objects']=copy.deepcopy(state['resting_objects'])
    return task
