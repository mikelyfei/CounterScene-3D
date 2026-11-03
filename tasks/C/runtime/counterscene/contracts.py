"""Strict output contract and immutable asset reconstruction."""
import copy
import hashlib
import json
import math


class ContractError(ValueError):
    pass


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def keys(value, expected, where):
    if not isinstance(value,dict) or set(value)!=set(expected):
        raise ContractError(f'{where}: expected exactly {sorted(expected)}')


def number(x, limit=100):
    if type(x) not in (int,float) or not math.isfinite(x) or abs(x)>limit:
        raise ContractError('expected finite bounded number (booleans forbidden)')
    return round(float(x),8)


def vector(value,n):
    if not isinstance(value,list) or len(value)!=n:
        raise ContractError(f'expected {n}-element list')
    return [number(x) for x in value]


def validate(task,scene,phase='initial'):
    keys(scene,{'schema_version','task_id','instance_id','phase','placements','joints','path_xy_m'},'SceneSpec')
    if scene['schema_version']!='1.0' or scene['task_id']!=task['id'] or scene['instance_id']!=task['instance_id'] or scene['phase']!=phase:
        raise ContractError('schema/task/instance/phase mismatch')
    placements=scene['placements']
    if not isinstance(placements,list) or len(placements)!=len(task['catalog']):
        raise ContractError('all editable objects must occur exactly once')
    ids=[]
    out=copy.deepcopy(scene)
    for p in out['placements']:
        keys(p,{'id','center_m','yaw_deg'},'placement')
        if not isinstance(p['id'],str):
            raise ContractError('object id must be a string')
        ids.append(p['id'])
        p['center_m']=vector(p['center_m'],3)
        p['yaw_deg']=number(p['yaw_deg'],360)
    if len(set(ids))!=len(ids) or set(ids)!=set(task['catalog']):
        raise ContractError('unknown, duplicate or missing object')
    out['placements'].sort(key=lambda p:p['id'])
    if not isinstance(scene['joints'],list):
        raise ContractError('joints must be a list')
    if 'door' in task:
        if len(scene['joints'])!=1:
            raise ContractError('one cabinet_door joint required')
        joint=scene['joints'][0]
        keys(joint,{'id','hinge_side'},'joint')
        if joint['id']!='cabinet_door' or joint['hinge_side'] not in ('left','right'):
            raise ContractError('invalid hinge')
    elif scene['joints']!=[]:
        raise ContractError('this task has no joints')
    path=scene['path_xy_m']
    if not isinstance(path,list) or ('start' in task and not 2<=len(path)<=12) or ('start' not in task and path):
        raise ContractError('path must contain 2-12 points for navigation; empty otherwise')
    out['path_xy_m']=[vector(p,2) for p in path]
    return out


def reconstruct(task,scene,phase='initial'):
    scene=validate(task,scene,phase)
    objects=[]
    for p in scene['placements']:
        objects.append(dict(id=p['id'],**copy.deepcopy(task['catalog'][p['id']]),
                            center_m=p['center_m'],yaw_deg=p['yaw_deg']))
    objects.extend(copy.deepcopy(task.get('fixed',[])))
    if phase=='post_event':
        objects.append(copy.deepcopy(task['event']['object']))
    return scene,objects
