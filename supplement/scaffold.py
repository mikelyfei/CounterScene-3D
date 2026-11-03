import copy,json,statistics

def obj(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)

def vector(length):
    return dict(type='array',items=dict(type='number'),minItems=length,maxItems=length)

def candidate_schema(task,family):
    if family=='D':
        mechanism=obj(dict(base=vector(2),link_lengths=vector(2),joint_limits_deg=dict(type='array',items=vector(2),minItems=3,maxItems=3)))
        return obj(dict(mechanism=mechanism,keyframes=dict(type='array',items=vector(3),minItems=2,maxItems=16),
            waypoint_keyframes=dict(type='array',items=dict(type='integer',minimum=0,maximum=15),minItems=len(task['waypoints']),maxItems=len(task['waypoints'])),
            events=dict(type='array',items=obj(dict(keyframe=dict(type='integer',minimum=0,maximum=15),action=dict(type='string',enum=['grasp','release']),ball=dict(type='string',enum=[b['id'] for b in task['balls']]))),minItems=2,maxItems=2*len(task['balls']))))
    poses={}
    for item in task['objects']:
        fields=dict(position=vector(3),yaw=dict(type='integer',enum=[0,90,180,270]))
        if family=='C':fields['height_m']=dict(type='number',minimum=item['size_range'][0][2],maximum=item['size_range'][1][2])
        poses[item['id']]=obj(fields)
    fields=dict(placements=obj(poses))
    if family=='B':fields['infeasible']=dict(type='array',items=obj(dict(id=dict(type='string'),reason=dict(type='string'))))
    if family=='E':fields['checker_issues']=dict(type='array',items=obj(dict(id=dict(type='string'),issue=dict(type='string',enum=['false_pass','false_fail','missing','wrong_rule']),evidence=dict(type='string'))))
    return obj(fields)

def candidate_action(candidate,task,family,tool):
    """Serialize model numbers unchanged; scale expansion uses public dimensions."""
    payload=copy.deepcopy(candidate if family=='D' else candidate['placements'])
    if family=='C':
        sizes={o['id']:o['size'] for o in task['objects']}
        for oid,pose in payload.items():
            scale=pose.pop('height_m')/sizes[oid][2]
            pose['size']=[round(v*scale,6) for v in sizes[oid]]
    action=dict(tool=tool,source='def build(public_params):\n    return '+repr(payload))
    for key in ('infeasible','checker_issues'):
        if key in candidate:action[key]=copy.deepcopy(candidate[key])
    return action

def choose(rows):
    # Ties use the first proposal; hidden measurements never enter selection.
    return max(range(len(rows)),key=lambda i:rows[i]['public_reward'])

def compact_prompt(text):
    """Drop irrelevant atlas mesh metadata, retaining public identity/pose legend."""
    for marker in ('Atlas staging is not the target: ','Atlas identity/pose legend='):
        if marker not in text:continue
        start=text.index(marker)+len(marker)
        atlas,length=json.JSONDecoder().raw_decode(text[start:])
        text=text[:start]+json.dumps(dict(layout=atlas['layout'],note='Catalogue staging only, NOT target placements or sizes.'))+text[start+length:]
    return text

def pool_metrics(rows):
    selected=rows[choose(rows)]
    return dict(first_public=rows[0]['public_reward'],selected_public=selected['public_reward'],
        random_public=statistics.mean(r['public_reward'] for r in rows),first_hidden=rows[0]['hidden_reward'],
        selected_hidden=selected['hidden_reward'],random_hidden=statistics.mean(r['hidden_reward'] for r in rows),
        valid_candidates=sum(r['native_schema_valid'] for r in rows),candidates=len(rows),
        selected_index=choose(rows),hidden_gain=selected['hidden_reward']-rows[0]['hidden_reward'],
        proxy_gap=selected['public_reward']-selected['hidden_reward'])
