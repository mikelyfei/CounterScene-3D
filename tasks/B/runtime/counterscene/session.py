"""Budgeted agent tools, immutable artifacts and an idempotent event state."""
import copy
import hashlib
import json
import time
import os
import tempfile
from datetime import datetime,timezone
from pathlib import Path
from .contracts import validate,digest
from .program import execute,ProgramError,literal_program
from .scoring import evaluate
from .render import render,views
from .tasks import public


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)
    # Background synchronization can briefly hold the destination open on Windows.
    # Retry only the local atomic rename, never an inference request.
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
             prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name);stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    for attempt in range(10):
        try:temporary.replace(path);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(min(.02*2**attempt,.2))


class Session:
    def __init__(self,task,start,gold,directory,model_id='scripted:gold'):
        self.task,self.start,self.gold=task,start,gold
        self.dir=Path(directory);self.dir.mkdir(parents=True,exist_ok=True)
        self.model_id=model_id
        self.state_path=self.dir/'state.json'
        if self.state_path.exists():
            self.state=json.loads(self.state_path.read_text(encoding='utf-8'))
            if self.state['task_hash']!=digest(task) or self.state['model_id']!=model_id:raise ValueError('resume task/model mismatch')
        else:
            self.state=dict(task_hash=digest(task),model_id=model_id,phase='initial',steps=0,
                            artifacts={},checkpoint=None,submitted=None,previous_event_hash=None)
            self.save()

    def save(self):write_json(self.state_path,self.state)

    def event(self,action,observation,latency_ms=0,**extra):
        event=dict(run_id=self.dir.name,task_id=self.task['id'],instance_id=self.task['instance_id'],
                   model_id=self.model_id,phase=self.state['phase'],event_index=self.state['steps'],
                   action=action,observation=observation,observation_hash=digest(observation),
                   status=observation.get('status','ok') if isinstance(observation,dict) else 'ok',
                   latency_ms=latency_ms,tokens=None,cost_usd=None,
                   timestamp=datetime.now(timezone.utc).isoformat(),
                   previous_event_hash=self.state['previous_event_hash'],**extra)
        event['event_hash']=digest(event)
        with (self.dir/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
        self.state['previous_event_hash']=event['event_hash'];self.save()

    def params(self):return public(self.task,self.state['phase'])

    def initial_observation(self):
        ref=copy.deepcopy(self.start if self.task['id']=='t3' else self.gold);ref['phase']='initial'
        current=copy.deepcopy(self.start);current['phase']='initial'
        a=render(self.task,ref,self.dir/'reference'/'front','front',reference=True)
        b=render(self.task,current,self.dir/'initial'/'iso')
        return dict(public_params=self.params(),start_scene=current,images=[str(a),str(b)],
                    image_labels=['target front view','current isometric view'],remaining_steps=self.task['max_steps'])

    def artifact(self,aid,current=True):
        if aid not in self.state['artifacts']:raise ValueError('unknown artifact_id')
        meta=self.state['artifacts'][aid]
        if current and meta['phase']!=self.state['phase']:raise ValueError('stale artifact: rebuild for current phase')
        scene=json.loads((self.dir/'artifacts'/aid/'scene.json').read_text(encoding='utf-8'))
        if digest(scene)!=meta['scene_hash']:raise ValueError('artifact hash mismatch')
        return scene

    def act(self,action):
        if self.state['submitted']:raise ValueError('session already submitted')
        if self.state['steps']>=self.task['max_steps']:raise ValueError('tool budget exhausted')
        self.state['steps']+=1;self.save();before=time.monotonic()
        try:
            try:encoded=json.dumps(action,allow_nan=False)
            except (ValueError,TypeError):
                action={'tool':'invalid_json_value','raw':repr(action)[:65536]}
                raise ValueError('action must contain only finite, JSON-serializable values')
            if len(encoded)>65536:
                action={'tool':'oversized_action','raw':encoded[:1024]}
                raise ValueError('action exceeds 64 KiB')
            if not isinstance(action,dict) or 'tool' not in action:raise ValueError('expected JSON action with tool')
            tool=action['tool']
            signatures={'inspect_reference':{'tool','view','angle'},'build_scene':{'tool','source'},
                        'inspect_output':{'tool','artifact_id','view','angle'},'check_public':{'tool','artifact_id'},
                        'checkpoint':{'tool','artifact_id'},'submit':{'tool','artifact_id'}}
            if tool not in signatures:raise ValueError('unknown tool; choose '+', '.join(signatures))
            if set(action)-signatures[tool]:raise ValueError(f'{tool} accepts only {sorted(signatures[tool])}; remove {sorted(set(action)-signatures[tool])}')
            if tool=='inspect_reference':
                ref=copy.deepcopy(self.start if self.task['id']=='t3' else self.gold);ref['phase']='initial'
                view=action.get('view','top')
                angle=action.get('angle',0)
                if type(angle) not in (int,float) or not 0<=angle<=90:raise ValueError('angle outside 0..90')
                if self.task.get('reference_motion_disabled') and angle!=0:
                    raise ValueError('Nonzero reference angles are disabled in this observation-control condition')
                # Information-control arms: refusals never mention the private target.
                if view not in self.task.get('reference_views',('front','top','side','iso')):
                    raise ValueError('reference view unavailable in this condition; available: '+', '.join(self.task['reference_views']))
                if angle>self.task.get('reference_max_angle',90):
                    raise ValueError('non-zero reference door angles are unavailable in this condition')
                path=self.dir/'reference'/(view if angle==0 else f'{view}_{angle}')
                file=render(self.task,ref,path,view,angle=angle,reference=True)
                obs=dict(status='ok',images=[str(file)],image_labels=[f'target {view}, door {angle} degrees'])
            elif tool=='build_scene':
                source=action['source'];scene=execute(source,self.params())
                scene=validate(self.task,scene,self.state['phase'])
                # A second actual interpretation detects unexpected nondeterminism.
                if digest(scene)!=digest(validate(self.task,execute(source,self.params()),self.state['phase'])):raise ProgramError('nondeterministic program')
                code_hash=hashlib.sha256(source.encode()).hexdigest();scene_hash=digest(scene)
                aid=digest([scene_hash,code_hash])[:16];dest=self.dir/'artifacts'/aid;dest.mkdir(parents=True,exist_ok=True)
                (dest/'scene.py').write_text(source,encoding='utf-8');write_json(dest/'scene.json',scene)
                file=render(self.task,scene,dest/'iso')
                self.state['artifacts'][aid]=dict(phase=self.state['phase'],scene_hash=scene_hash,code_hash=code_hash)
                obs=dict(status='ok',artifact_id=aid,scene_hash=scene_hash,code_hash=code_hash,images=[str(file)],image_labels=['built scene'])
                if self.task.get('feedback_protocol')=='stage-aware-v4':
                    previous=self.state.get('last_build')
                    if previous==aid:obs['notice']='This is identical to your previous build. Use check_public, checkpoint when ready, or submit; rebuilding does not advance the event.'
                    self.state['last_build']=aid
            elif tool=='inspect_output':
                scene=self.artifact(action['artifact_id']);angle=action.get('angle',0)
                if type(angle) not in (int,float) or not 0<=angle<=90:raise ValueError('angle outside 0..90')
                view=action.get('view','iso')
                file=render(self.task,scene,self.dir/'artifacts'/action['artifact_id']/f'{view}_{angle}',view,angle)
                obs=dict(status='ok',images=[str(file)],image_labels=[f'output {view}, door {angle} degrees'])
            elif tool=='check_public':
                scene=self.artifact(action['artifact_id']);result=evaluate(self.task,scene,self.state['phase'])
                # Do not turn hidden target hinge truth into a public answer oracle.
                checks=[c for c in result['checks'] if c['criterion_id']!='hinge_side']
                obs=dict(status='ok',checks=checks,images=[],note='Compare hinge side against reference image. Final T3 requires checkpoint.')
                minimal=self.task.get('diagnostic_feedback')=='minimal'
                if self.task.get('feedback_protocol')=='stage-aware-v4':
                    checks=[c for c in checks if not c['criterion_id'].startswith('relation:') and c['criterion_id']!='event_acknowledged']
                    obs['checks']=checks
                    obs['physical_checks_pass']=all(c['passed'] for c in checks)
                    if self.task['id']=='t3' and self.state['phase']=='initial':
                        obs['checkpoint_ready']=obs['physical_checks_pass']
                        obs['next_stage']='Call checkpoint with this artifact_id when ready; event acknowledgement is a later-stage requirement.'
                if minimal:
                    # Pass/fail only: no distances, footprints, angles or diagnostic images.
                    obs['checks']=[dict(criterion_id=c['criterion_id'],passed=c['passed']) for c in obs['checks']]
                    obs['feedback']='minimal pass/fail diagnostics in this condition'
                motion=next((c for c in checks if c['criterion_id']=='door_sweep'),None)
                if motion and not motion['passed'] and not minimal:
                    angle=motion['measured_value']['worst_angle_deg']
                    file=render(self.task,scene,self.dir/'artifacts'/action['artifact_id']/'collision','top',angle)
                    obs['images']=[str(file)];obs['image_labels']=[f'minimum clearance at {angle} degrees']
            elif tool=='checkpoint':
                if self.task['id']!='t3':raise ValueError('checkpoint only exists for t3')
                if not self.state['checkpoint']:
                    initial=self.artifact(action['artifact_id'])
                    pre=evaluate(self.task,initial,'initial')
                    exempt={'event_acknowledged'}
                    if self.task.get('feedback_protocol')=='stage-aware-v4':exempt.add('hinge_side')
                    violations=[c['criterion_id'] for c in pre['checks'] if not c['passed'] and c['criterion_id'] not in exempt]
                    if violations:raise ValueError('checkpoint requires feasible initial scene: '+', '.join(violations))
                    write_json(self.dir/'checkpoint_score.json',pre)
                    self.state['checkpoint']=action['artifact_id'];self.state['phase']='post_event'
                scene=copy.deepcopy(self.artifact(self.state['checkpoint'],current=False));scene['phase']='post_event'
                file=render(self.task,scene,self.dir/'event'/'iso')
                obs=dict(status='event_applied',event_id=self.task['event']['id'],phase='post_event',
                         public_params=self.params(),images=[str(file)],image_labels=['updated workshop after equipment delivery'])
            elif tool=='submit':
                scene=self.artifact(action['artifact_id'])
                if self.task['id']=='t3' and not self.state['checkpoint']:raise ValueError('checkpoint required before final submit')
                self.state['submitted']=action['artifact_id']
                obs=dict(status='submitted',artifact_id=action['artifact_id'])
            if self.state['phase']=='post_event' and not self.task.get('post_event_images',True) and obs.get('images'):
                # Images remain on disk for audit, but are not shown to the agent.
                obs['withheld_images']=len(obs['images']);obs['images']=[];obs.pop('image_labels',None)
            obs['remaining_steps']=self.task['max_steps']-self.state['steps']
            self.event(action,obs,round((time.monotonic()-before)*1000,2))
            return obs
        except (ValueError,TypeError,KeyError,IndexError) as error:
            obs=dict(status='tool_error',error=str(error))
            obs['remaining_steps']=self.task['max_steps']-self.state['steps']
            self.event(action,obs,round((time.monotonic()-before)*1000,2));return obs

    def final_result(self):
        aid=self.state['submitted']
        if not aid:return dict(status='not_submitted',success=False,R_det=0,failure_labels=['BUDGET_OR_PROTOCOL'])
        result=evaluate(self.task,self.artifact(aid),self.state['phase'])
        result.update(artifact_id=aid,steps=self.state['steps'],model_id=self.model_id,phase=self.state['phase'])
        write_json(self.dir/'result.json',result)
        return result


def verify_events(path):
    previous=None;count=0
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        e=json.loads(line);hashed=e.pop('event_hash')
        if e['previous_event_hash']!=previous or digest(e)!=hashed or digest(e['observation'])!=e['observation_hash']:
            raise ValueError(f'event chain invalid at record {count}')
        previous=hashed;count+=1
    return dict(valid=True,events=count,last_hash=previous)
