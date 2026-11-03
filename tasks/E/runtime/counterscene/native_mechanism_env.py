"""Calibrated controller actions executed on the actual rendered mechanism."""
import copy,hashlib,json,math
from pathlib import Path
from .agentic_program import execute_candidate
from .native_env import digest
from .native_candidate import retained_source_matches
from .asset_scene import verify_artifacts
from .native_mechanism import check_tool_motion,TOOL_BRIDGE_BOTTOM,TOOL_CONNECTOR_TOP,VERIFIED_LINK_RADIUS
from .native_mechanism_artifacts import render_motion
from .spatial_suite import kinematics as k,pick_place as pp,protocol
from .program import CONTRACT


class NativeMechanismEnv(protocol.MechanismEnv):
    def __init__(self,task,private,offsets,artifacts):
        super().__init__(copy.deepcopy(task),copy.deepcopy(private),14)
        self.offsets=offsets[:];self.artifacts=Path(artifacts).resolve();self.artifacts.mkdir(parents=True,exist_ok=True)
        self.executions={};self.observation_artifacts=[]
        self.capture_cache=None

    def preview(self):
        task=self.task
        base=task['base'] if not task['free_base'] else [(a+b)/2 for a,b in zip(*task['base_region'])]
        lengths=task['link_lengths'] if not task['free_len'] else [(a+b)/2 for a,b in task['link_length_range']]
        physical=dict(mechanism=dict(base=base,link_lengths=lengths,joint_limits_deg=task['hardware_limits_deg']),keyframes=[task['home']],events=[])
        out=(self.capture_cache or self.artifacts.parent/'previews')/digest(dict(task=task,physical=physical))
        result=json.loads((out/'result.json').read_text()) if out.exists() else render_motion(task,physical,out,preview=True)
        verify_artifacts(out,result['renders']+result['frames'])
        metadata=dict(scene_sha256=result['scene_artifact']['sha256'],renders=result['renders'],note='Public nominal template at physical home; not a planned solution')
        self.observation_artifacts.append(dict(directory=str(out),**metadata))
        return metadata,[(out/row['file']).read_bytes() for row in result['renders']]

    def prompt(self):
        text,_=super().prompt();metadata,images=self.preview()
        if self.task.get('stateful_operation'):
            text+='\nSequential operation: start at the actual terminal physical joints from the previous operation. After release, visit the final onward waypoint. Previously released objects remain fixed physical obstacles: '+json.dumps(self.task.get('resting_objects',[]))
        return text+f'''\nNative geometry: the attached views show the actual procedural mechanism, cabinets, balls and gripper. Input capture={json.dumps(metadata)}.
The rendered arm has verified capsule enclosure radius {VERIFIED_LINK_RADIUS}m (includes 0.1mm numerical allowance), not only a stick diagram. The fixed pedestal is 0.25x0.25x0.025m. Tool fingers are x=tip_x +/-0.0925, widths 0.025x0.06m, bottom=tip_z-0.04 and top=tip_z+{TOOL_BRIDGE_BOTTOM}; the bridge is 0.21x0.06x0.03m at bottom=tip_z+{TOOL_BRIDGE_BOTTOM}; connector runs from 0.13m before the tip along link2 to tip+[0,0,{TOOL_CONNECTOR_TOP}] with enclosure radius 0.0181m. Tool parts retain these world-axis orientations. Additional continuous tool/fixture/floor/non-adjacent-self and resting-ball clearances must pass; intended held-ball/tool contact allows at most 1cm conservative intrusion. No force dynamics. Unknown mesh enclosure is not a pass.
Controller calibration: physical joint angles = COMMAND angles + unknown fixed offsets. Home, limits and targets above use physical angles. Call {{"tool":"calibrate_hardware"}} for actual offsets. forward.q and keyframes are command angles; declared limits remain physical. For simulate/submit use {{"tool":"simulate" or "submit","source":"def build(public_params): ..."}}, returning the mechanism payload above. public_params has room only; imports, attribute/file/network/private access are forbidden. Commands must be finite, within +/-360 degrees; all hardware and declared physical limits still apply. Actual physical keyframes, meshes and renders are saved. A missed grasp leaves the ball where it was; release uses the actual tip, never the target position.
Program contract: {CONTRACT}''',images

    def materialize(self,source):
        key=digest(dict(task=self.task,offsets=self.offsets,source=source))
        if key in self.executions:return self.executions[key]
        out=self.artifacts/key
        if out.exists():
            retained=json.loads((out/'execution.json').read_text())
            if retained.get('task_sha256')!=digest(self.task) or retained.get('offsets')!=self.offsets or retained.get('source_sha256')!=hashlib.sha256(source.encode()).hexdigest() or not retained_source_matches(out/'candidate.py',source):raise ValueError('Retained native motion identity differs')
            verify_artifacts(out,retained['renders']+retained['frames'])
            self.executions[key]=retained;return retained
        executed=execute_candidate(source,self.task['room'])
        if not executed['executed']:return executed
        payload=copy.deepcopy(executed['payload'])
        try:
            keys=payload['keyframes']
            if not isinstance(keys,list) or not 1<=len(keys)<=32 or any(not isinstance(q,list) or len(q)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>360 for v in q) for q in keys):raise ValueError('Finite command triples within +/-360 and at most 32 keyframes required')
            payload['keyframes']=[[q[i]+self.offsets[i] for i in range(3)] for q in keys]
            if self.task['free_len']:
                lengths=payload['mechanism']['link_lengths']
                if len(lengths)!=2 or any(type(v) not in (int,float) or not lo<=v<=hi for v,(lo,hi) in zip(lengths,self.task['link_length_range'])):raise ValueError('Requested link scale outside permitted range')
            if self.task['free_base']:
                base=payload['mechanism']['base']
                if len(base)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>100 for v in base):raise ValueError('Finite bounded base coordinates required')
            clean=k.check_motion(self.task,payload)
            if not clean['valid_output']:raise ValueError('Invalid physical mechanism/keyframe schema')
            _,error=pp._events(self.task,payload,len(keys))
            if error:raise ValueError(error)
        except (KeyError,TypeError,ValueError,OverflowError) as error:return dict(executed=False,error=str(error)[:300])
        payload['mechanism']=dict(clean['mechanism'],joint_limits_deg=payload['mechanism']['joint_limits_deg'])
        out.mkdir(parents=True,exist_ok=False)
        (out/'candidate.py').write_bytes(source.encode('utf-8'));(out/'physical_motion.json').write_text(json.dumps(payload,indent=2))
        report=check_tool_motion(self.task,payload);scene=render_motion(self.task,payload,out/'scene')
        result=dict(executed=True,payload=payload,report=report,scene_artifact=dict(scene['scene_artifact'],file='scene/'+scene['scene_artifact']['file']),frames=[dict(row,file='scene/'+row['file']) for row in scene['frames']],renders=[dict(row,file='scene/'+row['file']) for row in scene['renders']],artifact_directory=str(out),source_sha256=hashlib.sha256(source.encode()).hexdigest(),task_sha256=digest(self.task),offsets=self.offsets)
        (out/'execution.json').write_text(json.dumps(result,indent=2));self.executions[key]=result
        return result

    def step(self,action):
        if not self.done and isinstance(action,dict) and action.get('tool') in ('simulate','submit'):
            result=self.materialize(action.get('source'))
            if not result['executed']:
                observation,images=protocol.Env.step(self,None);observation['error']=result['error'];return observation,images
            action=dict(result['payload'],tool=action['tool'],source=action['source'])
        return protocol.Env.step(self,action)

    def tool(self,action):
        if action['tool']=='calibrate_hardware':return dict(joint_zero_offsets_deg=self.offsets[:],equation='physical = command + offset')
        if action['tool']=='simulate':
            result=self.materialize(action['source']);root=Path(result['artifact_directory'])
            return dict(report=copy.deepcopy(result['report']),capture=dict(scene=result['scene_artifact'],renders=result['renders']),_images=[(root/row['file']).read_bytes() for row in result['renders']])
        if action['tool']=='forward':action=dict(action,q=[action['q'][i]+self.offsets[i] for i in range(3)])
        return super().tool(action)

    def final(self):
        if not self.submission:return dict(success=False,reason='no_submission')
        execution=self.materialize(self.submission['source']);report=execution['report'];arm=report['arm_report'];sub=execution['payload']
        return dict(success=report['ok'],waypoints_ok=sum(row['ok'] for row in arm['waypoints']),waypoints=len(self.task['waypoints']),segments_ok=sum(row['ok'] for row in report['segments']),segments=len(report['segments']),
                    limits_ok=all(row['ok'] for row in arm['limits']) and not any('outside declared' in problem for problem in arm['problems']),payload_ok=arm.get('payload',{}).get('ok'),payload_report=arm.get('payload'),
                    joint_travel_deg=sum(abs(a-b) for left,right in zip(sub['keyframes'],sub['keyframes'][1:]) for a,b in zip(left,right)),min_certified_clearance_m=min((row['certified_margin_m'] for row in report['segments']),default=None),
                    native_report=report,scene_artifact=execution['scene_artifact'],artifact_directory=execution['artifact_directory'])

    def terminal_state(self):
        from .motion_state import carried_motion_state
        if not self.submission:raise ValueError('A committed execution is required')
        return carried_motion_state(self.task,self.materialize(self.submission['source']),self.task.get('resting_objects',[]))

    def continue_from(self,state):
        from .motion_state import apply_motion_state
        self.task=apply_motion_state(self.task,state)
