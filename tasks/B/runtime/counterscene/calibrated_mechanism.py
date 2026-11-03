"""Physical mechanism planning with an observed controller-to-joint calibration."""
import copy,json
from .spatial_suite.protocol import MechanismEnv
from .agentic_program import execute_candidate


class CalibratedMechanismEnv(MechanismEnv):
    def __init__(self,task,private,offsets):
        super().__init__(task,private,14)
        self.offsets=offsets

    def prompt(self):
        text,images=super().prompt()
        return text+'''\nController calibration: the physical joint angles equal commanded angles plus unknown fixed zero offsets in degrees. All geometry, home, joint limits and targets above use PHYSICAL angles. Keyframes and forward.q use COMMAND angles; hardware/joint limits remain physical. Call {"tool":"calibrate_hardware"} to measure offsets. For simulate/submit, send {"tool":"simulate" or "submit", "source":"def build(public_params): ..."}; the restricted Python function returns the full mechanism payload with command-angle keyframes. public_params contains room only. No imports or attribute access.''',images

    def tool(self,action):
        if action['tool']=='calibrate_hardware':return dict(joint_zero_offsets_deg=self.offsets[:],equation='physical = command + offset')
        if action['tool']=='forward':
            action=dict(action,q=[action['q'][i]+self.offsets[i] for i in range(3)])
        return super().tool(action)

    def step(self,action):
        if self.done:return super().step(action)
        if isinstance(action,dict) and action.get('tool') in ('simulate','submit'):
            result=execute_candidate(action.get('source'),self.task['room'])
            if not result['executed']:
                obs,images=super().step(None);obs['error']=result['error'];return obs,images
            payload=copy.deepcopy(result['payload'])
            keys=payload.get('keyframes')
            if isinstance(keys,list):
                try:payload['keyframes']=[[q[i]+self.offsets[i] for i in range(3)] for q in keys]
                except (TypeError,IndexError):
                    obs,images=super().step(None);obs['error']='Invalid command keyframes';return obs,images
            action=dict(payload,tool=action['tool'],source=action.get('source'))
        return super().step(action)


def make_calibration_pair(folder):
    private=json.loads((folder/'private.json').read_text(encoding='utf-8'))
    return [CalibratedMechanismEnv(copy.deepcopy(private['task_internal']),copy.deepcopy(private),offsets)
            for offsets in ([12.,0.,0.],[-12.,0.,0.])]
