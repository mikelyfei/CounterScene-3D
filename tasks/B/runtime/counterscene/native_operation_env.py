"""Native stage operations with explicit public goals and private pose checks."""
import json,math
from .native_env import NativeSceneEnv


class NativeOperationEnv(NativeSceneEnv):
    def prompt(self):
        text,images=super().prompt()
        text+='\nCurrent operation='+json.dumps(self.task['operation'])+'\nInspect the actual current installation. Public operation requirements replace earlier placement requirements only where explicitly listed. Preserve locked objects and use real mesh evidence.'
        return text,images

    def tool(self,action):
        result=super().tool(action)
        if action.get('tool')=='inspect_scene' and 'error' not in result:result['operation']=self.task['operation']
        return result

    def final(self):
        result=super().final()
        if not self.submission:return result
        placements=self.materialize(self.submission['source'])['payload'];checks=[]
        for oid,target in self.priv['operation_targets'].items():
            pose=placements.get(oid,{})
            error=math.dist(pose['position'],target['position']) if 'position' in pose else None
            ok=error is not None and error<=.02 and pose.get('yaw',0)%360==target['yaw']%360
            checks.append(dict(object=oid,position_error_m=error,ok=ok))
        result['operation_checks']=checks
        result['success']=result['success'] and all(row['ok'] for row in checks)
        return result
