"""Active target capture plus multi-view reconstruction, with metric audit."""
import copy,json,math
from .spatial_suite import core,grounding,protocol
from .survey_repair import ProgramStepMixin

CENTER_TOLERANCE_M=.05


class ObservedGroundingEnv(ProgramStepMixin,protocol.GroundingEnv):
    def prompt(self):
        text,_=super().prompt()
        return text+'''\nThe scanner image has not been acquired yet. Call {"tool":"capture_reference"} to obtain the actual target reference. Other public camera queries remain available within their budget. For render/compare/submit send source, a restricted Python build(public_params) returning the full placements dictionary; public_params contains room only. Imports and attribute access are forbidden. In addition to hidden-view IoU and physical rules, every object centre must be reconstructed within 0.05 m of its target.''',[]

    def tool(self,action):
        if action['tool']=='capture_reference':return dict(result='Actual reference capture',_images=self.reference_images('reference'))
        return super().tool(action)

    def final(self):
        result=super().final()
        placements=(self.submission or {}).get('placements')
        objs,problems=core.build_objects(self.task,placements)
        target,_=core.build_objects(self.task,self.priv['gold'])
        errors={i:math.dist(core.center(objs[i]),core.center(target[i])) for i in self.priv['gold']} if not problems else {}
        result['metric_centres_ok']=not problems and all(d<=CENTER_TOLERANCE_M for d in errors.values())
        result['metric_centre_errors_m']=errors
        result['success']=result['success'] and result['metric_centres_ok']
        return result


def make_grounding_pair(folder):
    private=json.loads((folder/'private.json').read_text(encoding='utf-8'))
    task=private['task_internal']
    task['public_cameras']=grounding.cameras(task['room'],['reference']+task['queryable_views'])
    for oid,placement in private['gold'].items():
        if placement['position'][2]!=0:continue
        for delta in (.3,-.3,.6,-.6):
            for axis in (0,1):
                other=copy.deepcopy(private)
                other['gold'][oid]['position'][axis]=round(placement['position'][axis]+delta,4)
                report=core.check_scene(task,other['gold'])
                if not report['all_satisfied']:continue
                pair=[ObservedGroundingEnv(copy.deepcopy(task),p,14) for p in (copy.deepcopy(private),other)]
                # Both images are generated from each target; no v3 gold asset appearance is reused.
                assert pair[0].prompt()==pair[1].prompt()
                assert abs(delta)>2*CENTER_TOLERANCE_M
                return pair
    raise ValueError('No feasible target pair found')
