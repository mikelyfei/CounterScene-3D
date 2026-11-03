"""B survey-and-repair worlds: preserve measured locked geometry, then repair."""
import copy,json,math
from .spatial_suite import core,protocol,constraint_graph as cg
from .agentic_program import execute_candidate


class ProgramStepMixin:
    def step(self,action):
        if self.done:return super().step(action)
        if isinstance(action,dict) and action.get('tool') in ('check','submit','measure','render','compare'):
            result=execute_candidate(action.get('source'),self.task['room'])
            if not result['executed']:
                obs,images=super().step(None)
                obs['error']=result['error'];return obs,images
            action=dict(action,placements=result['payload'])
        return super().step(action)


class SurveyRepairEnv(ProgramStepMixin,protocol.RepairEnv):
    def prompt(self):
        text,images=super().prompt()
        text=text.replace(json.dumps(self.task['initial']),json.dumps({k:v for k,v in self.task['initial'].items() if k not in self.task['locked']}))
        text=text.replace(json.dumps(protocol.compact_report(self.task['initial_report'])),'Unavailable until the live scene is inspected.')
        text=text.replace('- {"tool": "check", "placements": {...}}: verifier report for a candidate.', '- {"tool": "inspect_scene"}: measure current placements, including immutable equipment.\n- {"tool": "check", "source": "def build(public_params): ..."}: execute a scene program and check it.')
        text=text.replace('"placements": {...}, "infeasible"','"source": "def build(public_params): ...", "infeasible"')
        return text+'\nLocked equipment coordinates are omitted above: inspect_scene measures their actual locations. The same equipment must remain there. Programs return the complete placements dictionary. Restricted Python: no imports or attribute access; literals, arithmetic, assignments, if/else, comprehensions and return are supported. public_params contains room only.',images

    def tool(self,action):
        if action['tool']=='inspect_scene':return dict(placements=copy.deepcopy(self.task['initial']))
        return super().tool(action)



def make_repair_pair(folder):
    private=json.loads((folder/'private.json').read_text(encoding='utf-8'))
    original=private['task_internal']
    impossible=[k for k,v in private['soft'].items() if not v['feasible']]
    for anchor in original['locked']:
        start=private['gold'][anchor]['position']
        for delta in (.15,-.15,.3,-.3,.6,-.6):
            for axis in (0,1):
                candidate=copy.deepcopy(private)
                task=candidate['task_internal']
                task['initial'][anchor]['position'][axis]=round(start[axis]+delta,4)
                candidate['gold'][anchor]['position'][axis]=round(start[axis]+delta,4)
                task['initial_report']=core.check_scene(task,task['initial'])
                if not cg.score_b(task,candidate,candidate['gold'],impossible)['success']:continue
                candidate['gold_edit']=core.edit_cost(task['initial'],candidate['gold'])
                worlds=[]
                for p in (copy.deepcopy(private),candidate):
                    worlds.append(SurveyRepairEnv(p['task_internal'],p,10))
                assert worlds[0].prompt()==worlds[1].prompt()
                assert math.dist(worlds[0].task['initial'][anchor]['position'],worlds[1].task['initial'][anchor]['position'])>.002
                return worlds
    raise ValueError('No certified alternate locked placement for this instance')


class SurveySceneMixin(ProgramStepMixin):
    def prompt(self):
        text,images=super().prompt()
        if 'initial' in self.task:
            text=text.replace(json.dumps(self.task['initial']),json.dumps({k:v for k,v in self.task['initial'].items() if k not in self.task['locked']}))
        text=text.replace('"placements": {...}','"source": "def build(public_params): ..."')
        return text+'\nAn existing installation fixture must stay at its measured position and yaw: '+json.dumps(self.task['locked'])+'. Call {"tool":"inspect_scene"} to measure it. All check/measure/submit candidates use source, a restricted Python build(public_params) returning the complete placements dictionary. No imports or attribute access. public_params contains room only.',images

    def tool(self,action):
        if action['tool']=='inspect_scene':
            return dict(placements={k:copy.deepcopy(self.task['initial'][k]) for k in self.task['locked']})
        return super().tool(action)

    def final(self):
        result=super().final()
        placements=(self.submission or {}).get('placements')
        moved=core.edit_cost(self.task['initial'],placements)['moved'] if isinstance(placements,dict) else self.task['locked']
        result['locked_moved']=sorted(set(moved)&set(self.task['locked']))
        result['success']=result['success'] and not result['locked_moved']
        return result


class SurveyConstructionEnv(SurveySceneMixin,protocol.ConstraintEnv):pass
class SurveyFeedbackEnv(SurveySceneMixin,protocol.FeedbackEnv):pass


def make_survey_pair(folder):
    if folder.name.startswith('B-'):return make_repair_pair(folder)
    private=json.loads((folder/'private.json').read_text(encoding='utf-8'))
    exp=private['task_internal']['experiment']
    if exp not in ('A','E'):raise ValueError('Survey geometry pair supports A/B/E')
    cls=SurveyConstructionEnv if exp=='A' else SurveyFeedbackEnv
    for anchor in private.get('clean_objects',list(private['gold'])):
        base=copy.deepcopy(private)
        base['task_internal'].setdefault('initial',copy.deepcopy(base['gold']))
        base['task_internal']['locked']=[anchor]
        if base['task_internal']['initial'][anchor]!=base['gold'][anchor]:continue
        for delta in (.15,-.15,.3,-.3,.6,-.6):
            for axis in (0,1):
                other=copy.deepcopy(base)
                for placements in (other['gold'],other['task_internal']['initial']):
                    placements[anchor]['position'][axis]=round(placements[anchor]['position'][axis]+delta,4)
                report=core.check_scene(other['task_internal'],other['gold'])
                if not report['all_satisfied']:continue
                pair=[cls(p['task_internal'],p,12) for p in (base,other)]
                assert pair[0].prompt()==pair[1].prompt()
                return pair
    raise ValueError('No certified fixture alternative')
