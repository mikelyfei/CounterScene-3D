"""Explicit partial-repair contracts; pending collisions are never hidden."""
import copy,json
from .native_env import NativeSceneEnv
from .spatial_suite import core
from .spatial_suite import constraint_graph as cg


def pair_key(objects):return tuple(sorted(objects))


def partial_geometry(report,allowed_pending_pairs,allowed_pending_issues=()):
    allowed={pair_key(pair) for pair in allowed_pending_pairs};pending=[];violations=[]
    known={'supported','approved_full_footprint_support','seat_faces_work_surface','screen_faces_seat'}
    if any(issue.get('check') not in known for issue in allowed_pending_issues):raise ValueError('Only explicitly witnessed pending support/facing issues may be deferred')
    declared={json.dumps(issue,sort_keys=True) for issue in allowed_pending_issues};pending_issues=[]
    for issue in report.get('physical',[]):
        if issue['check']=='no_interpenetration' and issue.get('native_status') in ('surface_intersection','containment') and pair_key(issue['objects']) in allowed:pending.append(issue)
        elif json.dumps(issue,sort_keys=True) in declared:pending_issues.append(issue)
        else:violations.append(issue)
    return dict(valid_output=report.get('valid_output',False),pending_collisions=pending,pending_issues=pending_issues,violations=violations)


class NativePartialRepairEnv(NativeSceneEnv):
    def prompt(self):
        text,images=super().prompt()
        return text+'\nRepair stage='+str(self.task['repair_stage'])+'. Only the current work zone is released for editing. Preserve locked prior/future zones. Declared pending collisions are visible and permitted only until their release stage: '+str(self.task['allowed_pending_pairs'])+'. Explicitly witnessed pending support/facing issues: '+str(self.task.get('allowed_pending_issues',[]))+'. Do not create new faults. Final-stage hard/soft and exact infeasibility requirements remain mandatory.',images

    def tool(self,action):
        result=super().tool(action)
        if action.get('tool')=='check' and result.get('valid_output'):
            result['stage_scope']=partial_geometry(result,self.task['allowed_pending_pairs'],self.task.get('allowed_pending_issues',[]))
        return result

    def final(self):
        if self.task['repair_stage_final']:return super().final()
        if not self.submission:return dict(success=False,reason='no_submission')
        execution=self.materialize(self.submission['source']);report=execution['geometry'];placements=execution['payload']
        scope=partial_geometry(report,self.task['allowed_pending_pairs'],self.task.get('allowed_pending_issues',[]))
        impossible={key for key,value in self.priv.get('soft',{}).items() if not value['feasible']}
        required=[row for row in report.get('constraints',[]) if row['id'] not in impossible]
        moved=set(core.edit_cost(self.task['initial'],placements)['moved'])&set(self.task['locked'])
        # A fixed gold completion is a sufficient witness, not a feasibility
        # decision: another continuation may work when this particular one fails.
        completion=copy.deepcopy(placements)
        for oid in self.priv['remaining_to_repair']:completion[oid]=copy.deepcopy(self.priv.get('completion_gold',self.priv['gold'])[oid])
        final_task=copy.deepcopy(self.priv['final_task'])
        verifier=NativeSceneEnv(final_task,self.priv,self.artifacts/'completion_witness')
        future=verifier.check_placements(completion)
        future_ok=future['valid_output'] and not future['physical'] and all(row['satisfied'] for row in future['constraints'] if row['id'] not in impossible)
        success=scope['valid_output'] and not scope['violations'] and not moved and all(row['satisfied'] for row in required)
        diagnostics=cg.score_e(self.task,self.priv,placements,self.submission.get('checker_issues',[]),report=report) if self.experiment=='E' else {}
        diagnostics.update(success=success,geometry=report,stage_scope=scope,locked_moved=sorted(moved),completion_witness_valid=future_ok,continuation_status='witnessed_by_fixed_gold' if future_ok else 'not_established_by_fixed_gold',continuation_scope='Fixed-gold failure does not establish infeasibility. Whole-episode success requires every local stage contract plus the actual final hard/physical continuation.',scene_artifact=execution['scene_artifact'],artifact_directory=execution['artifact_directory'])
        return diagnostics
