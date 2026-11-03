"""Progressive installation episodes: execute, environment change, replan."""
import copy
from .survey_repair import make_survey_pair
from .observed_grounding import make_grounding_pair
from .calibrated_mechanism import make_calibration_pair


class ProgressiveEnv:
    def __init__(self,worlds,stages,condition="structured",*,explicit_stages=False):
        if explicit_stages and (stages < 1 or len(worlds) != stages):
            raise ValueError('Explicit episodes require one independently certified world per stage')
        self.explicit_stages=explicit_stages
        self.transitions=[]
        self.condition=condition
        self.worlds=worlds;self.stages=stages;self.stage=0
        self.current=copy.deepcopy(worlds[0]);self.turn=0;self.max_turns=stages*14
        self.done=False;self.log=[];self.candidate_audits=[];self.visible_violations=[];self.audit_previous=None;self.accepted=[];self.last_result=None

    def __getattr__(self,key):return getattr(self.current,key)

    def prompt(self):
        text,images=self.current.prompt()
        return text+f'\nProgressive installation: {self.stages} ordered physical stages. Complete the current operation, then use commit_stage with the same program and answer fields as submit to execute and accept it. An external change follows accepted non-final stages; inspect/capture/calibrate the updated world and repair the previous solution. Final submit is for the last stage only. Commits are irreversible and do not reveal hidden correctness; use public verification before committing. All stages are privately scored and cannot be replaced by only a final scene. Total action cap: {self.max_turns}.',images

    def step(self,action):
        if self.done:return dict(error='Episode ended'),[]
        self.turn+=1
        obs={};images=[]
        if isinstance(action,dict) and 'source' in action:
            from .agentic_program import execute_candidate
            from .spatial_suite import core
            from .part2 import transition_metrics
            executed=execute_candidate(action['source'],self.current.task['room'])
            audit=dict(turn=self.turn,stage=self.stage,executed=executed['executed'])
            if self.current.experiment in ('C','D') and hasattr(self.current,'materialize') and action.get('tool') in ('render','compare','simulate','submit','commit_stage'):
                native=self.current.materialize(action['source']) if executed['executed'] else {}
                audit['valid_output']=bool(native.get('executed') and (self.current.experiment=='D' or native.get('geometry',{}).get('valid_output')))
            if self.current.experiment in ('A','B','E'):
                initial=self.current.task.get('initial',{})
                check=getattr(self.current,'check_placements',lambda placements:core.check_scene(self.current.task,placements))
                previous=self.audit_previous or check(initial)
                report=check(executed['payload'])
                before={r['id']:r['satisfied'] for r in previous.get('constraints',[])}
                after={r['id']:r['satisfied'] for r in report.get('constraints',[])}
                audit.update(transition_metrics(before,after,self.visible_violations))
                audit.update(valid_output=report['valid_output'],physical_issues=len(report['physical']),constraint_satisfied=sum(after.values()),constraint_total=len(self.current.task['constraints']))
                self.audit_previous=report
            self.candidate_audits.append(audit)
        tool=action.get('tool') if isinstance(action,dict) else None
        if tool=='submit' and self.stage<self.stages-1:
            obs=dict(error='Current installation must be executed with commit_stage before the external change occurs')
        elif tool=='commit_stage':
            trial=copy.deepcopy(self.current)
            trial.step(dict(action,tool='submit'))
            result=trial.final()
            motion_state=None;state_error=None
            if trial.experiment=='D' and trial.task.get('stateful_operation') and trial.submission is not None:
                try:motion_state=trial.terminal_state()
                except ValueError as error:state_error=str(error)
            if state_error:
                obs=dict(error=state_error)
            elif trial.submission is None or result.get('reason')=='invalid_output':
                obs=dict(error='Program did not produce a valid executable payload; correct the public output format before committing.')
            else:
                self.accepted.append(dict(stage=self.stage,action=copy.deepcopy(action),result=result))
                self.stage+=1
                if self.stage==self.stages:
                    self.current=trial;self.done=True;self.last_result=result
                    obs=dict(execution_committed=True,all_stages_completed=True)
                else:
                    previous=trial.submission.get('placements')
                    self.current=copy.deepcopy(self.worlds[self.stage if self.explicit_stages else self.stage%len(self.worlds)])
                    if motion_state is not None:self.current.continue_from(motion_state)
                    self.audit_previous=None;self.visible_violations=[]
                    if self.current.experiment in ('A','B','E') and previous is not None:
                        from .spatial_suite import core
                        expected=self.current.task['initial']
                        initial=copy.deepcopy(previous)
                        changed=set(self.current.task['locked'])
                        event=self.current.priv.get('physical_event')
                        if self.explicit_stages and self.current.task.get('carry_all_previous_state'):
                            changed=set()
                        elif self.explicit_stages and event:
                            changed={event['host']}
                        elif self.current.experiment=='E':changed.update(self.current.priv['corrupted'])
                        for oid in changed:initial[oid]=copy.deepcopy(expected[oid])
                        self.current.task['initial']=initial
                        if self.current.experiment=='B':
                            check=getattr(self.current,'check_placements',lambda placements:core.check_scene(self.current.task,placements))
                            self.current.task['initial_report']=check(initial)
                            self.current.priv['gold_edit']=core.edit_cost(initial,self.current.priv['gold'])
                    transition=dict(from_stage=self.stage-1,to_stage=self.stage,
                        physical_event=copy.deepcopy(self.current.priv.get('physical_event')),
                        before_event=copy.deepcopy(previous),
                        after_event=copy.deepcopy(self.current.task.get('initial')),
                        committed_action=copy.deepcopy(action),
                        committed_result=copy.deepcopy(result),
                        scope='Actual committed state and environment update; target gold is not substituted for model state')
                    after=transition['after_event']
                    transition['environment_changed_objects']=[oid for oid in previous if previous[oid]!=after.get(oid)] if isinstance(previous,dict) and isinstance(after,dict) else None
                    if self.current.experiment=='C':
                        transition['carried_candidate']=copy.deepcopy(previous)
                        transition['scope']='Actual prior reconstruction is retained in the trace; new target must be acquired through public views. No target gold is substituted for model output.'
                    self.transitions.append(transition)
                    obs=dict(execution_committed=True,stage=self.stage,event='External update completed: remeasure the live installation/target/controller; preserve constraints and complete the next operation.')
                    if motion_state is not None:
                        transition['actual_motion_state']=copy.deepcopy(motion_state)
                        metadata,images=self.current.preview()
                        obs.update(actual_motion_state=copy.deepcopy(motion_state),task=copy.deepcopy(self.current.task),capture=metadata)
                    if self.current.task.get('operation'):
                        obs.update(operation=copy.deepcopy(self.current.task['operation']),active_constraints=copy.deepcopy(self.current.task['constraints']),scene_policy=copy.deepcopy(self.current.task.get('scene_policy',{})))
                        obs['event']='Previous operation committed. Continue the next operation from the actual current installation; no placement reset occurred.'
                    if self.current.experiment=='C' and self.current.task.get('observation_contract')=='three-view-v1':
                        observations,images=self.current.observe_stage()
                        obs.update(observations)
        else:
            obs,images=self.current.step(action)
            if self.current.done:
                self.done=True;self.last_result=self.current.final()
        if self.turn>=self.max_turns:self.done=True
        if tool=='check' and 'error' not in obs:
            from .part2 import feedback
            obs=feedback(obs,self.condition)
        if tool=='check':self.visible_violations=[r['id'] for r in obs.get('constraints',[]) if not r['satisfied']]
        obs['total_actions_left']=max(0,self.max_turns-self.turn)
        self.log.append(dict(turn=self.turn,stage=self.stage,action=copy.deepcopy(action),observation=copy.deepcopy(obs),n_images=len(images)))
        return obs,images

    def final(self):
        success=bool(self.last_result and self.last_result['success'] and all(r['result']['success'] for r in self.accepted) and (len(self.accepted)==self.stages or len(self.accepted)==self.stages-1 and self.current.submission is not None))
        results=[r['result'] for r in self.accepted]+([self.last_result] if self.last_result and len(self.accepted)<self.stages else [])
        executed=len(self.accepted)+int(bool(self.last_result and len(self.accepted)<self.stages and self.current.submission is not None and self.last_result.get('reason')!='invalid_output'))
        schemas=[a['valid_output'] for a in self.candidate_audits if 'valid_output' in a]
        return dict(transitions=copy.deepcopy(self.transitions),candidate_audits=self.candidate_audits,execution_valid=bool(self.candidate_audits) and all(a['executed'] for a in self.candidate_audits),execution_valid_scope='Restricted program returned a dictionary; physical correctness is separate',native_schema_valid=all(schemas) if schemas else None,stage_results=results,success=success,stages_completed=len(self.accepted)+(1 if success and len(self.accepted)<self.stages else 0),stage_executions=executed,stage_successes=sum(bool(r['success']) for r in results),stages_required=self.stages,last_stage_result=self.last_result)


def make_progressive(folder,condition="structured"):
    exp=folder.name[0]
    if condition not in ('execution','scalar','structured') or exp!='B' and condition!='structured':raise ValueError('Unsupported feedback condition')
    factory=make_grounding_pair if exp=='C' else make_calibration_pair if exp=='D' else make_survey_pair
    worlds=factory(folder)
    level=folder.name.split('-')[1]
    stages={'L1':1,'L2':2,'L3':3,'k06':1,'k14':2,'k34':3}[level]
    # D's full continuous pick/carry/place sequence is one operation; no unverified reset between motions.
    return ProgressiveEnv(worlds,1 if exp=='D' else stages,condition)
