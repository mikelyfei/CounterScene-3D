"""Part 2 feedback boundary and trajectory metrics; no provider dependencies."""
from copy import deepcopy


def feedback(report, condition):
    if condition == 'structured':
        return deepcopy(report)
    if condition not in ('execution', 'scalar'):
        raise ValueError('Unknown feedback condition: ' + condition)
    out = {'valid_output': report.get('valid_output', False)}
    if condition == 'scalar':
        rows = report.get('constraints', [])
        out.update(pass_check=bool(report.get('all_satisfied')), score=(sum(bool(r['satisfied']) for r in rows) / len(rows) if rows else None))
    return out


def transition_metrics(before, after, reported_violations):
    """Missing/invalid next-state checks count as not satisfied, never as repairs."""
    satisfied = {k for k, v in before.items() if v}
    violated = set(reported_violations)
    repaired = {k for k, v in before.items() if not v and after.get(k, False)}
    regressed = {k for k in satisfied if not after.get(k, False)}
    return dict(alignment=len(violated & repaired)/len(violated) if violated else None,
                regression=len(regressed)/len(satisfied) if satisfied else None,
                regression_count=len(regressed), regression_denominator=len(satisfied),
                aligned_count=len(violated & repaired), alignment_denominator=len(violated))

class FrozenRepair:
    """Offline B episode using exported v3 data, never regenerating instances."""
    def __init__(self, folder, condition):
        import json
        from pathlib import Path
        from .spatial_suite.protocol import RepairEnv
        folder = Path(folder)
        self.private = json.loads((folder/'private.json').read_text(encoding='utf-8'))
        public = json.loads((folder/'public.json').read_text(encoding='utf-8'))
        self.env = RepairEnv(self.private['task_internal'], self.private, public['max_turns'])
        self.condition = condition
        self.previous = self.env.task['initial_report']
        self.visible_violations = []
        self.trace = []

    def prompt(self):
        import json
        from .spatial_suite.protocol import compact_report
        text, images = self.env.prompt()
        original = json.dumps(compact_report(self.env.task['initial_report']))
        if text.count(original) != 1:
            raise ValueError('Initial report boundary changed')
        visible = feedback(compact_report(self.previous), self.condition)
        self.visible_violations = [r['id'] for r in visible.get('constraints', []) if not r['satisfied']]
        return text.replace(original, json.dumps(visible)) + '\nFeedback condition: ' + self.condition, images

    def step(self, action):
        from .spatial_suite import core
        if self.env.done:
            return self.env.step(action)
        obs, images = self.env.step(action)
        metrics = None
        if isinstance(action, dict) and action.get('tool') in ('check', 'submit'):
            report = core.check_scene(self.env.task, action.get('placements'))
            before = {r['id']:r['satisfied'] for r in self.previous.get('constraints', [])}
            after = {r['id']:r['satisfied'] for r in report.get('constraints', [])}
            metrics = transition_metrics(before, after, self.visible_violations)
            metrics['new_physical_violations'] = sorted(physical_violations(report) - physical_violations(self.previous))
            metrics['physical_violations'] = sorted(physical_violations(report))
            metrics['valid_output'] = report.get('valid_output', False)
            if action.get('tool') == 'check':
                metrics['reward_audit'] = reward_audit(report if self.condition != 'execution' else None, report, list(before))
            self.previous = report
        if isinstance(action, dict) and action.get('tool') == 'check' and 'error' not in obs:
            budget = {k:v for k,v in obs.items() if k not in ('constraints','physical','all_satisfied','valid_output','problems')}
            obs = dict(feedback(obs, self.condition), **budget)
        self.visible_violations = [r['id'] for r in obs.get('constraints', []) if not r['satisfied']]
        self.trace.append(dict(action=deepcopy(action), observation=deepcopy(obs), metrics=metrics))
        return obs, images

    def final(self):
        return self.env.final()

    def summary(self):
        from .spatial_suite import core
        final = self.final()
        placements = (self.env.submission or {}).get('placements') or {}
        initial = self.env.task['initial']
        protected = set(initial) - set(self.private['corrupted'])
        preserved, total = preservation_counts(self.env.task, placements, protected)
        initial_rows = self.env.task['initial_report']['constraints']
        impossible = {i for i,v in self.private['soft'].items() if not v['feasible']}
        targets = {r['id'] for r in initial_rows if not r['satisfied'] and r['id'] not in impossible}
        report = core.check_scene(self.env.task, placements)
        satisfied = {r['id'] for r in report.get('constraints', []) if r['satisfied']}
        sums = {key:sum((t['metrics'] or {}).get(key, 0) for t in self.trace)
                for key in ('regression_count','regression_denominator','aligned_count','alignment_denominator')}
        return dict(success=final['success'], final=final, **sums,
                    regression=sums['regression_count']/sums['regression_denominator'] if sums['regression_denominator'] else None,
                    alignment=sums['aligned_count']/sums['alignment_denominator'] if sums['alignment_denominator'] else None,
                    repair=len(targets & satisfied)/len(targets) if targets else None,
                    preservation=preserved/total if total else None,
                    protected_attributes=total, preserved_attributes=preserved,
                    actions=len(self.trace), submitted=self.env.submission is not None)


def frozen_env(folder, view_condition='active'):
    """Load exported task snapshots; C camera positions follow the frozen camera rule."""
    import json
    from pathlib import Path
    from .spatial_suite import protocol, grounding
    folder=Path(folder)
    private=json.loads((folder/'private.json').read_text(encoding='utf-8'))
    public=json.loads((folder/'public.json').read_text(encoding='utf-8'))
    task=private['task_internal']
    exp=public['experiment']
    classes={'A':protocol.ConstraintEnv,'B':protocol.RepairEnv,'C':protocol.GroundingEnv,
             'D':protocol.MechanismEnv,'E':protocol.FeedbackEnv}
    kwargs={}
    if exp=='C':
        task['public_cameras']=grounding.cameras(task['room'],['reference']+task['queryable_views'])
        render_root=folder.parent.parent/'renders'/folder.name
        kwargs['images']={v:(render_root/('gold_assets_'+v+'.png')).read_bytes()
                          for v in task['public_cameras'] if (render_root/('gold_assets_'+v+'.png')).exists()}
    if view_condition not in ('active','fixed'):
        raise ValueError('Unknown view condition')
    env=classes[exp](task,private,public['max_turns'],**kwargs)
    if exp=='C':
        def frozen_reference_images(view):
            if view=='reference':return [(folder/name).read_bytes() for name in public['images']]
            diagram=(folder/('query_'+view+'.png')).read_bytes()
            return [diagram]+([kwargs['images'][view]] if view in kwargs['images'] else [])
        # Historical replay must use its archived observations, never silently regenerate them with changed code.
        env.reference_images=frozen_reference_images
    if exp=='C' and view_condition=='fixed':
        original_tool=env.tool
        def fixed_tool(action):
            if action.get('tool')=='query_view':
                return dict(error='Additional reference views are disabled in the fixed-view condition')
            return original_tool(action)
        env.tool=fixed_tool
        original_prompt=env.prompt
        def fixed_prompt():
            text,images=original_prompt()
            return text+'\nFixed-view condition: query_view is disabled; only the initial reference is available.',images
        env.prompt=fixed_prompt
    return env


def run_client(episode, client, max_tokens, on_response=None):
    """Provider-neutral loop. The client sees only the serialized public history."""
    import base64
    import json
    from .spatial_suite.protocol import parse_action
    text,images=episode.prompt()
    messages=[dict(role='system',content='Solve the 3D task. Reply with exactly one JSON tool action.'),
              dict(role='user',content=text,images=[base64.b64encode(i).decode() for i in images])]
    calls=[]
    audits=[]
    candidates=[]
    query_events=[]
    env=episode.env if isinstance(episode,FrozenRepair) else episode
    infrastructure_error=None
    while not env.done:
        try:
            result,latency=client.generate(deepcopy(messages),max_tokens=max_tokens)
        except Exception as error:
            infrastructure_error=type(error).__name__
            break
        raw=result['message']['content']
        action,extra=parse_action(raw)
        calls.append(dict(response_id=result.get('provider_response',{}).get('id'),raw=raw,extra_content=extra,latency_ms=latency,
                          usage=result.get('provider_response',{}).get('usage'),cost_usd=result.get('api_cost_usd')))
        try:
            if on_response:on_response(dict(messages=deepcopy(messages),call=deepcopy(calls[-1]),action=action,provider_result=result))
            audit_env=getattr(env,'current',env)
            candidate_payload=action.get('placements') if isinstance(action,dict) else None
            if isinstance(action,dict) and 'source' in action:
                from .agentic_program import execute_candidate
                executed=execute_candidate(action['source'],audit_env.task['room'])
                candidate_payload=executed['payload'] if executed['executed'] else None
            held_before=len(env.held) if env.experiment=='C' else 0
            observation,images=episode.step(action)
            if env.experiment=='C' and isinstance(action,dict):
                if action.get('tool')=='query_view' and len(env.held)>held_before:
                    query_events.append(dict(turn=env.turn,view=action.get('view')))
                if candidate_payload is not None:
                    from .spatial_suite import grounding
                    score=getattr(audit_env,'score_placements',lambda placements:grounding.score_c(audit_env.task,audit_env.priv,placements,audit_env.held))
                    quality=score(candidate_payload)
                    candidates.append(dict(turn=env.turn,queries_so_far=len(query_events),quality=quality.get('hidden_mean_iou',0.0)))
            if env.experiment=='E' and isinstance(action,dict) and action.get('tool')=='check':
                from .spatial_suite import core
                check=getattr(audit_env,'check_placements',lambda placements:core.check_scene(audit_env.task,placements))
                truth=check(candidate_payload)
                visible=dict(observation,valid_output=observation.get('valid_output',True))
                audit=reward_audit(visible,truth,[c['id'] for c in audit_env.task['constraints']])
                audits.append(dict(turn=env.turn,**audit))
        except Exception as error:
            infrastructure_error='environment:'+type(error).__name__
            observation=dict(error='Local execution or evaluation failed; episode is infrastructure-ineligible',error_type=type(error).__name__)
            images=[]
        messages.extend([dict(role='assistant',content=raw),
                         dict(role='user',content=json.dumps(observation),images=[base64.b64encode(i).decode() for i in images])])
        if infrastructure_error is not None:break
    try:
        if isinstance(episode,FrozenRepair):summary=episode.summary()
        else:
            final=episode.final();summary=dict(success=final['success'],final=final,actions=env.turn)
    except Exception as error:
        infrastructure_error=infrastructure_error or 'environment:'+type(error).__name__
        summary=dict(success=False,final=dict(success=False,reason='local_evaluation_failure'),actions=env.turn)
    if env.experiment=='C':summary.update(information_metrics(candidates,len(query_events)))
    trace=episode.trace if isinstance(episode,FrozenRepair) else env.log
    summary['failure_labels']=failure_labels(summary['final'],trace) if infrastructure_error is None else ['infrastructure:'+infrastructure_error]
    return dict(messages=messages,summary=summary,calls=calls,audits=audits,candidates=candidates,query_events=query_events,trace=deepcopy(episode.trace if isinstance(episode,FrozenRepair) else env.log),
                evaluation_eligible=infrastructure_error is None,infrastructure_error=infrastructure_error)


def reward_audit(visible, hidden, constraint_ids):
    """Matched [0,1] constraint mean gated by each checker's physical verdict.

    Missing checker rows count as failures. This is a diagnostic proxy, not strict
    task success (e.g. B's impossibility claims are scored separately).
    """
    def quality(report):
        if not report.get('valid_output',False) or report.get('physical'):
            return 0.0
        by={r['id']:bool(r['satisfied']) for r in report.get('constraints',[])}
        return sum(by.get(i,False) for i in constraint_ids)/len(constraint_ids) if constraint_ids else None
    q=quality(hidden)
    r=quality(visible) if visible is not None else None
    return dict(visible_reward=r,hidden_quality=q,reward_gap=r-q if r is not None and q is not None else None)


def physical_violations(report):
    """Stable physical issue IDs, independent of checker ordering and depth noise."""
    return {r['check']+':'+ '+'.join(sorted(r['objects'])) for r in report.get('physical',[])}


def experiment_matrix(model_labels):
    """Paired 96-cell development plan: 36 B, 48 A/C/D/E, 12 fixed C."""
    rows=[]
    for exp in 'ABCDE':
        conditions=['execution','scalar','structured'] if exp=='B' else ['structured']
        for level in ('L1','L2','L3'):
            for seed in (0,1):
                for model in model_labels:
                    for condition in conditions:
                        for view in (['active','fixed'] if exp=='C' else ['active']):
                            rows.append(dict(instance=f'{exp}-{level}-s{seed}' if exp!='A' else f'A-k'+{'L1':'06','L2':'14','L3':'34'}[level]+f'-s{seed}',
                                             model=model,condition=condition,view_condition=view,repeat=1))
    return rows


from .openai_client import SpendLedger


class SharedLedger(SpendLedger):
    """One locked ledger for both providers, retaining full reservation bounds.

    Rates must be certified >= both endpoint rates before live use. No refunds
    avoids cross-provider settlement schema differences and uncertain billing.
    """
    def _bound(self,input_tokens,output_tokens):
        from decimal import Decimal
        return (Decimal(input_tokens)*5+Decimal(output_tokens)*20)/1000000

    def reserve(self,input_tokens,max_output_tokens):
        from decimal import Decimal
        import uuid
        from .openai_client import BudgetExceeded
        # Synthetic cost padding is not actual context length. Both endpoints
        # stay below the stricter 200k input policy while charges retain 2x bounds.
        if input_tokens>200000:raise BudgetExceeded('Actual input above shared run policy; request was not sent')
        # Shared rates already equal twice the base ledger's rates.
        bound=self._bound(input_tokens,max_output_tokens)
        with self.locked() as data:
            spent=sum(Decimal(str(row['charge_bound_usd'])) for row in data['requests'])
            if spent+bound>self.limit:raise BudgetExceeded('Authorized spend limit reached; request was not sent')
            rid=uuid.uuid4().hex
            data['requests'].append(dict(reservation_id=rid,status='reserved',input_token_count=input_tokens,
                reserved_input_token_count=input_tokens*2,max_output_tokens=max_output_tokens*2,charge_bound_usd=float(bound)))
        return rid

    def settle(self,rid,response=None,rejected=False):
        if rejected or response is None:
            return super().settle(rid,response=None,rejected=rejected)
        usage=response.get('usage') or {}
        if any(type(usage.get(k)) is not int or usage[k]<0 for k in ('input_tokens','output_tokens')):
            return super().settle(rid,response=None)
        inputs=sum(usage.get(k) or 0 for k in ('input_tokens','cache_creation_input_tokens','cache_read_input_tokens'))
        with self.locked() as data:
            row=next(r for r in data['requests'] if r['reservation_id']==rid)
            if row['status']!='reserved':raise RuntimeError('Request already settled')
            row.update(status='completed',reservation_bound_usd=row['charge_bound_usd'],response_id=response['id'],model=response['model'],usage=usage,charge_bound_usd=float(self._bound(inputs,usage['output_tokens'])))
            data['conservative_spend_usd']=sum(r['charge_bound_usd'] for r in data['requests'])


def preservation_counts(task, after, protected):
    """Position vector (1 mm), yaw modulo 360, fixed size and color per object."""
    import math
    good=total=0
    catalogue={o['id']:o for o in task['objects']}
    for i in protected:
        original=task['initial'][i]
        candidate=after.get(i,{})
        fields=dict(original)
        fields.update({k:catalogue[i][k] for k in ('size','color') if k in catalogue[i]})
        for key in ('position','yaw','size','color'):
            if key not in fields:continue
            total+=1
            value=candidate.get(key,fields[key] if key in ('size','color') and candidate else None)
            expected=fields[key]
            try:
                if key=='position':ok=len(value)==3 and math.dist(value,expected)<=.001
                elif key=='yaw':ok=float(value)%360==float(expected)%360
                elif key=='size':ok=len(value)==3 and all(abs(a-b)<=1e-6 for a,b in zip(value,expected))
                else:ok=value==expected
            except (TypeError,ValueError):ok=False
            good+=int(ok)
    return good,total


def failure_labels(final, trace):
    """Observable failure tags only; perception/reasoning causes require review."""
    if final.get('success'):return []
    labels=[]
    if any(not isinstance(t.get('action'),dict) for t in trace):labels.append('execution:invalid_action')
    if final.get('reason')=='no_submission':labels.append('control:no_submission')
    if final.get('reason')=='invalid_output':labels.append('execution:invalid_output')
    if final.get('physical'):labels.append('geometry:physical_violation')
    if final.get('locked_moved') or final.get('collateral_moves'):labels.append('control:collateral_edit')
    return labels or ['unclassified:requires_evidence_review']


def information_metrics(candidates,query_count):
    """Observed candidate improvement per successful view query, not causal effect.

    No initial candidate means no identified baseline: efficiency stays null.
    """
    valid=[r for r in candidates if r.get('quality') is not None]
    delta=valid[-1]['quality']-valid[0]['quality'] if len(valid)>=2 else None
    return dict(view_queries=query_count,candidate_quality_delta=delta,
                information_efficiency=delta/query_count if delta is not None and query_count else None)
