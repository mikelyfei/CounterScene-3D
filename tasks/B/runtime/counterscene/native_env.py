"""Native A/B/E scene environment; no box render or box physical score path."""
import copy,hashlib,json,math
from pathlib import Path
from .agentic_program import execute_candidate
from .asset_scene import AssetScene,instance_specs,evaluate_scene,verify_artifacts,support_descendants
from .native_candidate import save_native_candidate,retained_source_matches
from .native_scoring import check_native_scene
from .errors import ArtifactExecutionError
from .spatial_suite import core,protocol,constraint_graph as cg
from .program import CONTRACT


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


class NativeSceneEnv(protocol.Env):
    def __init__(self,task,private,artifacts,max_turns=14):
        super().__init__(copy.deepcopy(task),copy.deepcopy(private),max_turns)
        self.experiment=task['experiment']
        if self.experiment not in 'ABCE':raise ValueError('Native scene environment requires A/B/C/E')
        self.artifacts=Path(artifacts).resolve();self.artifacts.mkdir(parents=True,exist_ok=True)
        self.executions={};self.observation_artifacts=[]
        self.feedback_condition='structured'
        self.capture_cache=None

    def materialize(self,source):
        if not isinstance(source,str):return dict(executed=False,payload=None,error='source must be a string')
        key=digest(dict(task=self.task,source=source))
        if key not in self.executions:
            out=self.artifacts/key
            if out.exists():
                retained=json.loads((out/'execution.json').read_text())
                if retained.get('task_sha256')!=digest(self.task) or retained.get('source_sha256')!=hashlib.sha256(source.encode()).hexdigest():raise ValueError('Retained candidate identity mismatch')
                if not retained_source_matches(out/'candidate.py',source):raise ValueError('Retained program differs')
                verify_artifacts(out,retained.get('renders',[])+([retained['scene_artifact']] if 'scene_artifact' in retained else []))
                self.executions[key]=retained
            else:
                cameras={**self.task['public_cameras'],**self.priv['hidden_cameras']} if self.experiment=='C' else None
                self.executions[key]=save_native_candidate(self.task,source,out,render=True,extra_cameras=cameras,pixel_evidence=self.experiment=='C')
            self.executions[key]['artifact_directory']=str(out)
        return self.executions[key]

    def check_placements(self,placements):
        return self.materialize('def build(public_params):\n    return '+repr(placements)).get('geometry',dict(valid_output=False,output_problems=['Invalid native instances'],constraints=[],physical=[],all_satisfied=False))

    def scene_capture(self,task,placements,label,root=None):
        specs=instance_specs(task,placements)
        key=digest(dict(label=label,instances=specs,room=task['room'],camera=task.get('camera')));out=(Path(root) if root else self.capture_cache or self.artifacts)/key
        try:
            if out.exists():mesh=json.loads((out/'result.json').read_text())
            else:mesh=evaluate_scene(AssetScene(specs),out,render=True,room=task['room'],camera=task.get('camera'))
            verify_artifacts(out,mesh['renders']+[mesh['scene_artifact']])
        except (OSError,ValueError,RuntimeError) as error:raise ArtifactExecutionError('Native scene capture failed; retained attempt: '+str(out)) from error
        images=[(out/row['file']).read_bytes() for row in mesh['renders']]
        metadata=dict(scene_sha256=mesh['scene_artifact']['sha256'],renders=mesh['renders'],instances={oid:{k:row[k] for k in ('asset','asset_sha256','position','yaw','uniform_scale','native_bounds','world_bounds')} for oid,row in mesh['instances'].items()})
        self.observation_artifacts.append(dict(label=label,directory=str(out),**metadata))
        return metadata,images,mesh

    def catalogue(self):
        columns=math.ceil(math.sqrt(len(self.task['objects'])));rows=math.ceil(len(self.task['objects'])/columns)
        atlas={o['id']:dict(position=[1.75+3.5*(n%columns),1.75+3.5*(n//columns),0],yaw=0) for n,o in enumerate(self.task['objects'])}
        task=dict(objects=self.task['objects'],room=[3.5*columns,3.5*rows,3],camera=None)
        # Both paired worlds receive one identical capture built from public catalogue data only.
        metadata,images,_=self.scene_capture(task,atlas,'asset_catalogue',root=self.capture_cache or self.artifacts.parent/'catalogues')
        return dict(layout=atlas,note='Isolated catalogue staging positions, not the task solution',**metadata),images

    def asset_capture(self,oid):
        obj=next((o for o in self.task['objects'] if o['id']==oid),None)
        if obj is None:return dict(error='Unknown catalogue instance')
        size=obj['size'];distance=max(size)*2.5;centre=[1.5,1.5,size[2]/2]
        camera=dict(eye=[centre[0]+.6*distance,centre[1]-.8*distance,centre[2]+.5*distance],target=centre,up=[0,0,1],fov_deg=35,width=640,height=480)
        task=dict(objects=[obj],room=[3,3,max(3,size[2]+1)],camera=camera)
        metadata,images,_=self.scene_capture(task,{oid:dict(position=[1.5,1.5,0],yaw=0)},'asset_detail_'+oid,root=self.capture_cache or self.artifacts.parent/'catalogues')
        return dict(result='Actual catalogue asset close-up; staging pose, not target',instance=oid,capture=dict(scene_sha256=metadata['scene_sha256'],renders=metadata['renders'][:1],instances=metadata['instances']),_images=images[:1])

    def prompt(self):
        atlas,images=self.catalogue()
        nominal={o['id']:dict(asset=instance_specs({'objects':[o]},{o['id']:dict(position=[0,0,0],yaw=0)})[o['id']]['asset'],size=o['size']) for o in self.task['objects']}
        types=[t for t in self.task['relation_types'] if t not in ('on_top_of','on_shelf','touching','occludes','appears_left_of','appears_larger_than')]
        extra=''
        if self.experiment=='B':extra='Satisfy all hard and jointly feasible soft constraints; identify exactly impossible soft IDs in infeasible [{id,reason}]. Preserve locked objects and minimise edits. Correctness of infeasibility claims is private.'
        if self.experiment=='E':extra='The check tool is deliberately unreliable: missing entries, wrong verdicts/rules and numeric noise may occur. measure gives actual geometry evidence. Report verdict faults in checker_issues [{id,issue,evidence}], issue=false_pass|false_fail|missing|wrong_rule. Physical pairs use id=no_interpenetration plus objects=[id,id]. Do not report small numeric noise. Preserve clean parts of the installation.'
        text=f'''Native-asset spatial task {self.task['instance']} ({self.experiment}). Manipulate these real asset instances by fixed IDs; they are not replaceable bounding-box meshes.
Frame: x right, y back, z up; metres. position=[centre_x,centre_y,bottom_z], finite coordinates within +/-100m; yaw in {{0,90,180,270}} degrees. Room={json.dumps(self.task['room'])}.
Assets preserve their source materials and native proportions. ID colour suffixes are opaque legacy names. Fixed asset identity and scale cannot change. Every ID must appear exactly once; no asset replacement, deletion or new instance.
Catalogue={json.dumps(nominal)}
The attached views show actual catalogue meshes, not a target solution. Atlas identity/pose legend={json.dumps(atlas)}
World-direction/metric relations use the declared catalogue envelopes and centres. Physical support, contact, room bounds and intersection use actual mesh evidence. Support requires the full projected footprint covered by upward host surfaces within 1cm; unknown topology is not a collision-free pass. touching requires a <=1cm actual surface-distance witness without excessive penetration. on_shelf requires the named actual board. Image relations use the supplied reference camera: actual 80x60 mesh-ray solo silhouette area, projected bounds centre, occludes>=0.5 pair-only occlusion. No force/stability simulation.
Other relation definitions:\n{protocol.relation_glossary(types)}
Reference camera={json.dumps(self.task.get('camera'))}
Scene policy={json.dumps(self.task.get('scene_policy',{}))}
Constraints:\n{protocol.constraint_lines(self.task['constraints'],with_priority=self.experiment=='B')}
Existing locked installation IDs={json.dumps(self.task.get('locked',[]))}. Its actual pose is withheld: inspect_scene before constructing a compatible scene. {extra}
Tools: inspect_asset(id) returns a close-up of that actual catalogue asset at a staging pose (no target information). inspect_scene returns actual live placements and mesh views (A: locked fixtures only). check executes source and reports constraints/physics (E: unreliable). measure executes source and returns actual bounds, surfaces and pair diagnostics. render executes source and returns actual mesh images. submit executes source and ends the stage irreversibly without revealing private correctness.
For check/measure/render/submit use {{"tool":"...","source":"def build(public_params): ..."}}. build returns the COMPLETE placements dictionary {{id:{{"position":[x,y,z],"yaw":0}}}}. Only room is in public_params. Restricted Python: no imports, attribute access, filesystem/network, or private task data. submit also accepts infeasible (B) or checker_issues (E). Valid but incorrect scenes still commit; use public tools first.
Program contract: {CONTRACT}
{self._budget()} {protocol.ACTION_RULE}'''
        return text,images

    def step(self,action):
        if not self.done and isinstance(action,dict) and action.get('tool') in ('check','measure','render','submit','compare'):
            source=action.get('source');executed=execute_candidate(source,self.task['room'])
            result=self.materialize(source) if executed['executed'] else executed
            if not result['executed'] or not result.get('geometry',{}).get('valid_output'):
                observation,images=super().step(None)
                observation['error']=result.get('error') or 'Invalid native instance output'
                return observation,images
            action=dict(action,placements=result['payload'])
        return super().step(action)

    def tool(self,action):
        tool=action['tool']
        if tool not in ('inspect_asset','inspect_scene','check','measure','render'):return dict(error='Unknown native scene tool')
        if tool=='inspect_asset':return self.asset_capture(action.get('id'))
        if tool=='inspect_scene':
            placements=self.task['initial']
            if self.experiment=='A' and not self.task.get('show_live_installation'):placements={oid:placements[oid] for oid in self.task['locked']}
            task=dict(self.task,objects=[o for o in self.task['objects'] if o['id'] in placements])
            metadata,images,_=self.scene_capture(task,placements,'live_installation')
            observation=dict(placements=copy.deepcopy(placements),capture=metadata,_images=images)
            if self.priv.get('physical_event'):
                observation['physical_event']=copy.deepcopy(self.priv['physical_event'])
            if self.priv.get('stage_work_order'):
                observation['new_work_order']=copy.deepcopy(self.priv['stage_work_order'])
                observation['work_order_note']='Additional mandatory installation requirements revealed by the live survey; preserve all original constraints.'
            if self.experiment=='B':
                from .part2 import feedback
                observation['report']=feedback(self.check_placements(self.task['initial']),self.feedback_condition)
            return observation
        execution=self.materialize(action.get('source'))
        if tool=='check':
            report=execution['geometry']
            return cg.noisy_check(self.task,self.priv['faults'],action['placements'],self.turn,report=report) if self.experiment=='E' else copy.deepcopy(report)
        if tool in ('render','measure'):
            root=Path(execution['artifact_directory']);mesh=json.loads((root/'scene/result.json').read_text())
            if tool=='render':return dict(result='Actual executed native scene',capture=dict(scene=execution['scene_artifact'],renders=execution['renders']),_images=[(root/row['file']).read_bytes() for row in execution['renders']])
            return dict(instances=mesh['instances'],supports=mesh['support_witnesses'],pairs=mesh['pairs'],reference_view=mesh.get('reference_view'))
        return dict(error='Unknown native tool')

    def final(self):
        if not self.submission:return dict(success=False,reason='no_submission')
        execution=self.materialize(self.submission['source']);report=execution['geometry'];placements=execution['payload']
        if self.experiment=='B':
            claimed=[v.get('id') if isinstance(v,dict) else v for v in self.submission.get('infeasible',[])]
            result=cg.score_b(self.task,self.priv,placements,claimed,report=report)
        elif self.experiment=='E':
            result=cg.score_e(self.task,self.priv,placements,self.submission.get('checker_issues',[]),report=report)
            possible=set(result['clean_disturbed'])&support_descendants(self.task,self.task.get('locked',[]))
            forced=set()
            if possible:
                initial=self.check_placements(self.task['initial'])
                unsupported={oid for row in initial['physical'] if row['check'] in ('supported','approved_full_footprint_support') for oid in row['objects']}
                still_bad={oid for row in report['physical'] if row['check'] in ('supported','approved_full_footprint_support') for oid in row['objects']}
                forced=possible&unsupported-still_bad
            result['forced_support_repairs']=sorted(forced)
            result['unnecessary_clean_disturbed']=sorted(set(result['clean_disturbed'])-forced)
        else:result=cg.score_a(self.task,placements,report=report)
        moved=set(core.edit_cost(self.task['initial'],placements)['moved'])&set(self.task.get('locked',[]))
        result.update(locked_moved=sorted(moved),scene_artifact=execution.get('scene_artifact'),artifact_directory=execution.get('artifact_directory'))
        result['success']=result['success'] and not moved
        return result
