"""Active native C observations with evaluator-only hidden masks and target poses."""
import copy,json
from pathlib import Path
from .native_env import NativeSceneEnv,digest
from .asset_scene import AssetScene,evaluate_scene,instance_specs,verify_artifacts
from .native_grounding import compare_views,score_grounding
from .errors import ArtifactExecutionError
from .spatial_suite import grounding,protocol
from .program import CONTRACT
from .three_views import three_view_cameras


class NativeGroundingEnv(NativeSceneEnv):
    def __init__(self,task,private,artifacts,max_turns=14):
        task=copy.deepcopy(task);task['public_cameras']=grounding.cameras(task['room'],['reference']+task['queryable_views'])
        if task.get('observation_contract')=='three-view-v1':
            task['public_cameras'].update(three_view_cameras(task['room']))
            task['initial_view_names']=task.get('initial_view_names',['front','side','top'])
        super().__init__(task,private,artifacts,max_turns);self.held=[]
        self.view_condition='active'

    def prompt(self):
        atlas,images=self.catalogue()
        objects={o['id']:{key:o[key] for key in ('size','size_range','size_options') if key in o} for o in self.task['objects']}
        text=f'''Native-asset active reconstruction {self.task['instance']}. View condition={self.view_condition}; fixed condition disables query_view. Reconstruct the live target using the same real asset instances shown in the attached catalogue atlas. Atlas staging is not the target: {json.dumps(atlas)}.
Fixed IDs, source materials and native proportions must remain. ID colour suffixes are opaque names. No deletion/replacement/new object. Sizes and permitted uncertainty={json.dumps(objects)}. Reconstructed size must use one uniform scale (nominal dimensions have 1mm rounding); height controls real mesh scale.
Frame: x right, y back, z up, metres; position=[centre_x,centre_y,bottom_z], finite coordinates within +/-100m; yaw=0/90/180/270 degrees. Room={json.dumps(self.task['room'])}.
Target image is not acquired yet. capture_reference obtains the actual reference. Then query_view(view) obtains one named public view; at most {self.task['query_budget']} distinct extra views. Cameras={json.dumps(self.task['public_cameras'])}. Images are real material meshes; no proxy-box input.
Source actions use restricted Python def build(public_params): ... returning COMPLETE {{id:{{position:[x,y,z],yaw:0,size:[x,y,z]}}}}. Only room is public_params. Imports, attribute/file/network/private-task access are forbidden.
Tools: {{"tool":"inspect_asset","id":"instance"}} gives an actual asset close-up at a public staging pose (no target information); {{"tool":"capture_reference"}}; {{"tool":"query_view","view":"name"}}; {{"tool":"render","source":"...","view":"public name"}} renders your candidate; {{"tool":"compare","source":"..."}} returns one mean actual-mesh silhouette IoU per acquired view only; {{"tool":"submit","source":"..."}} irrevocably executes the output. No hidden acceptance/retry signal.
All actual mesh physical/support/room rules must hold; projected footprint needs actual upward support surfaces within 1cm; unknown intersection is not a pass. Success additionally requires hidden-camera mean per-object visible silhouette IoU >=0.8 at 160x120 actual triangle rays AND every actual bounds centre within 0.05m of the target. Hidden cameras, target transforms and hidden score are evaluator-only. No dynamics/stability simulation.
Program contract: {CONTRACT}
{self._budget()} {protocol.ACTION_RULE}'''
        if self.task.get('observation_contract')=='three-view-v1':
            observation,target_images=self.observe_stage()
            text=text.replace('Target image is not acquired yet.','Initial calibrated target views are attached after the catalogue.')
            text+='\nCurrent stage target observations='+json.dumps(observation)+'\nAfter a target update, use newly supplied stage views to revise the preceding reconstruction. Preserve still-correct objects. Three perpendicular views are calibrated perspective images, not orthographic projections.'
            images=images+target_images
        return text,images

    def observe_stage(self):
        names=self.task.get('initial_view_names',['reference']);rows=[];images=[]
        for name in names:
            if name not in self.task['public_cameras']:raise ValueError('Unknown stage observation camera')
            observation=self.capture(name);captured=observation['_images'];rows.append(dict(view=name,camera=self.task['public_cameras'][name],capture=observation['capture']))
            images.extend(captured)
            if name not in self.held:self.held.append(name)
        return dict(stage_views=rows,scope='Actual current target RGB and public camera calibration only'),images

    def target_evidence(self):
        cameras={**self.task['public_cameras'],**self.priv['hidden_cameras']}
        out=self.artifacts/('private_target_'+digest(dict(placements=self.priv['gold'],cameras=cameras)))
        if out.exists():
            try:result=json.loads((out/'result.json').read_text());verify_artifacts(out,[result['scene_artifact']]);return result
            except (OSError,ValueError) as error:raise ArtifactExecutionError('Retained native target evidence unavailable') from error
        return evaluate_scene(AssetScene(instance_specs(self.task,self.priv['gold'])),out,room=self.task['room'],extra_cameras=cameras,pixel_evidence=True)

    def score_placements(self,placements):
        execution=self.materialize('def build(public_params):\n    return '+repr(placements))
        if not execution.get('geometry',{}).get('valid_output'):return dict(success=False,reason='invalid_output')
        mesh=json.loads((Path(execution['artifact_directory'])/'scene/result.json').read_text())
        return score_grounding(self.task,self.priv,placements,mesh,self.target_evidence(),self.held)

    def capture(self,view):
        task=dict(self.task,camera=self.task['public_cameras'][view])
        metadata,images,_=self.scene_capture(task,self.priv['gold'],'target_public_'+view)
        # The requested RGB view is public; target transforms, scale and other rendered views are not.
        capture=dict(scene_sha256=metadata['scene_sha256'],renders=metadata['renders'][:1])
        return dict(result='Actual target capture '+view,capture=capture,_images=images[:1])

    def tool(self,action):
        tool=action['tool']
        if tool=='inspect_asset':return self.asset_capture(action.get('id'))
        if tool=='capture_reference':
            observation=self.capture('reference')
            if 'reference' not in self.held:self.held.insert(0,'reference')
            return observation
        if tool=='query_view':
            if self.view_condition=='fixed':return dict(error='Additional target views disabled in fixed-view condition')
            view=action.get('view')
            if not self.held:return dict(error='Acquire capture_reference first')
            if view not in self.task['queryable_views']:return dict(error='Unknown public view')
            initial=set(self.task.get('initial_view_names',['reference']))
            if view not in self.held and len(set(self.held)-initial)>=self.task['query_budget']:return dict(error='Query budget exhausted')
            observation=self.capture(view)
            if view not in self.held:self.held.append(view)
            return observation
        if tool=='compare':
            if not self.held:return dict(error='No acquired target view')
            execution=self.materialize(action['source']);mesh=json.loads((Path(execution['artifact_directory'])/'scene/result.json').read_text());target=self.target_evidence()
            return {view:dict(mean_object_iou=round(compare_views(target['view_evidence'][view],mesh['view_evidence'][view])['mean_object_iou'],3)) for view in self.held}
        if tool=='render':
            view=action.get('view')
            if view not in self.task['public_cameras']:return dict(error='Unknown public view')
            metadata,images,_=self.scene_capture(dict(self.task,camera=self.task['public_cameras'][view]),action['placements'],'candidate_public_'+view)
            return dict(result='Actual candidate '+view,capture=dict(scene_sha256=metadata['scene_sha256'],renders=metadata['renders'][:1]),_images=images[:1])
        return dict(error='Unknown native reconstruction tool')

    def final(self):
        if not self.submission:return dict(success=False,reason='no_submission')
        execution=self.materialize(self.submission['source'])
        if not execution.get('geometry',{}).get('valid_output'):return dict(success=False,reason='invalid_output')
        mesh=json.loads((Path(execution['artifact_directory'])/'scene/result.json').read_text())
        result=score_grounding(self.task,self.priv,execution['payload'],mesh,self.target_evidence(),self.held)
        if result.get('reason')=='invalid_native_target_evidence':raise ArtifactExecutionError('Native private target verification failed')
        result.update(artifact_directory=execution['artifact_directory'],scene_artifact=execution['scene_artifact'])
        return result
