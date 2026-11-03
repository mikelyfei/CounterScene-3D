"""Execute a restricted candidate against immutable real asset instances."""
import hashlib,json
from pathlib import Path
from .agentic_program import execute_candidate
from .asset_scene import AssetScene,instance_specs,evaluate_scene
from .native_scoring import check_native_scene


def retained_source_matches(path,source):
    raw=Path(path).read_bytes()
    # Legacy Windows TextIO translated LF even inside existing CRLF sequences.
    return raw in (source.encode('utf-8'),source.replace('\n','\r\n').encode('utf-8'))


def save_native_candidate(task,source,out,render=True,extra_cameras=None,pixel_evidence=False):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'candidate.py').write_bytes(source.encode('utf-8'))
    result=execute_candidate(source,task['room'])
    result['source_sha256']=hashlib.sha256(source.encode()).hexdigest()
    result['task_sha256']=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()
    if result['executed']:
        try:scene=AssetScene(instance_specs(task,result['payload']))
        except (ValueError,KeyError,TypeError) as error:
            result.update(executed=False,error=str(error)[:300])
        else:
            (out/'placements.json').write_text(json.dumps(result['payload'],indent=2),encoding='utf-8')
            mesh=evaluate_scene(scene,out/'scene',render=render,room=task['room'],camera=task.get('camera'),extra_cameras=extra_cameras,pixel_evidence=pixel_evidence)
            result['geometry']=check_native_scene(task,result['payload'],mesh)
            result['scene_artifact']=dict(mesh['scene_artifact'],file='scene/'+mesh['scene_artifact']['file'])
            result['renders']=[dict(row,file='scene/'+row['file']) for row in mesh['renders']]
    (out/'execution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
