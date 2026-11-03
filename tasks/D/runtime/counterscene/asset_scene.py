"""Validated native asset instance transforms; Blender handles actual geometry."""
import copy,math
from pathlib import Path
from .errors import ArtifactExecutionError

PROCEDURAL={'proc_bookcase','proc_storage','proc_stool','proc_soccer','proc_rugby','proc_column'}
ASSETS=Path(__file__).resolve().parents[1]/'assets/kenney_furniture'


def blender_binary():
    import os
    configured=os.environ.get('COUNTERSCENE_BLENDER')
    path=Path(configured).expanduser() if configured else Path(__file__).resolve().parents[1]/'.tools/blender/blender-4.5.15-windows-x64/blender.exe'
    if not path.is_absolute():raise ValueError('COUNTERSCENE_BLENDER must be an absolute executable path')
    if not path.is_file():raise FileNotFoundError('Blender 4.5 executable unavailable: '+str(path))
    return path.resolve()


def verify_artifacts(folder,rows):
    import hashlib
    folder=Path(folder).resolve()
    for row in rows:
        path=(folder/row['file']).resolve()
        if not path.is_relative_to(folder) or not path.is_file():raise ArtifactExecutionError('Missing or out-of-scope native artifact')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ArtifactExecutionError('Native artifact content hash differs')


def support_descendants(task,hosts):
    reached=set(hosts);original=set(hosts)
    while True:
        children={c['args'][0].split('#')[0] for c in task['constraints'] if c['type'] in ('on_top_of','on_shelf') and c['args'][1].split('#')[0] in reached}-reached
        if not children:return reached-original
        reached.update(children)


def vector(value):
    if not isinstance(value,(list,tuple)) or len(value)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>100 for x in value):raise ValueError('Expected three finite coordinates bounded to +/-100 metres')
    return list(value)


class AssetScene:
    def __init__(self,instances):
        if not isinstance(instances,dict) or not instances:raise ValueError('Nonempty instance map required')
        self.instances={}
        for oid,item in instances.items():
            asset=item.get('asset')
            if not isinstance(oid,str) or not isinstance(asset,str) or asset not in ({p.stem for p in ASSETS.glob('*.glb')}|PROCEDURAL):raise ValueError('Unknown asset or instance ID')
            height=item.get('height')
            if type(height) not in (int,float) or not math.isfinite(height) or not .01<=height<=10:raise ValueError('Asset height outside allowed range')
            self.instances[oid]=dict(asset=asset,height=height,position=vector(item.get('position')),yaw=0)
            if asset in PROCEDURAL:
                size=vector(item.get('size'))
                if any(v<=0 for v in size) or abs(size[2]-height)>1e-6:raise ValueError('Invalid procedural size')
                self.instances[oid]['size']=size
            self.rotate(oid,item.get('yaw',0))

    def move(self,oid,position):
        if oid not in self.instances:raise ValueError('Unknown instance')
        self.instances[oid]['position']=vector(position)

    def rotate(self,oid,yaw):
        if oid not in self.instances or type(yaw) not in (int,float) or not math.isfinite(yaw):raise ValueError('Invalid rotation')
        self.instances[oid]['yaw']=yaw%360

    def snapshot(self):return copy.deepcopy(self.instances)


def evaluate_scene(scene,out,render=False,room=None,camera=None,extra_cameras=None,pixel_evidence=False):
    """Inspect/check/render actual instances in isolated, bounded, hidden Blender."""
    import json,subprocess
    out=Path(out).resolve()
    if out.exists():raise ValueError('Artifact directory exists')
    out.parent.mkdir(parents=True,exist_ok=True)
    request=out.with_suffix('.request.json');log=out.with_suffix('.blender.log')
    if request.exists() or log.exists():raise ValueError('Attempt evidence already exists')
    request.write_text(json.dumps(dict(instances=scene.snapshot(),render=render,room=room or [4,4,3],camera=camera,extra_cameras=extra_cameras,pixel_evidence=pixel_evidence),indent=2),encoding='utf-8')
    root=Path(__file__).resolve().parents[1]
    with log.open('w',encoding='utf-8') as handle:
        try:completed=subprocess.run([str(blender_binary()),'--background','--factory-startup','--threads','4','--python-exit-code','1','--python',str(root/'scripts/native_asset_scene_worker.py'),'--','--request',str(request),'--out',str(out)],stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=120)
        except (subprocess.SubprocessError,OSError) as error:raise ArtifactExecutionError('Blender could not complete; inspect retained log') from error
    if completed.returncode:raise ArtifactExecutionError('Native scene worker failed; inspect retained log')
    if not (out/'result.json').exists():raise ArtifactExecutionError('Native scene worker failed; inspect retained log')
    return json.loads((out/'result.json').read_text(encoding='utf-8'))

MODEL_BY_ID={'cabinet_blue':'kitchencabinet','table_green':'table','lamp_white':'lamproundfloor','chair_cyan':'chair','plant_pink':'pottedplant','tv_black':'televisionmodern','desk_blue':'desk','deskchair_black':'chairdesk','monitor_gray':'computerscreen','laptop_gray':'laptop','sidetable_brown':'sidetable','flowerpot_green':'plantsmall1'}
PROCEDURAL_BY_ID={'shelf_orange':'proc_bookcase','stool_purple':'proc_stool','soccer_white':'proc_soccer','rugby_brown':'proc_rugby','column_cyan':'proc_column'}


def instance_specs(task,placements):
    """Complete catalogue mapping; dimensions remain task-owned, not agent-owned."""
    if not isinstance(placements,dict) or set(placements)!={o['id'] for o in task['objects']}:raise ValueError('Exact task instance coverage required')
    result={}
    for obj in task['objects']:
        oid=obj['id'];p=placements[oid]
        if not isinstance(p,dict) or set(p)-{'position','yaw','size'}:raise ValueError('Only task instance transforms and permitted size are accepted')
        size=vector(p.get('size',obj['size']))
        if obj.get('size_range'):
            lo,hi=obj['size_range']
            if any(not lo[i]<=size[i]<=hi[i] for i in range(3)):raise ValueError('Size outside task range')
            factor=size[2]/obj['size'][2]
            # Public nominal dimensions are rounded to millimetres; only height drives real uniform scaling.
            if any(abs(size[i]-obj['size'][i]*factor)>.001 for i in range(3)):raise ValueError('Native asset scaling must preserve proportions')
        elif size!=obj['size']:raise ValueError('Task-owned scale is immutable')
        asset=MODEL_BY_ID.get(oid,PROCEDURAL_BY_ID.get(oid,'proc_storage'))
        yaw=p.get('yaw',0)
        if type(yaw) not in (int,float) or not math.isfinite(yaw) or yaw%360 not in (0,90,180,270):raise ValueError('Task yaw must be a right angle')
        item=dict(asset=asset,height=size[2],position=vector(p['position']),yaw=yaw%360)
        if asset in PROCEDURAL:item['size']=size
        result[oid]=item
    return result
