"""Render/export executed physical D motion, never benchmark gold coordinates."""
import argparse,hashlib,json,sys
from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.certify_native_mechanism import check_snapshot
from counterscene.native_mechanism import executed_ball_timeline
from counterscene.spatial_suite import kinematics as k,grounding,blender_render as br


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);request=json.loads(Path(args.request).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    task=request['task'];sub=request['physical'];mech=sub['mechanism']
    geo=dict(gold=[k.fk(mech,q) for q in sub['keyframes']],frames_per_segment=1,events=sub['events'],balls=task['balls'],obstacles=task['obstacles'],waypoints=task['waypoints'],radius=k.RADIUS,executed_ball_positions=executed_ball_timeline(task,sub))
    geo['resting_objects']=task.get('resting_objects',[])
    frames=[];renders=[]
    selected=0 if request.get('preview') else ((sub['events'][0]['keyframe']+sub['events'][1]['keyframe'])//2 if len(sub['events'])>=2 else len(sub['keyframes'])//2)
    for frame,q in enumerate(sub['keyframes']):
        evidence=check_snapshot(task,geo,mech,q,frame)
        for obj in list(bpy.data.objects):
            if obj.get('physical_category')=='guide':bpy.data.objects.remove(obj,do_unlink=True)
        path=out/f'frame_{frame:02d}.glb'
        bpy.ops.export_scene.gltf(filepath=str(path),export_format='GLB',export_extras=True)
        frames.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),physical_q=q,enclosure=evidence,ball_positions=geo['executed_ball_positions'][frame]))
        if frame==selected:
            room=task['room'];br.SAMPLES=8;br.lights(*room)
            br.box_mesh('floor',(room[0],room[1],.02),(room[0]/2,room[1]/2,-.02),br.principled('floor',(.35,.37,.39),.8),0)
            for view,camera in grounding.cameras(room,['reference','front_left','back']).items():
                br.setup_render(640,480);br.camera(camera);image=out/(view+'.png');bpy.context.scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
                renders.append(dict(file=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest(),camera=camera,keyframe=frame,raster_width=640,raster_height=480))
    result=dict(frames=frames,renders=renders,scene_artifact=frames[selected],source_geometry_sha256=hashlib.sha256((ROOT/'scripts/spatial_scene_props.py').read_bytes()).hexdigest(),scope='Actual executed physical keyframes and meshes; continuous enclosure collision certificate supplied separately; preview is a public template, not a solution')
    (out/'result.json').write_text(json.dumps(result,indent=2));print('Native physical motion frames exported',len(frames),flush=True)


if __name__=='__main__':main()
