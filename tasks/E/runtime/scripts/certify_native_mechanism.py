"""Verify actual rendered D mesh enclosure at every gold keyframe in Blender.

Continuous safety is checked independently in native_mechanism; this checks
that the render builders use precisely the primitives its bounds describe.
"""
import argparse,json,math,sys
from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from counterscene.native_mechanism import tool_geometry,CONNECTOR_RADIUS,VERIFIED_LINK_RADIUS
from counterscene.spatial_suite import kinematics as k
from scripts.spatial_scene_props import mechanism,ball_positions
from scripts.audit_native_asset import mesh_data


def check_snapshot(task,geo,mech,q,frame):
    bpy.ops.wm.read_factory_settings(use_empty=True);mechanism(geo,'gold',frame)
    points,wrist,end,toolbox=tool_geometry(mech,q);ballpos=ball_positions(geo,'gold',frame)
    counts={};maximum=0.
    for obj in list(bpy.data.objects):
        if obj.type!='MESH':continue
        name=obj.name;vertices,_=mesh_data([obj]);box=None;segment=None;radius=0
        if name.startswith('trajectory') or name.startswith('virtual ') or 'floor marking' in name:category='guide'
        elif name.startswith('fixed post'):category='post';segment=points[:2];radius=VERIFIED_LINK_RADIUS
        elif name.startswith('moving link 1') or name.startswith('joint 1'):category='link1';segment=points[1:3];radius=VERIFIED_LINK_RADIUS
        elif name.startswith('moving link 2') or name.startswith('joint 2'):category='link2';segment=points[2:4];radius=VERIFIED_LINK_RADIUS
        elif name.startswith('gripper wrist connector'):category='connector';segment=[wrist,end];radius=CONNECTOR_RADIUS
        elif name.startswith('gripper '):category='gripper';box=toolbox
        elif name.startswith('pedestal'):
            category='pedestal';x,y=mech['base'];box=[x-.125,x+.125,y-.125,y+.125,0,.025]
        elif name.startswith('released '):
            category='released_ball';record=next(o for o in task.get('resting_objects',[]) if o['id']==name[9:]);centre=record['position'];segment=[centre,centre];radius=record['radius_m']
        elif name.startswith('ball '):
            category='ball';bid=name[5:];centre=ballpos[bid] or points[3];segment=[centre,centre];radius=next(b['radius_m'] for b in task['balls'] if b['id']==bid)
        else:
            obstacle=next((o for o in task['obstacles'] if name.startswith(o['id']+' ')),None)
            if obstacle is None:raise AssertionError('Unclassified physical mesh '+name)
            category='fixture';box=obstacle['box']
        counts[category]=counts.get(category,0)+1
        obj['physical_category']=category
        if category=='guide':continue
        excess=max((max(max(box[2*i]-v[i],v[i]-box[2*i+1]) for i in range(3)) if box else k._seg_seg_dist(segment[0],segment[1],v,v)-radius) for v in vertices)
        maximum=max(maximum,excess)
        assert excess<2e-6,(frame,name,excess)
    return dict(keyframe=frame,mesh_counts=counts,max_enclosure_excess_m=maximum)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);rows=[]
    for folder in sorted(Path(args.source).glob('D-*')):
        private=json.loads((folder/'private.json').read_text());task=private['task_internal'];gold=private['gold'];mech=gold['mechanism']
        geo=dict(gold=[k.fk(mech,q) for q in gold['keyframes']],frames_per_segment=1,events=gold['events'],balls=task['balls'],obstacles=task['obstacles'],waypoints=task['waypoints'],radius=k.RADIUS)
        for frame,q in enumerate(gold['keyframes']):
            row=check_snapshot(task,geo,mech,q,frame)
            rows.append(dict(instance=folder.name,**row))
        print(folder.name,'actual mesh enclosure PASS',flush=True)
    Path(args.out).write_text(json.dumps(dict(rows=rows,scope='Every gold keyframe actual evaluated vertex; continuous envelope safety recorded separately; guides excluded'),indent=2))


if __name__=='__main__':main()
