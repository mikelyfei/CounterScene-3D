"""Export real asset triangles for the A-E main viewport; no model/API calls."""
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bpy
from mathutils import Vector
from counterscene.spatial_suite import blender_render as br
from scripts.render_spatial_ae import MODELS,rod,sphere
from scripts.scene_mesh_export import export


def storage(oid,b,color):
 sx,sy,sz=b[1]-b[0],b[3]-b[2],b[5]-b[4];x,y,z=(b[0]+b[1])/2,(b[2]+b[3])/2,b[4]
 mat=br.principled(oid+' painted shell',color,.55);trim=br.principled(oid+' edge fittings',(.13,.16,.17),.35,.55)
 thick=min(sx,sy,sz)*.08
 # A built storage prop: separate shell, lid, edge bands and recessed front grip.
 br.box_mesh(oid+' shell',(sx,sy,sz*.88),(x,y,z),mat,bevel=thick*.35)
 br.box_mesh(oid+' lid',(sx,sy,sz*.12),(x,y,z+sz*.88),trim,bevel=thick*.25)
 for dx in (-sx*.38,sx*.38):br.box_mesh(oid+' band',(thick,sy,sz),(x+dx,y,z),trim,bevel=thick*.2)
 br.box_mesh(oid+' grip',(sx*.28,thick,sz*.10),(x,y-sy/2+thick*.6,z+sz*.54),trim,bevel=thick*.2)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(a.root)
 records=[]
 for folder in sorted((root/'instances').iterdir()):
  if not (folder/'review_geometry.json').exists():continue
  geo=json.loads((folder/'review_geometry.json').read_text(encoding='utf-8'));room=geo['room'];W,D,H=room
  states=['gold']+(['initial'] if 'initial' in geo else [])+(['overfit'] if 'overfit' in geo else [])
  if geo['kind']=='arm':states=['gold','static']
  for state in states:
   bpy.ops.wm.read_factory_settings(use_empty=True);walls=br.build_room(W,D,H)
   # The outer backdrop is render scenery, never the floor or a fit target.
   bpy.data.objects.remove(bpy.data.objects['ground_outside'],do_unlink=True)
   for o in walls['left']:o.hide_render=True
   from scripts.spatial_scene_props import room_objects,mechanism
   if geo['kind']=='arm':points=mechanism(geo,state)
   else:room_objects(geo,state,'assets')
   bpy.context.view_layer.update();dest=root/'meshes'/folder.name;dest.mkdir(parents=True,exist_ok=True)
   filename=state+'.js';parts,count=export(bpy.context.scene,dest,folder.name,filename,state,True)
   records.append(dict(instance=folder.name,state=state,file='meshes/'+folder.name+'/'+filename,vertices=count,parts=len(parts),sha256=hashlib.sha256((dest/filename).read_bytes()).hexdigest(),arm_reference=points if geo['kind']=='arm' else None))
 (root/'mesh_manifest.json').write_text(json.dumps(dict(version='asset-viewport-1',scenes=records,review_only=True,limits=['Furniture is fitted to the analytic envelopes','Decorative arm housing is not a new collision model']),indent=2),encoding='utf-8')
 print(json.dumps(dict(mesh_states=len(records),complete=True)),flush=True)
if __name__=='__main__':main()
