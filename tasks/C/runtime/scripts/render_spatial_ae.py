"""A-E renderer: preserve analytic task geometry; reuse local CC0 furniture for review."""
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bpy
from mathutils import Vector
from counterscene.spatial_suite import blender_render as br,grounding

# Kenney CC0 meshes fitted into the scored boxes. The bookcase (scored parts), stool, column and balls are procedural.
MODELS={'cabinet_blue':'kitchenCabinet','table_green':'table','lamp_white':'lampRoundFloor','chair_cyan':'chair','plant_pink':'pottedPlant','tv_black':'televisionModern',
        'desk_blue':'desk','deskchair_black':'chairDesk','monitor_gray':'computerScreen','laptop_gray':'laptop','sidetable_brown':'sideTable','flowerpot_green':'plantSmall1'}


def sphere(name,p,r,mat):
 bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=r,location=p);o=bpy.context.object;o.name=name;o.data.materials.append(mat)
 return o


def rod(name,a,b,r,mat):
 a,b=Vector(a),Vector(b)
 if (b-a).length<1e-8:return
 bpy.ops.mesh.primitive_cylinder_add(vertices=20,radius=r,depth=(b-a).length,location=(a+b)/2);o=bpy.context.object;o.name=name;o.rotation_euler=(b-a).to_track_quat('Z','Y').to_euler();o.data.materials.append(mat)


def checker_floor(room):
 m=bpy.data.materials.new('MAT_Scored_Checker');m.use_nodes=True;nt=m.node_tree;p=nt.nodes['Principled BSDF']
 tex=nt.nodes.new('ShaderNodeTexChecker');geo=nt.nodes.new('ShaderNodeNewGeometry');nt.links.new(geo.outputs['Position'],tex.inputs['Vector']);tex.inputs['Scale'].default_value=2
 tex.inputs['Color1'].default_value=(.36,.39,.4,1);tex.inputs['Color2'].default_value=(.62,.65,.64,1)
 nt.links.new(tex.outputs['Color'],p.inputs['Base Color']);p.inputs['Roughness'].default_value=.8
 bpy.data.objects['floor'].data.materials.clear();bpy.data.objects['floor'].data.materials.append(m)


def main():
 p=argparse.ArgumentParser();p.add_argument('--instance',required=True);p.add_argument('--out',required=True);p.add_argument('--samples',type=int,default=16);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
 source=Path(a.instance);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 geo=json.loads((source/'review_geometry.json').read_text(encoding='utf-8'));private=json.loads((source/'private.json').read_text(encoding='utf-8'));task=private['task_internal'];public=json.loads((source/'public.json').read_text(encoding='utf-8'));exp=public['experiment']
 W,D,H=geo['room'];rows=[];br.SAMPLES=a.samples
 states=['gold']+(['initial'] if 'initial' in geo else [])+(['overfit'] if 'overfit' in geo else [])
 if geo['kind']=='arm':states=['gold','static']
 for state in states:
  for mode in (['mechanism'] if geo['kind']=='arm' else ['proxy','assets']):
   bpy.ops.wm.read_factory_settings(use_empty=True);walls=br.build_room(W,D,H);br.lights(W,D,H)
   if mode in ('proxy','mechanism'):checker_floor(geo['room'])
   from scripts.spatial_scene_props import room_objects,mechanism
   if mode=='mechanism':mechanism(geo,state)
   else:room_objects(geo,state,mode)
   cams=grounding.cameras(geo['room'],['reference','front_left','back'])
   if exp=='C' and state=='gold' and mode=='assets':cams=grounding.cameras(geo['room'],list(grounding.PUBLIC_VIEWS)+list(grounding.HIDDEN_VIEWS))
   # Only public camera images may accompany the exact scored diagram in model input.
   for view,camera in cams.items():
    if view=='back' and exp not in ('C','D'):continue
    br.setup_render(640,480);br.camera(camera)
    for which,objects in walls.items():
     for obj in objects:obj.hide_render=br.blocks(camera,W,D,which)
    file=out/f'{state}_{mode}_{view}.png';bpy.context.scene.render.filepath=str(file);bpy.ops.render.render(write_still=True)
    rows.append(dict(state=state,mode=mode,view=view,file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),width=640,height=480))
 manifest=dict(instance=public['instance'],experiment=exp,blender=bpy.app.version_string,samples=a.samples,images=rows,
  source_hash=hashlib.sha256((source/'review_geometry.json').read_bytes()).hexdigest(),review_only=True,
  geometry='Proxy renders use exact scorer boxes/capsules; furniture renders are fitted visualizations, not mesh scorers.',
  asset_provenance=str(ROOT/'assets/kenney_furniture/provenance.json'),limits=['Motion image shows a midpoint and path; continuous certification comes from scorer','C public-camera appearance may supplement, never replace, its exact scored diagram; hidden renders remain private'])
 (out/'render_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print(json.dumps(dict(instance=public['instance'],renders=len(rows),complete=True)),flush=True)
if __name__=='__main__':main()
