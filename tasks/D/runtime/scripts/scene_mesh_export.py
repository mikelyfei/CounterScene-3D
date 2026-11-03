"""Export evaluated Blender meshes for the local native WebGL reviewer."""
import array,base64,json
def export(scene,folder,scene_id,filename='scene.js',phase=None,visible_only=False):
 import bpy
 values=array.array('f');parts=[];deps=bpy.context.evaluated_depsgraph_get()
 for obj in scene.objects:
  if obj.type not in ('MESH','FONT') or visible_only and obj.hide_render:continue
  ev=obj.evaluated_get(deps);mesh=ev.to_mesh();mesh.calc_loop_triangles();matrix=ev.matrix_world;nm=matrix.to_3x3().inverted().transposed();start=len(values)//9
  for tri in mesh.loop_triangles:
   material=mesh.materials[tri.material_index] if len(mesh.materials)>tri.material_index else None
   color=list(material.diffuse_color[:3]) if material else [.5]*3
   if material and material.use_nodes:
    node=material.node_tree.nodes.get('Principled BSDF')
    if node:color=list(node.inputs['Base Color'].default_value[:3])
   for vi in tri.vertices:
    v=mesh.vertices[vi];values.extend([*(matrix@v.co),*(nm@v.normal).normalized(),*color])
  parts.append(dict(name=obj.name,start=start,count=len(values)//9-start,role=obj.get('b25_role',obj.get('role','environment')),min_level=obj.get('min_level',1),event_only=bool(obj.get('event_only',False))))
  ev.to_mesh_clear()
 payload=dict(id=scene_id,phase=phase,parts=parts,vertex_count=len(values)//9,buffer=base64.b64encode(values.tobytes()).decode())
 (folder/filename).write_text('window.onBenchmarkScene('+json.dumps(payload,separators=(',',':'))+');',encoding='utf-8')
 return parts,len(values)//9
