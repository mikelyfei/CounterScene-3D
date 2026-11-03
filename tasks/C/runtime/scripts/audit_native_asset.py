"""Owned Blender-only audit: native-proportion GLB vs solid-envelope controls."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def mesh_data(objects):
    vertices=[];faces=[]
    depsgraph=bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        evaluated=obj.evaluated_get(depsgraph);mesh=evaluated.to_mesh()
        start=len(vertices);vertices.extend(evaluated.matrix_world@v.co for v in mesh.vertices)
        mesh.calc_loop_triangles()
        faces.extend(tuple(start+i for i in triangle.vertices) for triangle in mesh.loop_triangles)
        evaluated.to_mesh_clear()
    return vertices,faces


def bounds(vertices):return [[min(v[i] for v in vertices),max(v[i] for v in vertices)] for i in range(3)]


def overlap_boxes(a,b):return all(min(a[i][1],b[i][1])-max(a[i][0],b[i][0])>1e-6 for i in range(3))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    asset=ROOT/'assets/kenney_furniture/table.glb'
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(asset))
    meshes=[o for o in bpy.data.objects if o.type=='MESH']
    verts,faces=mesh_data(meshes);native=bounds(verts)
    scale=.8/(native[2][1]-native[2][0]);centre=Vector([(native[0][0]+native[0][1])/2,(native[1][0]+native[1][1])/2,native[2][0]])
    # Bake one uniform transform; preserve native proportions and actual holes.
    for obj in meshes:
        matrix=obj.matrix_world.copy();obj.parent=None;obj.matrix_world.identity()
        for vertex in obj.data.vertices:vertex.co=(matrix@vertex.co-centre)*scale
    verts,faces=mesh_data(meshes);table_bounds=bounds(verts);table_bvh=BVHTree.FromPolygons(verts,faces)
    from counterscene.blender_mesh_checks import relation
    rows=[]
    for name,x,y,z in [('under_table_gap',.25,.25,.25),('through_tabletop',0,0,.8)]:
        bpy.ops.mesh.primitive_cube_add(size=.12,location=(x,y,z));probe=bpy.context.object
        pv,pf=mesh_data([probe]);pb=bounds(pv);hits=table_bvh.overlap(BVHTree.FromPolygons(pv,pf))
        rows.append(dict(case=name,aabb_overlap=overlap_boxes(table_bounds,pb),surface_intersection_pairs=len(hits),probe_centre=[x,y,z],solid_relation=relation(verts,faces,pv,pf)))
        bpy.data.objects.remove(probe,do_unlink=True)
    assert rows[0]['aabb_overlap'] and rows[0]['surface_intersection_pairs']==0,rows
    assert rows[1]['surface_intersection_pairs']>0,rows
    result=dict(asset=str(asset.relative_to(ROOT)),sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),blender=bpy.app.version_string,native_bounds=native,uniform_scale=scale,world_bounds=table_bounds,vertices=len(verts),polygons=len(faces),cases=rows,limits=['Surface intersection only; full containment and non-manifold semantics are not certified','Controls use a fixed table GLB, not all catalogue assets','Native proportions preserved with uniform scale'])
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
