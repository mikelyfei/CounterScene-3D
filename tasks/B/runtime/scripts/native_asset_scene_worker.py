"""Native GLB scene worker. Run in bundled Blender, not the model interpreter."""
import argparse,hashlib,itertools,json,math,sys
from pathlib import Path
import bpy
from mathutils import Matrix,Vector
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from counterscene.asset_scene import AssetScene,ASSETS,PROCEDURAL
from counterscene.blender_mesh_checks import relation,solid_components,compound_relation
from counterscene.spatial_suite import blender_render as br,grounding
from scripts.audit_native_asset import mesh_data,bounds
from counterscene.support_coverage import rectangle_coverage,contact_certificate


def load_instance(oid,item):
    procedural=item['asset'] in PROCEDURAL
    path=ROOT/'scripts/spatial_scene_props.py' if procedural else ASSETS/(item['asset']+'.glb')
    before=set(bpy.data.objects)
    if not procedural:bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        from scripts.spatial_scene_props import stool,soccer_ball,rugby_ball,column
        from scripts.build_spatial_assets import storage
        from counterscene.spatial_suite.core import composite_parts
        sx,sy,sz=item['size'];box=[-sx/2,sx/2,-sy/2,sy/2,0,sz];color=(.45,.5,.55)
        if item['asset']=='proc_bookcase':
            for name,offset,size in composite_parts('bookcase',item['size']):
                br.box_mesh(oid+' '+name,size,[offset[0],offset[1],offset[2]],br.principled(name,color,.7),0)
        else:
            builder={'proc_storage':storage,'proc_stool':stool,'proc_soccer':soccer_ball,'proc_rugby':rugby_ball,'proc_column':column}[item['asset']]
            if item['asset']=='proc_soccer':builder(oid,box)
            else:builder(oid,box,color)
    new=[o for o in bpy.data.objects if o not in before];meshes=[o for o in new if o.type=='MESH']
    vertices,faces=mesh_data(meshes);native=bounds(vertices)
    scale=item['height']/(native[2][1]-native[2][0]);origin=Vector([(native[0][0]+native[0][1])/2,(native[1][0]+native[1][1])/2,native[2][0]])
    rotation=Matrix.Rotation(math.radians(item['yaw']),4,'Z');position=Vector(item['position'])
    for obj in meshes:
        transform=obj.matrix_world.copy();obj.parent=None;obj.matrix_world=Matrix.Identity(4)
        if obj.data.users>1:obj.data=obj.data.copy()
        for v in obj.data.vertices:v.co=rotation@((transform@v.co-origin)*scale)+position
        obj.name=oid+'__'+obj.name
    for obj in new:
        if obj.type!='MESH':bpy.data.objects.remove(obj,do_unlink=True)
    vertices,faces=mesh_data(meshes)
    components=solid_components(vertices,faces)
    return vertices,faces,[mesh_data([obj]) for obj in meshes],dict(asset=item['asset'],asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),position=item['position'],yaw=item['yaw'],uniform_scale=scale,native_bounds=native,world_bounds=bounds(vertices),vertices=len(vertices),polygons=len(faces),closed_components=sum(c['closed'] for c in components),components=len(components),collision_enclosures=[c['bounds'] for c in components])


def export_scene(out,descriptions):
    """Export the scored meshes before adding presentation floor/lights."""
    for obj in bpy.data.objects:
        if obj.type!='MESH':continue
        oid=next((oid for oid in descriptions if obj.name.startswith(oid+'__')),None)
        if oid is None:raise ValueError('Unbound mesh in scored scene')
        obj['instance_id']=oid
        obj['asset_sha256']=descriptions[oid]['asset_sha256']
    path=out/'scene.glb'
    bpy.ops.export_scene.gltf(filepath=str(path),export_format='GLB',export_extras=True)
    return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),instance_ids=sorted(descriptions),coordinate_system='glTF Y-up; exported from scored Blender Z-up scene')


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--out',required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    request=json.loads(Path(a.request).read_text(encoding='utf-8'));scene=AssetScene(request['instances']);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    geometry={};descriptions={};parts={}
    for oid,item in scene.snapshot().items():
        v,f,part_geometry,description=load_instance(oid,item);geometry[oid]=(v,f);parts[oid]=part_geometry;descriptions[oid]=description
    enclosures={oid:description.pop('collision_enclosures') for oid,description in descriptions.items()}
    collisions=[]
    for left,right in itertools.combinations(geometry,2):collisions.append(dict(objects=[left,right],**compound_relation(parts[left],parts[right],tolerance=.01)))
    top_triangles={}
    for host,(vertices,faces) in geometry.items():
        top_triangles[host]=[]
        for face in faces:
            tri=[vertices[i] for i in face]
            normal=(tri[1]-tri[0]).cross(tri[2]-tri[0])
            if normal.length>1e-12 and normal.normalized().z>.5:top_triangles[host].append(tri)
    supports=[]
    for oid,description in descriptions.items():
        bottom=description['world_bounds'][2][0]
        if abs(bottom)<=.01:
            supports.append(dict(object=oid,status='floor_contact',support='floor'));continue
        x,y=description['position'][:2];origin=Vector((x,y,bottom+.005));hits=[];nearby=[]
        for host,(vertices,faces) in geometry.items():
            if host==oid:continue
            tree=BVHTree.FromPolygons(vertices,faces)
            location,normal,_,distance=tree.ray_cast(origin,Vector((0,0,-1)),1.0)
            if location is not None:nearby.append(dict(host=host,hit_z=location.z,normal_z=normal.z,gap_m=bottom-location.z))
            if location is not None and normal.z>.5 and abs(location.z-bottom)<=.01:hits.append(host)
        b=description['world_bounds'];rect=[b[0][0],b[0][1],b[1][0],b[1][1]]
        footprint={host:rectangle_coverage(rect,triangles,bottom) for host,triangles in top_triangles.items() if host!=oid}
        covered=[host for host,c in footprint.items() if c['covered']]
        supports.append(dict(footprint_supports=covered,footprint_coverage=footprint,object=oid,status='full_footprint_contact' if covered else 'centre_contact_witness' if hits else 'no_centre_support_witness',supports=hits,nearest_surfaces=nearby,limits='Projected bbox footprint covered by actual near-height upward triangles; static stability is not simulated'))
    for pair in collisions:
        left,right=pair['objects'];av,af=geometry[left];bv,bf=geometry[right]
        a,b=descriptions[left]['world_bounds'],descriptions[right]['world_bounds']
        lower=sum(max(0,a[i][0]-b[i][1],b[i][0]-a[i][1])**2 for i in range(3))**.5
        pair['surface_distance_lower_bound_m']=lower
        if lower<=.01:
            atree=BVHTree.FromPolygons(av,af);btree=BVHTree.FromPolygons(bv,bf)
            distances=[btree.find_nearest(v)[3] for v in av]+[atree.find_nearest(v)[3] for v in bv]
            pair['surface_distance_upper_bound_m']=min(d for d in distances if d is not None)
    witnesses={w['object']:w for w in supports}
    for pair in collisions:
        if pair['status'] in ('disjoint','within_axis_tolerance'):continue
        for child,host in (pair['objects'],pair['objects'][::-1]):
            b=descriptions[child]['world_bounds'];rect=[b[0][0],b[0][1],b[1][0],b[1][1]]
            vertices,faces=geometry[host];triangles=[[vertices[i] for i in face] for face in faces]
            proof=contact_certificate(rect,b[2][0],b[2][1],triangles,host in witnesses[child].get('footprint_supports',[]))
            if proof['certified']:
                pair.update(status='certified_support_contact',support_contact_proof=proof,child=child,host=host);break
    result=dict(collision_enclosures=enclosures,support_witnesses=supports,instances=descriptions,pairs=collisions,renders=[],limits=['Native GLB proportions preserved; transforms act on actual vertices','Surface intersection is a diagnostic, not penetration depth','Unknown/open topology is not a certified collision-free pass','No dynamics or stability simulation'])
    if request.get('camera'):
        from counterscene.blender_native_views import measure_view
        result['reference_view']=measure_view(geometry,descriptions,request['camera'])
    if request.get('extra_cameras'):
        from counterscene.blender_native_views import measure_view
        result['view_evidence']={name:measure_view(geometry,descriptions,camera,width=160 if request.get('pixel_evidence') else 80,height=120 if request.get('pixel_evidence') else 60,include_pixels=request.get('pixel_evidence',False)) for name,camera in request['extra_cameras'].items()}
    result['scene_artifact']=export_scene(out,descriptions)
    if request.get('render',False):
        room=request.get('room',[4,4,3]);br.SAMPLES=8;br.lights(*room)
        br.box_mesh('floor',(room[0],room[1],.02),(room[0]/2,room[1]/2,-.02),br.principled('floor',(.35,.37,.39),.8),0)
        cameras=grounding.cameras(room,['reference','front_left','back'])
        if request.get('camera'):cameras['reference']=request['camera']
        for view,camera in cameras.items():
            br.setup_render(640,480);br.camera(camera)
            path=out/(view+'.png');bpy.context.scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
            result['renders'].append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),camera=camera,raster_width=640,raster_height=480))
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(instances=len(descriptions),pairs=len(collisions),renders=len(result['renders']))),flush=True)

if __name__=='__main__':main()
