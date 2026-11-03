"""Reference-view measurements from actual Blender BVH triangles, not boxes."""
import math
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from .spatial_suite.raycast import _basis
from .spatial_suite.core import _project


def measure_view(geometry,descriptions,camera,width=80,height=60,include_pixels=False):
    trees={oid:BVHTree.FromPolygons(*mesh) for oid,mesh in geometry.items()}
    alone={oid:0 for oid in trees};hidden={a:{b:0 for b in trees if b!=a} for a in trees}
    forward,right,up=[Vector(v) for v in _basis(camera)]
    origin=Vector(camera['eye']);th=math.tan(math.radians(camera['fov_deg'])/2);tw=th*width/height
    visible=[];depth=[]
    for y in range(height):
        sy=(1-2*(y+.5)/height)*th
        for x in range(width):
            sx=(2*(x+.5)/width-1)*tw;direction=(forward+sx*right+sy*up).normalized();distances={}
            for oid,tree in trees.items():
                hit,_,_,distance=tree.ray_cast(origin,direction)
                if hit is not None:distances[oid]=distance;alone[oid]+=1
            if include_pixels:
                oid=min(distances,key=lambda name:(distances[name],name)) if distances else None
                visible.append(oid);depth.append(distances[oid] if oid is not None else None)
            for front,da in distances.items():
                for back,db in distances.items():
                    if front!=back and da+1e-6<db:hidden[front][back]+=1
    centres={}
    for oid,d in descriptions.items():centres[oid]=_project(camera,[(a+b)/2 for a,b in d['world_bounds']])[:2]
    result=dict(method='Actual mesh ray hits; pair-only occlusion; unoccluded silhouette pixel area',width=width,height=height,camera=camera,unoccluded_pixels=alone,projected_centres=centres,pair_occlusion={a:{b:hidden[a][b]/alone[b] if alone[b] else None for b in hidden[a]} for a in hidden})
    if include_pixels:result.update(visible_ids=visible,visible_depth_m=depth)
    return result
