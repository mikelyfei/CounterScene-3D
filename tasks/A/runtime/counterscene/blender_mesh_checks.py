"""Blender mesh diagnostics with conservative topology/containment handling.

No dynamics or penetration-depth claim. Non-closed components remain unknown.
"""
from collections import Counter,defaultdict
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def solid_components(vertices,faces):
    keys=[tuple(round(x,6) for x in v) for v in vertices]
    edges=defaultdict(list)
    for i,face in enumerate(faces):
        for a,b in zip(face,face[1:]+face[:1]):edges[tuple(sorted((keys[a],keys[b])))].append(i)
    neighbors=defaultdict(set)
    for indexes in edges.values():
        for i in indexes:neighbors[i].update(indexes)
    groups=[];unseen=set(range(len(faces)))
    while unseen:
        todo=[unseen.pop()];group=[]
        while todo:
            i=todo.pop();group.append(i)
            fresh=neighbors[i]&unseen;unseen-=fresh;todo.extend(fresh)
        selected=[faces[i] for i in group];counts=Counter()
        for face in selected:
            for a,b in zip(face,face[1:]+face[:1]):counts[tuple(sorted((keys[a],keys[b])))]+=1
        indices={index for face in selected for index in face}
        enclosure=[[min(vertices[index][axis] for index in indices),max(vertices[index][axis] for index in indices)] for axis in range(3)]
        groups.append(dict(tree=BVHTree.FromPolygons(vertices,selected),closed=all(n==2 for n in counts.values()),representative=Vector(vertices[selected[0][0]]),faces=len(selected),bounds=enclosure))
    return groups


def contains(tree,point,epsilon=1e-6):
    point=Vector(point)
    nearest=tree.find_nearest(point)
    if nearest[3] is not None and nearest[3]<=epsilon:return None
    outcomes=[]
    for direction in ((1,.371,.173),(.219,1,.413),(.317,.191,1)):
        direction=Vector(direction).normalized();origin=point.copy();hits=0
        for _ in range(10000):
            location,_,_,distance=tree.ray_cast(origin,direction)
            if location is None:break
            hits+=1;origin=location+direction*epsilon
        else:return None
        outcomes.append(hits%2==1)
    return outcomes[0] if all(v==outcomes[0] for v in outcomes) else None


def relation(av,af,bv,bf,tolerance=1e-6):
    bounds=lambda vs:[(min(v[i] for v in vs),max(v[i] for v in vs)) for i in range(3)]
    a,b=bounds(av),bounds(bv)
    broad=all(min(a[i][1],b[i][1])-max(a[i][0],b[i][0])>1e-6 for i in range(3))
    if not broad:return dict(aabb_overlap=False,status='disjoint',surface_pairs=0)
    overlap=min(min(a[i][1],b[i][1])-max(a[i][0],b[i][0]) for i in range(3))
    if overlap<=tolerance:return dict(aabb_overlap=True,status='within_axis_tolerance',surface_pairs=0,separation_translation_upper_bound_m=overlap,tolerance_m=tolerance)
    surface=BVHTree.FromPolygons(av,af).overlap(BVHTree.FromPolygons(bv,bf))
    if surface:return dict(aabb_overlap=True,status='surface_intersection',surface_pairs=len(surface))
    ac,bc=solid_components(av,af),solid_components(bv,bf)
    if not all(c['closed'] for c in ac+bc):return dict(aabb_overlap=True,status='unknown_open_or_nonmanifold',surface_pairs=0)
    answers=[contains(b['tree'],a['representative']) for a in ac for b in bc]+[contains(a['tree'],b['representative']) for b in bc for a in ac]
    if any(x is True for x in answers):status='containment'
    elif any(x is None for x in answers):status='unknown_boundary_or_ray_disagreement'
    else:status='disjoint'
    return dict(aabb_overlap=True,status=status,surface_pairs=0)


def compound_relation(aparts,bparts,tolerance=1e-6):
    """Union of declared mesh parts; touching separate solids are not welded."""
    results=[relation(*a,*b,tolerance=tolerance) for a in aparts for b in bparts]
    statuses={r['status'] for r in results}
    priority=['surface_intersection','containment','unknown_open_or_nonmanifold','unknown_boundary_or_ray_disagreement']
    status=next((s for s in priority if s in statuses),'within_axis_tolerance' if 'within_axis_tolerance' in statuses else 'disjoint')
    return dict(aabb_overlap=any(r['aabb_overlap'] for r in results),status=status,surface_pairs=sum(r['surface_pairs'] for r in results),part_pairs=len(results))
