"""Conservative planar footprint coverage by actual support triangles."""


def area(poly):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1])))/2 if len(poly)>=3 else 0


def halfplane(poly,a,b,inside):
    def signed(p):return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
    result=[]
    if not poly:return result
    previous=poly[-1];dp=signed(previous);previous_ok=dp>=-1e-12 if inside else dp<=1e-12
    for point in poly:
        d=signed(point);ok=d>=-1e-12 if inside else d<=1e-12
        if ok!=previous_ok:
            denominator=dp-d
            if abs(denominator)>1e-15:
                t=dp/denominator;result.append((previous[0]+t*(point[0]-previous[0]),previous[1]+t*(point[1]-previous[1])))
        if ok:result.append(point)
        previous,dp,previous_ok=point,d,ok
    return result


def subtract_triangle(poly,triangle):
    a,b,c=triangle
    if (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])<0:triangle=[a,c,b]
    remaining=poly;outside=[]
    for a,b in zip(triangle,triangle[1:]+triangle[:1]):
        part=halfplane(remaining,a,b,False)
        if area(part)>1e-12:outside.append(part)
        remaining=halfplane(remaining,a,b,True)
        if not remaining:break
    return outside


def rectangle_coverage(rect,triangles,height,tolerance=.01):
    x0,x1,y0,y1=rect
    if x1<=x0 or y1<=y0:raise ValueError('Nonempty footprint required')
    remaining=[[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]];used=0
    for triangle in triangles:
        if len(triangle)!=3 or any(abs(v[2]-height)>tolerance for v in triangle):continue
        projected=[(v[0],v[1]) for v in triangle]
        if area(projected)<1e-12:continue
        remaining=[piece for poly in remaining for piece in subtract_triangle(poly,projected)];used+=1
        if not remaining:break
    missing=sum(area(poly) for poly in remaining);total=(x1-x0)*(y1-y0)
    return dict(covered=missing<=1e-8,uncovered_area_m2=missing,footprint_area_m2=total,support_triangles_used=used,height_tolerance_m=tolerance)


def clip_coordinate(poly,axis,bound,keep_below):
    if not poly:return []
    result=[];previous=poly[-1];dp=(bound-previous[axis]) if keep_below else (previous[axis]-bound)
    for point in poly:
        d=(bound-point[axis]) if keep_below else (point[axis]-bound)
        if (d>=0)!=(dp>=0):
            t=dp/(dp-d);result.append(tuple(previous[i]+t*(point[i]-previous[i]) for i in range(3)))
        if d>=0:result.append(point)
        previous,dp=point,d
    return result


def contact_certificate(rect,bottom,top,host_triangles,footprint_covered,tolerance=.01):
    if not footprint_covered:return dict(certified=False,reason='No full support coverage')
    heights=[]
    for triangle in host_triangles:
        polygon=list(triangle)
        for axis,bound,below in [(0,rect[0],False),(0,rect[1],True),(1,rect[2],False),(1,rect[3],True),(2,top+tolerance,True)]:
            polygon=clip_coordinate(polygon,axis,bound,below)
            if not polygon:break
        heights.extend(p[2] for p in polygon)
    if not heights:return dict(certified=False,reason='No host surface in inspected footprint/slab')
    upper=max(0,max(heights)-bottom)
    return dict(certified=upper<=tolerance+1e-6,host_surface_intrusion_upper_bound_m=upper,tolerance_m=tolerance,method='Actual host triangles clipped to supported child footprint and upper slab')
