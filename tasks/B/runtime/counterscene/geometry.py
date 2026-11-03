"""Small, dependency-free convex geometry kernel. Distances are in metres.

Footprints are convex CCW polygons; touching polygons have zero distance.
No mesh physics or dynamic stability is implied by these predicates.
"""
import math

EPS = 1e-10


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def point_segment(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    den = dx*dx+dy*dy
    t = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/den)) if den else 0.
    return math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)


def edges(poly):
    return zip(poly, poly[1:]+poly[:1])


def contains(poly, p, tolerance=EPS):
    return all(cross(a, b, p) >= -tolerance for a, b in edges(poly))


def segments_intersect(a, b, c, d):
    if min(point_segment(a,c,d), point_segment(b,c,d),
           point_segment(c,a,b), point_segment(d,a,b)) <= EPS:
        return True
    return cross(a,b,c)*cross(a,b,d) < 0 and cross(c,d,a)*cross(c,d,b) < 0


def segment_distance(a, b, c, d):
    if segments_intersect(a,b,c,d):
        return 0.
    return min(point_segment(a,c,d), point_segment(b,c,d),
               point_segment(c,a,b), point_segment(d,a,b))


def segment_polygon(a, b, poly):
    if contains(poly,a) or contains(poly,b):
        return 0.
    return min(segment_distance(a,b,c,d) for c,d in edges(poly))


def polygon_distance(a, b):
    if contains(a,b[0]) or contains(b,a[0]):
        return 0.
    return min(segment_distance(p,q,r,s) for p,q in edges(a) for r,s in edges(b))


def rectangle(center, size, yaw=0.):
    a = math.radians(yaw)
    c,s = math.cos(a),math.sin(a)
    return [[center[0]+c*x-s*y, center[1]+s*x+c*y]
            for x,y in [(-size[0]/2,-size[1]/2),(size[0]/2,-size[1]/2),
                        (size[0]/2,size[1]/2),(-size[0]/2,size[1]/2)]]


def footprint(obj):
    return rectangle(obj['center_m'],obj['size_m'],obj.get('yaw_deg',0))


def zinterval(obj):
    return (obj['center_m'][2]-obj['size_m'][2]/2,
            obj['center_m'][2]+obj['size_m'][2]/2)


def solid_distance(a,b):
    za,zb = zinterval(a),zinterval(b)
    vertical = max(0.,za[0]-zb[1],zb[0]-za[1])
    return math.hypot(vertical,polygon_distance(footprint(a),footprint(b)))


def door_object(door, side, angle):
    """Left opens counter-clockwise, right clockwise; hinge is fixed."""
    sign = 1 if side == 'left' else -1
    yaw = angle if sign == 1 else 180-angle
    hinge = door['hinges'][side]
    a=math.radians(yaw)
    width,thickness,height=door['size_m']
    return dict(id='door',size_m=door['size_m'],yaw_deg=yaw,
                center_m=[hinge[0]+width/2*math.cos(a),
                          hinge[1]+width/2*math.sin(a),height/2])


def sweep(door,side,obstacles,clearance=.03,samples=61,max_samples=961):
    """Lipschitz lower bound certifies the entire continuous angular interval.

    Every angle is within delta/2 of a sample. All points on the door move
    by at most r_max*delta/2. A failed lower bound is never labelled success.
    """
    radius=math.hypot(door['size_m'][0],door['size_m'][1]/2)
    if not obstacles:
        return dict(status='pass',certified_clearance_m=None,samples=0)
    while True:
        measured=[]
        for i in range(samples):
            angle=90*i/(samples-1)
            obj=door_object(door,side,angle)
            dist,other=min((solid_distance(obj,o),o['id']) for o in obstacles)
            measured.append((dist,angle,other))
        gap,angle,other=min(measured)
        bound=radius*math.pi/2/(samples-1)/2
        lower=gap-bound
        status='pass' if lower>=clearance+EPS else 'fail' if gap<clearance-EPS else 'uncertain'
        if status!='uncertain' or samples>=max_samples:
            return dict(status=status,sampled_clearance_m=gap,
                        certified_clearance_m=lower,motion_bound_m=bound,
                        worst_angle_deg=angle,obstacle=other,samples=samples)
        samples=min(max_samples,2*(samples-1)+1)
