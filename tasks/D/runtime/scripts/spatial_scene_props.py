"""Shared scene modeling for logical indoor items and grounded pick/place."""
import math
from counterscene.spatial_suite import blender_render as br
from scripts.render_spatial_ae import MODELS,rod,sphere


def stool(oid,b,color):
    sx,sy,h=b[1]-b[0],b[3]-b[2],b[5]-b[4];x,y,z=(b[0]+b[1])/2,(b[2]+b[3])/2,b[4]
    mat=br.principled(oid+' seat',color,.65);leg=br.principled(oid+' legs',(.12,.15,.16),.45,.4)
    br.box_mesh(oid+' seat',(sx,sy,.06),(x,y,z+h-.06),mat,.008)
    for dx in (-sx*.35,sx*.35):
        for dy in (-sy*.35,sy*.35):rod(oid+' leg',(x+dx,y+dy,z),(x+dx,y+dy,z+h-.06),.02,leg)


def cabinet(oid,b,color,column=False):
    sx,sy,h=b[1]-b[0],b[3]-b[2],b[5]-b[4];x,y,z=(b[0]+b[1])/2,(b[2]+b[3])/2,b[4]
    body=br.principled(oid+' body',color,.6,.2);trim=br.principled(oid+' fittings',(.12,.15,.16),.4,.5)
    br.box_mesh(oid+' body',(sx,sy if column else sy-.03,h),(x,y if column else y+.015,z),body,.006)
    if column:
        for zz in (z+.015,z+h-.03):br.box_mesh(oid+' cap',(sx,sy,.03),(x,y,zz),trim,.003)
    else:
        for fraction in (.28,.6):
            br.box_mesh(oid+' drawer front',(sx*.88,.012,h*.25),(x,y-sy/2+.018,z+h*fraction),body,.003)
            br.box_mesh(oid+' handle',(sx*.30,.012,.015),(x,y-sy/2+.006,z+h*(fraction+.14)),trim,.003)


def soccer_ball(oid,b):
    """Icosphere fitted to the box; faces near the 12 icosahedron vertices are the black pentagons."""
    import bpy
    from mathutils import Vector
    x,y,z=(b[0]+b[1])/2,(b[2]+b[3])/2,b[4];r=(b[1]-b[0])/2
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3,radius=r,location=(x,y,z+r));o=bpy.context.object;o.name=oid
    white=br.principled(oid+' white panels',(.86,.86,.84),.45);black=br.principled(oid+' black panels',(.03,.03,.035),.45)
    o.data.materials.append(white);o.data.materials.append(black)
    phi=(1+5**.5)/2;tips=[Vector(v).normalized() for s in ((-1,1),(1,-1),(1,1),(-1,-1)) for v in ((0,s[0],s[1]*phi),(s[0],s[1]*phi,0),(s[1]*phi,0,s[0]))]
    for poly in o.data.polygons:
        if max(poly.normal.dot(t) for t in tips)>.955:poly.material_index=1
    bpy.ops.object.shade_smooth();return o


def rugby_ball(oid,b,color):
    """Prolate spheroid lying on its side (long axis = longer footprint axis) with a white lace stripe."""
    import bpy
    sx,sy,sz=b[1]-b[0],b[3]-b[2],b[5]-b[4];x,y,z=(b[0]+b[1])/2,(b[2]+b[3])/2,b[4]
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=16,radius=.5,location=(x,y,z+sz/2));o=bpy.context.object;o.name=oid;o.scale=(sx,sy,sz)
    o.data.materials.append(br.principled(oid+' leather',color,.55));bpy.ops.object.shade_smooth()
    lace=br.principled(oid+' lace',(.92,.9,.86),.6);along=sx>=sy
    br.box_mesh(oid+' lace',(sx*.32 if along else .012,.012 if along else sy*.32,.012),(x,y,z+sz-.008),lace,.002)
    for k in (-2,-1,0,1,2):
        dx,dy=(k*sx*.06,0) if along else (0,k*sy*.06)
        br.box_mesh(oid+' stitch',(.008 if along else .03,.03 if along else .008,.01),(x+dx,y+dy,z+sz-.008),lace,.001)
    return o


def column(oid,b,color):
    """Round pillar fitted to its square box: shaft, base plinth and capital."""
    sx,sy,h=b[1]-b[0],b[3]-b[2],b[5]-b[4];x,y,z=(b[0]+b[1])/2,(b[2]+b[3])/2,b[4]
    stone=br.principled(oid+' plaster',color,.75)
    br.box_mesh(oid+' plinth',(sx,sy,.06),(x,y,z),stone,.01);br.box_mesh(oid+' capital',(sx,sy,.06),(x,y,z+h-.06),stone,.01)
    rod(oid+' shaft',(x,y,z+.06),(x,y,z+h-.06),min(sx,sy)*.38,stone)


def room_objects(geo,state,mode):
    for row in geo[state]:
        b=row['aabb'];oid=row['id'];colorname=oid.rsplit('_',1)[-1];color=br.PALETTE.get(colorname,(.35,.45,.4));size=[b[1]-b[0],b[3]-b[2],b[5]-b[4]];pos=[(b[0]+b[1])/2,(b[2]+b[3])/2,b[4]]
        if row.get('envelope'):continue                      # drawn through its scored parts
        if row.get('part_of'):                               # bookcase panels and boards: exact scored boxes in wood
            color=br.PALETTE.get(row['part_of'].rsplit('_',1)[-1],(.35,.45,.4))
            br.box_mesh(oid,size,pos,br.principled(oid+' proxy',color,.7) if mode=='proxy' else br.wood_material(oid+' wood',color,(1.,1.,6.)),0 if mode=='proxy' else .004);continue
        if mode=='proxy':br.box_mesh(oid,size,pos,br.principled(oid+' proxy',color,.7),0);continue
        if oid=='stool_purple':stool(oid,b,color);continue
        if oid=='soccer_white':soccer_ball(oid,b);continue
        if oid=='rugby_brown':rugby_ball(oid,b,color);continue
        if oid=='column_cyan':column(oid,b,color);continue
        model=MODELS.get(oid)
        if model:
            yaw=row['yaw'];native=[size[1],size[0],size[2]] if yaw%180 else size
            br.import_fitted(dict(id=oid,model=model,box=dict(size=native,position=pos,yaw=yaw),color=colorname),br.PALETTE)
        else:
            from scripts.build_spatial_assets import storage
            storage(oid,b,color)


def ball_positions(geo,state,key):
    """Where each ball is at keyframe `key` of the gold motion: resting at pickup, on the gripper, or placed."""
    if 'executed_ball_positions' in geo:return geo['executed_ball_positions'][key]
    span={}
    for e in geo['events']:span.setdefault(e['ball'],{})[e['action']]=e['keyframe']
    out={}
    for b in geo['balls']:
        g,r=span[b['id']]['grasp'],span[b['id']]['release']
        out[b['id']]=b['pickup'] if state!='gold' or key<g else b['destination'] if key>=r else None     # None = on the gripper
    return out


def default_key(geo):
    """Mid-carry of the first ball: the most informative single frame."""
    e=geo['events'];return (e[0]['keyframe']+e[1]['keyframe'])//2


def mechanism(geo,state,key=None):
    """Arm at gold keyframe `key` (default mid-carry) with every ball where it is at that moment."""
    steel=br.principled('arm anodized blue',(.035,.25,.43),.28,.65);gold=br.principled('joint housing',(.82,.42,.08),.3,.55);dark=br.principled('machined pedestal',(.16,.22,.25),.4,.5);ball=br.principled('carried ball',(.95,.53,.045),.42)
    frames=geo[state];key=default_key(geo) if key is None else key
    points=frames[min(len(frames)-1,key*geo['frames_per_segment'])] if state=='gold' else frames[len(frames)//2]
    rod('fixed post',points[0],points[1],geo['radius'],dark)
    br.box_mesh('pedestal',(.25,.25,.025),[points[0][0],points[0][1],0],dark,.008)
    for n in (1,2):
        end=points[n+1]
        if n==2:
            length=math.dist(points[2],points[3]);end=[points[3][j]-.13*(points[3][j]-points[2][j])/length for j in range(3)]
        rod('moving link '+str(n),points[n],end,geo['radius'],steel);sphere('joint '+str(n),points[n],geo['radius'],gold)
    tip=points[3]
    for bid,pos in ball_positions(geo,state,key).items():
        sphere('ball '+bid,pos or tip,geo['balls'][0]['radius_m'],ball)
    for obj in geo.get('resting_objects',[]):
        sphere('released '+obj['id'],obj['position'],obj['radius_m'],ball)
    from counterscene.native_mechanism import TOOL_BRIDGE_BOTTOM,TOOL_CONNECTOR_TOP
    for sign in (-1,1):br.box_mesh('gripper finger '+str(sign),(.025,.06,TOOL_BRIDGE_BOTTOM+.04),(tip[0]+sign*.0925,tip[1],tip[2]-.04),dark,.005)
    wrist=[points[3][j]-.13*(points[3][j]-points[2][j])/math.dist(points[2],points[3]) for j in range(3)]
    rod('gripper wrist connector',wrist,[tip[0],tip[1],tip[2]+TOOL_CONNECTOR_TOP],.018,dark)
    br.box_mesh('gripper bridge',(.21,.06,.03),(tip[0],tip[1],tip[2]+TOOL_BRIDGE_BOTTOM),dark,.006)
    for ob in geo['obstacles']:cabinet(ob['id'],ob['box'],(.27,.34,.37),column=ob.get('kind')=='floor_column')
    marker=br.principled('virtual transfer guide',(.05,.7,.57),.6)
    for wp in geo['waypoints']:
        p=wp['position']
        if wp.get('role')=='transfer':
            # A wire cross is a UI guide, not an unsupported solid object.
            for axis in (0,1,2):
                a=list(p);b=list(p);a[axis]-=.06;b[axis]+=.06;rod('virtual '+wp['id'],a,b,.003,marker)
        else:
            r=geo['balls'][0]['radius_m'];z=p[2]-r+.004;r*=1.45          # ring on whatever the ball rests on (floor or cabinet top)
            for j in range(24):
                a=[p[0]+r*math.cos(j*math.pi/12),p[1]+r*math.sin(j*math.pi/12),z];b=[p[0]+r*math.cos((j+1)*math.pi/12),p[1]+r*math.sin((j+1)*math.pi/12),z];rod(wp['id']+' floor marking',a,b,.003,gold)
    for j in range(1,len(geo[state])):rod('trajectory',geo[state][j-1][3],geo[state][j][3],.004,gold)
    return points
