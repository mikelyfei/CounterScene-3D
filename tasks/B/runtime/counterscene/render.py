"""Trusted orthographic cuboid renderer: portable PNG + inspectable SVG.

Pure standard library; no GPU, browser, or candidate-provided images needed.
This is a schematic renderer, not photorealistic Blender output.
"""
import html
import math
from pathlib import Path
import struct
import zlib
from .contracts import reconstruct
from .geometry import footprint,zinterval,door_object

FONT={
'A':'01110100011000111111100011000110001','B':'11110100011000111110100011000111110',
'C':'01111100001000010000100001000001111','D':'11110100011000110001100011000111110',
'E':'11111100001000011110100001000011111','F':'11111100001000011110100001000010000',
'G':'01111100001000010111100011000101111','H':'10001100011000111111100011000110001',
'I':'11111001000010000100001000010011111','J':'00111000100001000010100101001001100',
'K':'10001100101010011000101001001010001','L':'10000100001000010000100001000011111',
'M':'10001110111010110101100011000110001','N':'10001110011010110011100011000110001',
'O':'01110100011000110001100011000101110','P':'11110100011000111110100001000010000',
'Q':'01110100011000110001101011001001101','R':'11110100011000111110101001001010001',
'S':'01111100001000001110000010000111110','T':'11111001000010000100001000010000100',
'U':'10001100011000110001100011000101110','V':'10001100011000110001100010101000100',
'W':'10001100011000110101101011101110001','X':'10001100010101000100010101000110001',
'Y':'10001100010101000100001000010000100','Z':'11111000010001000100010001000011111',
'0':'01110100011001110101110011000101110','1':'00100011000010000100001000010001110',
'2':'01110100010000100010001000100011111','3':'11110000010000101110000010000111110',
'4':'00010001100101010010111110001000010','5':'11111100001000011110000010000111110',
'6':'01110100001000011110100011000101110','7':'11111000010001000100010000100001000',
'8':'01110100011000101110100011000101110','9':'01110100011000101111000010000101110',
'-':'00000000000000011111000000000000000','.':'00000000000000000000000000011000110',
':':'00000001100011000000001100011000000','/':'00001000100001000100010000100010000',
'=':'00000000001111100000111110000000000','+':'00000001000010011111001000010000000'}


def rgb(color):
    return tuple(int(color[i:i+2],16) for i in (1,3,5)) if isinstance(color,str) else color


class Canvas:
    def __init__(self,w=960,h=680):
        self.w,self.h=w,h
        self.pixels=bytearray(rgb('#111b2d')*(w*h))
        self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="#111b2d"/>']

    def pixel(self,x,y,color):
        if 0<=x<self.w and 0<=y<self.h:
            i=3*(int(y)*self.w+int(x));self.pixels[i:i+3]=bytes(color)

    def line(self,a,b,color='#42516a',width=1):
        color=rgb(color);dx,dy=b[0]-a[0],b[1]-a[1]
        n=max(1,int(max(abs(dx),abs(dy))))
        for i in range(n+1):
            x,y=round(a[0]+dx*i/n),round(a[1]+dy*i/n)
            for xx in range(x-width//2,x+width//2+1):
                for yy in range(y-width//2,y+width//2+1):self.pixel(xx,yy,color)
        self.svg.append(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="rgb{color}" stroke-width="{width}"/>')

    def polygon(self,points,color):
        color=rgb(color)
        low=max(0,math.floor(min(p[1] for p in points)));high=min(self.h-1,math.ceil(max(p[1] for p in points)))
        for y in range(low,high+1):
            crosses=[]
            for a,b in zip(points,points[1:]+points[:1]):
                if (a[1]<=y+.5<b[1]) or (b[1]<=y+.5<a[1]):crosses.append(a[0]+(y+.5-a[1])*(b[0]-a[0])/(b[1]-a[1]))
            crosses.sort()
            for x0,x1 in zip(crosses[::2],crosses[1::2]):
                left=max(0,math.ceil(x0));right=min(self.w-1,math.floor(x1))
                if right>=left:
                    start=3*(y*self.w+left);self.pixels[start:start+3*(right-left+1)]=bytes(color)*(right-left+1)
        pts=' '.join(f'{x:.2f},{y:.2f}' for x,y in points)
        self.svg.append(f'<polygon points="{pts}" fill="rgb{color}"/>')
        for a,b in zip(points,points[1:]+points[:1]):self.line(a,b,'#142035')

    def text(self,pos,text,color='#dce6f4',scale=2):
        x,y=map(int,pos);col=rgb(color)
        for index,ch in enumerate(text.upper()):
            glyph=FONT.get(ch)
            if not glyph:continue
            for i,v in enumerate(glyph):
                if v=='1':
                    for dx in range(scale):
                        for dy in range(scale):self.pixel(x+index*6*scale+i%5*scale+dx,y+i//5*scale+dy,col)
        self.svg.append(f'<text x="{x}" y="{y+7*scale}" fill="{color}" font-family="monospace" font-size="{9*scale}">{html.escape(text)}</text>')

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        def chunk(kind,data):return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
        scan=b''.join(b'\0'+self.pixels[y*self.w*3:(y+1)*self.w*3] for y in range(self.h))
        path.with_suffix('.png').write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',self.w,self.h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(scan,6))+chunk(b'IEND',b''))
        path.with_suffix('.svg').write_text(''.join(self.svg)+'</svg>',encoding='utf-8')
        return path.with_suffix('.png')


def render(task,scene,path,view='iso',angle=0,reference=False):
    _,objects=reconstruct(task,scene,scene['phase'])
    if view not in ('iso','top','front','side'):raise ValueError('unknown view')
    concealed=reference and angle==0 and task.get('reference_motion_probe_required',False)
    if 'door' in task:
        # Both installations have the same closed solid. Canonicalize its face
        # order too, avoiding a spurious hinge cue from rasterizer shading.
        side='left' if concealed else scene['joints'][0]['hinge_side']
        objects.append(dict(door_object(task['door'],side,angle),color='#8dd8c7'))
    w,h=task['bounds_m'];floor=.8 if task['id']=='t1' else 0
    def raw(x,y,z):
        if view=='top':return (x,-y)
        if view=='front':return (x,-z)
        if view=='side':return (y,-z)
        return (x-.65*y,-.36*x-.48*y-.9*z)
    corners=[raw(x,y,z) for x in (0,w) for y in (0,h) for z in (0,1.3)]
    xmin,xmax=min(p[0] for p in corners),max(p[0] for p in corners)
    ymin,ymax=min(p[1] for p in corners),max(p[1] for p in corners)
    factor=min(800/max(.1,xmax-xmin),450/max(.1,ymax-ymin))
    def project(x,y,z):
        a,b=raw(x,y,z)
        return (80+(a-xmin)*factor,130+(b-ymin)*factor)
    c=Canvas();c.text((32,28),'COUNTERSCENE / '+task['id'].upper()+' / '+view.upper(),scale=3)
    c.text((32,65),('REFERENCE' if reference else scene['phase'].upper())+f' / DOOR {angle:02.0f} DEG',color='#8dd8c7')
    c.polygon([project(x,y,floor) for x,y in [(0,0),(w,0),(w,h),(0,h)]],'#26364a')
    for axis,length in [(0,w),(1,h)]:
        for i in range(int(length/.2)+1):
            t=i*.2
            a,b=([t,0,floor],[t,h,floor]) if axis==0 else ([0,t,floor],[w,t,floor])
            c.line(project(*a),project(*b),'#33445b')
    if 'free_zone' in task:
        x0,y0,x1,y1=task['free_zone']
        c.polygon([project(x,y,floor+.001) for x,y in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]],'#574830')
        for i in range(8):
            y=y0+(y1-y0)*i/8;c.line(project(x0,y,floor+.002),project(x1,min(y1,y+.1),floor+.002),'#cfa658')
    faces=[]
    for obj in objects:
        poly=footprint(obj);lo,hi=zinterval(obj)
        vertices=[(x,y,z) for z in (lo,hi) for x,y in poly]
        for indices,shade in [([4,5,6,7],1.),([0,1,5,4],.65),([1,2,6,5],.78),([2,3,7,6],.72),([3,0,4,7],.6)]:
            if view=='top' and indices!=[4,5,6,7]:continue
            verts=[vertices[i] for i in indices]
            depth=sum(v[0]+v[1]+v[2]*.1 for v in verts)/4
            color=tuple(int(v*shade) for v in rgb(obj.get('color','#94a3b8')))
            faces.append((depth,[project(*v) for v in verts],color))
    for _,points,color in sorted(faces,key=lambda f:-f[0]):c.polygon(points,color)
    pathpoints=scene['path_xy_m']
    for a,b in zip(pathpoints,pathpoints[1:]):c.line(project(*a,.02),project(*b,.02),'#c4f478',4)
    for label,key in [('A','start'),('B','goal')]:
        if key in task:
            p=project(*task[key],.02);c.text((p[0]+5,p[1]-20),label,color='#c4f478',scale=3)
    for o in objects:
        p=project(*o['center_m'][:2],zinterval(o)[1]);c.text((p[0]+5,p[1]-18),o['id'],scale=1)
    if 'door' in task and not concealed:
        side=scene['joints'][0]['hinge_side'];hinge=task['door']['hinges'][side]
        p=project(*hinge,task['door']['size_m'][2]+.03)
        marker=[(p[0]+7*math.cos(i*math.pi/4),p[1]+7*math.sin(i*math.pi/4)) for i in range(8)]
        c.polygon(marker,'#f4a5d3');c.text((p[0]+12,p[1]-7),'HINGE',color='#f4a5d3',scale=2)
    origin=project(0,0,floor)
    c.line(origin,project(min(w,.5),0,floor),'#ef6b65',3)
    c.line(origin,project(0,min(h,.5),floor),'#8dd8c7',3)
    c.text((32,620),'WORLD: +X RED / +Y MINT / +Z UP / UNITS M',color='#91a4bf')
    c.text((32,646),'GRID 0.20 M / TRUSTED CUBOID RECONSTRUCTION',color='#91a4bf',scale=1)
    return c.save(path)


def views(task,scene,dest,reference=False):
    return {v:str(render(task,scene,Path(dest)/v,v,reference=reference)) for v in ('iso','front','top','side')}
