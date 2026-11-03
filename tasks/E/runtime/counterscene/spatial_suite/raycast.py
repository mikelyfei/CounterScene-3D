"""Trusted pure-Python perspective ray caster for axis-aligned box scenes.

Returns colour, per-pixel object index and camera depth, so silhouettes and depth can be
scored exactly. shortcut: CPU ray casting (~1 s per 320x240 view); swap for Blender/Cycles
only if photorealism ever becomes part of the contract.
"""
import math
import struct
import zlib

from . import core

COLORS = {'red': (200, 50, 45), 'blue': (45, 90, 200), 'green': (50, 160, 70), 'yellow': (235, 200, 40),
          'white': (240, 240, 235), 'orange': (240, 130, 30), 'purple': (130, 60, 170), 'gray': (70, 80, 95),
          'cyan': (40, 190, 200), 'pink': (240, 130, 180), 'brown': (130, 85, 45), 'black': (30, 30, 30)}
SKY, FLOOR_A, FLOOR_B, OUTSIDE = (205, 222, 238), (175, 175, 170), (150, 150, 146), (120, 130, 140)
LIGHT = (0.35, -0.5, 0.79)


def color_of(oid, obj=None):
    if obj and obj.get('color') in COLORS:
        return COLORS[obj['color']]
    return COLORS.get(oid.split('#')[0].split('_')[-1], (180, 180, 180))


def base_name(n):
    """Parts of a composite object ('table#2') are scored as one object ('table')."""
    return n.split('#')[0] if n else n


def camera(eye, target, fov_deg=50., width=320, height=240):
    return dict(eye=[round(v, 4) for v in eye], target=[round(v, 4) for v in target], fov_deg=fov_deg, width=width, height=height, up=[0, 0, 1])


def _basis(cam):
    e, t = cam['eye'], cam['target']
    f = [t[i] - e[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in f))
    f = [v / n for v in f]
    up=cam.get('up',[0,0,1])
    r = [f[1]*up[2]-f[2]*up[1],f[2]*up[0]-f[0]*up[2],f[0]*up[1]-f[1]*up[0]]
    n = math.sqrt(sum(v*v for v in r))
    if n<1e-12:raise ValueError('Camera up is collinear with view direction')
    r = [value/n for value in r]
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    return f, r, u


def render(objs, room, cam, scale=1):
    """objs: {id: object dict}. Returns dict(rgb=bytes, ids=list, depth=list, width, height)."""
    W, H = cam['width'] // scale, cam['height'] // scale
    f, r, u = _basis(cam)
    th = math.tan(math.radians(cam['fov_deg']) / 2)
    tw = th * W / H
    e = cam['eye']
    objs = core.solids(objs)
    names = list(objs)
    boxes = [core.aabb(objs[i]) for i in names]
    cols = [color_of(i, objs[i]) for i in names]
    rgb, ids, depth = bytearray(), [], []
    for py in range(H):
        sy = (1 - 2 * (py + .5) / H) * th
        for px in range(W):
            sx = (2 * (px + .5) / W - 1) * tw
            d = [f[k] + sx * r[k] + sy * u[k] for k in range(3)]
            best, hit, normal = 1e9, -2, None
            for bi, b in enumerate(boxes):
                t0, t1, ax = -1e9, 1e9, 0
                for k in range(3):
                    if abs(d[k]) < 1e-12:
                        if e[k] < b[2 * k] or e[k] > b[2 * k + 1]:
                            t0 = 1e9
                            break
                        continue
                    a, c = (b[2 * k] - e[k]) / d[k], (b[2 * k + 1] - e[k]) / d[k]
                    if a > c:
                        a, c = c, a
                    if a > t0:
                        t0, ax = a, k
                    t1 = min(t1, c)
                if t0 <= t1 and 1e-6 < t0 < best:
                    best, hit = t0, bi
                    normal = [0., 0., 0.]
                    normal[ax] = -1. if d[ax] > 0 else 1.
            if d[2] < -1e-9:
                tf = -e[2] / d[2]
                if tf < best:
                    x, y = e[0] + tf * d[0], e[1] + tf * d[1]
                    best, hit = tf, -1
                    inside = 0 <= x <= room[0] and 0 <= y <= room[1]
                    base = (FLOOR_A if (math.floor(x / .5) + math.floor(y / .5)) % 2 == 0 else FLOOR_B) if inside else OUTSIDE
            if hit >= 0:
                lam = .55 + .45 * max(0., sum(normal[k] * LIGHT[k] for k in range(3)))
                rgb += bytes(min(255, int(c * lam)) for c in cols[hit])
            elif hit == -1:
                rgb += bytes(base)
            else:
                rgb += bytes(SKY)
            ids.append(hit)
            depth.append(best * sum(d[k] * f[k] for k in range(3)) if hit != -2 else 0.)
    return dict(rgb=bytes(rgb), ids=ids, depth=depth, width=W, height=H, names=names)


def png(img):
    W, H, raw = img['width'], img['height'], img['rgb']
    rows = b''.join(b'\x00' + raw[y * W * 3:(y + 1) * W * 3] for y in range(H))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(rows, 9)) + chunk(b'IEND', b'')


def compare(gold, cand):
    """Per-object mask IoU (keyed by object name), foreground IoU and depth error between two renders."""
    gi = [base_name(gold['names'][i]) if i >= 0 else None for i in gold['ids']]
    ci = [base_name(cand['names'][i]) if i >= 0 else None for i in cand['ids']]
    names = set(n for n in gi if n) | set(n for n in ci if n)
    per = {}
    for n in names:
        inter = sum(1 for a, b in zip(gi, ci) if a == n and b == n)
        union = sum(1 for a, b in zip(gi, ci) if a == n or b == n)
        per[n] = inter / union if union else 1.
    fg_i = sum(1 for a, b in zip(gi, ci) if a and b)
    fg_u = sum(1 for a, b in zip(gi, ci) if a or b)
    errs = [abs(dg - dc) for a, b, dg, dc in zip(gi, ci, gold['depth'], cand['depth']) if a and a == b]
    return dict(object_iou=per, mean_object_iou=sum(per.values()) / len(per) if per else 1.,
                foreground_iou=fg_i / fg_u if fg_u else 1., depth_abs_err=sum(errs) / len(errs) if errs else None,
                visible_px={n: sum(1 for a in gi if a == n) for n in names})
