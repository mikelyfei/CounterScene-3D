"""Realistic review/reference renders of a box-scene with CC0 Kenney furniture meshes.

  blender -b -P <this file> -- scene.json out_dir

scene.json: {"room": [W, D, H],
             "camera_views": {name: {"eye": [x,y,z], "target": [x,y,z], "fov_deg": 50, "width": 640, "height": 480}},
             "objects": [{"id", "model", "box": {"size": [sx,sy,sz], "position": [cx,cy,bottom_z], "yaw": deg}, "color": name}
                         | {"id", "parts": [{"size": [sx,sy,sz], "position": [cx,cy,bottom_z]}], "color": name}]}
Frame: x right, y back, z up, metres. Yaw 0 faces -y, counter-clockwise seen from above.
Geometry truth stays the boxes: each mesh is fitted non-uniformly into its yaw-0 box and then rotated,
so silhouettes match the scorer's boxes in extent but not in exact shape. Writes out_dir/<view>.png.

Look follows the blender-skills guidance (MIT, github.com/arjun988/blender-skills): plausible PBR values,
motivated key/fill/rim lighting with consistent colour temperature, AgX view transform, Cycles + OIDN.
"""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
SEARCH = [HERE / 'glb', HERE.parents[2] / 'assets/kenney_furniture', HERE.parents[1] / 'assets/kenney_furniture']
PALETTE = {'red': (0.62, 0.08, 0.06), 'blue': (0.06, 0.18, 0.55), 'green': (0.10, 0.38, 0.12), 'yellow': (0.85, 0.62, 0.06),
           'white': (0.82, 0.82, 0.80), 'orange': (0.85, 0.30, 0.04), 'purple': (0.30, 0.10, 0.45), 'gray': (0.30, 0.30, 0.31),
           'brown': (0.30, 0.15, 0.06), 'black': (0.025, 0.025, 0.03), 'pink': (0.85, 0.35, 0.50), 'cyan': (0.05, 0.50, 0.60)}
# Native front of each model after glTF import, as a yaw (deg) that turns it to face -y. Kenney furniture already faces -y
# (verified on sofa, TV, chair renders), so the table is empty; add entries for other asset sources.
FRONT_YAW = {}
SAMPLES = 64


def find_model(name):
    for d in SEARCH:
        for p in d.glob('*.glb') if d.exists() else []:
            if p.stem.lower() == name.lower():
                return p
    raise FileNotFoundError(f'no GLB for model {name!r} in {[str(d) for d in SEARCH]}')


def principled(name, color, rough, metal=0., sheen=0.):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*color, 1)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    if sheen and 'Sheen Weight' in b.inputs:
        b.inputs['Sheen Weight'].default_value = sheen
    return m


def wood_material(name='MAT_Wood_Floor', tint=None, scale=(1., 9., 1.)):
    """Procedural oak: stretched noise grain through a colour ramp, raw-wood roughness 0.55-0.65."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes['Principled BSDF']
    tc, mp = nt.nodes.new('ShaderNodeTexCoord'), nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = scale
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 6.
    noise.inputs['Detail'].default_value = 8.
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    light, dark = (0.42, 0.25, 0.12), (0.24, 0.13, 0.05)
    if tint:
        light = tuple(0.5 * a + 0.5 * c for a, c in zip(light, tint))
        dark = tuple(0.5 * a + 0.5 * c * 0.6 for a, c in zip(dark, tint))
    ramp.color_ramp.elements[0].color = (*dark, 1)
    ramp.color_ramp.elements[1].color = (*light, 1)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], noise.inputs['Vector'])
    nt.links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = 0.6
    return m


def box_mesh(name, size, position, mat, bevel=0.006):
    sx, sy, sz = size
    bpy.ops.mesh.primitive_cube_add(size=1, location=(position[0], position[1], position[2] + sz / 2))
    o = bpy.context.object
    o.name = name
    o.scale = (sx, sy, sz)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    if bevel:
        mod = o.modifiers.new('bevel', 'BEVEL')
        mod.width, mod.segments = min(bevel, min(size) / 4), 2
    return o


def import_fitted(obj, palette):
    """Import the GLB, bake its transforms, fit it into the yaw-0 box, rotate by yaw, place on bottom_z."""
    box = obj['box']
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(find_model(obj['model'])))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == 'MESH']
    front = Matrix.Rotation(math.radians(FRONT_YAW.get(obj['model'].lower(), 0)), 4, 'Z')
    pts = [front @ o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    size, (cx, cy, z0), yaw = box['size'], box['position'], math.radians(box.get('yaw', 0))
    s = [size[i] / max(hi[i] - lo[i], 1e-6) for i in range(3)]
    c, sn = math.cos(yaw), math.sin(yaw)
    for o in meshes:
        m = front @ o.matrix_world
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
        if o.data.users > 1:
            o.data = o.data.copy()
        for v in o.data.vertices:
            q = m @ v.co
            x, y, z = ((q[0] - (lo[0] + hi[0]) / 2) * s[0], (q[1] - (lo[1] + hi[1]) / 2) * s[1], (q[2] - lo[2]) * s[2])
            v.co = (cx + x * c - y * sn, cy + x * sn + y * c, z0 + z)
        o.name = f'{obj["id"]}_{o.name}'
    for o in new:
        if o.type != 'MESH':
            bpy.data.objects.remove(o)
    if obj.get('color'):
        tint_dominant(meshes, palette[obj['color']])
    return meshes


def tint_dominant(meshes, rgb):
    """Recolour the material covering the largest surface area; keep secondary materials (legs, screens, soil)."""
    area = {}
    for o in meshes:
        for p in o.data.polygons:
            if o.material_slots:
                mat = o.material_slots[p.material_index].material
                area[mat] = area.get(mat, 0) + p.area
    if not area:
        return
    dom = max(area, key=area.get)
    new = dom.copy()
    new.name = 'MAT_Tint_' + dom.name
    b = new.node_tree.nodes.get('Principled BSDF') if new.use_nodes else None
    if b:
        b.inputs['Base Color'].default_value = (*rgb, 1)
        b.inputs['Roughness'].default_value = max(b.inputs['Roughness'].default_value, 0.45)   # painted / plastic
        b.inputs['Metallic'].default_value = 0.
    else:
        new.diffuse_color = (*rgb, 1)
    for o in meshes:
        for slot in o.material_slots:
            if slot.material == dom:
                slot.material = new


def build_room(W, D, H):
    floor_mat = wood_material()
    wall_mat = principled('MAT_Plaster_Wall', (0.78, 0.76, 0.71), 0.9)
    trim = principled('MAT_Paint_Trim', (0.92, 0.91, 0.88), 0.4)
    ground = principled('MAT_Concrete_Outside', (0.09, 0.09, 0.095), 0.9)
    box_mesh('ground_outside', (W + 40, D + 40, 0.01), (W / 2, D / 2, -0.06), ground, bevel=0)
    box_mesh('floor', (W, D, 0.05), (W / 2, D / 2, -0.05), floor_mat, bevel=0)
    back = [box_mesh('wall_back', (W + 0.12, 0.1, H), (W / 2 - 0.06, D + 0.05, 0), wall_mat, bevel=0),
            box_mesh('skirting_back', (W, 0.015, 0.08), (W / 2, D - 0.0075, 0), trim, bevel=0.003)]
    left = [box_mesh('wall_left', (0.1, D, H), (-0.05, D / 2, 0), wall_mat, bevel=0),
            box_mesh('skirting_left', (0.015, D, 0.08), (0.0075, D / 2, 0), trim, bevel=0.003)]
    return dict(back=back, left=left)


def lights(W, D, H):
    """Motivated rig: daylight 'window' key from front-left, soft cooler fill, warm rim from back-right, dim sky."""
    def area(name, loc, target, energy, size, kelvin_rgb):
        bpy.ops.object.light_add(type='AREA', location=loc)
        L = bpy.context.object
        L.name = name
        L.data.energy, L.data.size, L.data.color = energy, size, kelvin_rgb
        L.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    centre = (W / 2, D / 2, 0.4)
    k = W * D / 16.
    area('LGT_Key_Window', (-1.5, -2.0, H + 1.5), centre, 900 * k, 3.0, (1.0, 0.97, 0.92))     # ~5500K
    area('LGT_Fill_Soft', (W + 2.0, -1.5, H), centre, 280 * k, 4.0, (0.90, 0.94, 1.0))        # ~6500K
    area('LGT_Rim_Back', (W + 0.5, D + 1.5, H + 1.0), centre, 220 * k, 2.0, (1.0, 0.88, 0.74))  # ~3800K
    w = bpy.context.scene.world or bpy.data.worlds.new('World')
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.55, 0.62, 0.72, 1)
    bg.inputs['Strength'].default_value = 0.35


def setup_render(width, height):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    cy = sc.cycles
    cy.samples, cy.use_adaptive_sampling, cy.use_denoising = SAMPLES, True, True
    cy.denoiser = 'OPENIMAGEDENOISE'
    cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces, cy.transmission_bounces = 6, 3, 3, 4
    sc.render.use_persistent_data = True
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for kind in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                if any(d.type == kind for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = True
                    cy.device = 'GPU'
                    break
            except TypeError:
                continue
    except Exception:
        pass
    sc.view_settings.view_transform = 'AgX'
    try:
        sc.view_settings.look = 'AgX - Medium High Contrast'
    except TypeError:
        pass
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = width, height, 100
    sc.render.image_settings.file_format = 'PNG'


def camera(view):
    cam = bpy.data.objects.get('CAM_View')
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        cam.name = 'CAM_View'
        bpy.context.scene.camera = cam
    cam.data.sensor_fit = 'VERTICAL'
    cam.data.angle = math.radians(view.get('fov_deg', 50))
    cam.data.clip_start, cam.data.clip_end = 0.05, 200
    cam.location = view['eye']
    cam.rotation_euler = (Vector(view['target']) - Vector(view['eye'])).to_track_quat('-Z', 'Y').to_euler()
    return cam


def blocks(view, W, D, which):
    """Hide a wall whenever the camera is on its outer side (the wall would sit between eye and room)."""
    e = view['eye']
    return e[1] > D - 0.05 if which == 'back' else e[0] < 0.05


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    scene_path, out = Path(argv[0]).resolve(), Path(argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(scene_path.read_text(encoding='utf-8'))
    W, D, H = spec['room']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    walls = build_room(W, D, H)
    for obj in spec['objects']:
        if obj.get('parts'):
            tint = PALETTE.get(obj.get('color')) if obj.get('color') else None
            mat = wood_material(f'MAT_Wood_{obj["id"]}', tint, scale=(1., 1., 6.))
            for n, p in enumerate(obj['parts']):
                box_mesh(f'{obj["id"]}_part{n}', p['size'], p['position'], mat)
        else:
            import_fitted(obj, PALETTE)
    lights(W, D, H)
    for name, view in spec['camera_views'].items():
        setup_render(view.get('width', 640), view.get('height', 480))
        camera(view)
        for which, objs in walls.items():
            for o in objs:
                o.hide_render = blocks(view, W, D, which)
        bpy.context.scene.render.filepath = str(out / f'{name}.png')
        bpy.ops.render.render(write_still=True)
        print('rendered', name, flush=True)


if __name__ == '__main__':
    main()
