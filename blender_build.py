"""Construction des objets Blender à partir des données BFRES / BNTX / FSKA / KCL."""

import colorsys
import os
import zlib

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from . import bfres as bfres_mod
from . import bntx as bntx_mod
from . import byml as byml_mod

# Y-up (Switch) -> Z-up (Blender) : (x, y, z) -> (x, -z, y)
C3 = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
C4 = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))

UV_ATTRS = ("_u0", "_u1", "_u2", "_u3")
TEX_EXT = (".png", ".tga", ".dds", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class Settings:
    def __init__(self, scale=0.01, convert_axes=True, import_armature=True,
                 import_weights=True, import_normals=True, merge_collision=True,
                 texture_dir="", import_textures=True, import_animations=True):
        self.scale = scale
        self.convert_axes = convert_axes
        self.import_armature = import_armature
        self.import_weights = import_weights
        self.import_normals = import_normals
        self.merge_collision = merge_collision
        self.texture_dir = texture_dir
        self.import_textures = import_textures
        self.import_animations = import_animations
        self.warnings = []


# --------------------------------------------------------------------------- #
#  Conversions d'espace
# --------------------------------------------------------------------------- #
def conv_matrix(st):
    """Cs : espace jeu -> espace Blender (axes + échelle)."""
    s = Matrix.Diagonal((st.scale, st.scale, st.scale, 1.0))
    return (C4 @ s) if st.convert_axes else s


def conv(m, st):
    cs = conv_matrix(st)
    return cs @ m @ cs.inverted()


def unconv(m, st):
    cs = conv_matrix(st)
    return cs.inverted() @ m @ cs


def _to_blender(points, st):
    pts = np.asarray(points, np.float64)
    if st.convert_axes:
        pts = pts @ C3.T
    return pts * st.scale


def _dir_to_blender(vecs, st):
    v = np.asarray(vecs, np.float64)
    if st.convert_axes:
        v = v @ C3.T
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return v / n


def _new_mesh_object(name, verts, faces, collection):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    obj = bpy.data.objects.new(name, me)
    collection.objects.link(obj)
    return obj, me


def _finish_mesh(me, vertex_normals=None):
    me.validate(verbose=False, clean_customdata=False)
    if len(me.polygons):
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    if vertex_normals is not None and len(vertex_normals) == len(me.vertices):
        if hasattr(me, "use_auto_smooth"):  # Blender < 4.1
            me.use_auto_smooth = True
        try:
            me.normals_split_custom_set_from_vertices([tuple(n) for n in vertex_normals])
        except Exception:
            pass
    me.update()


def _child_collection(parent, name):
    col = bpy.data.collections.new(name)
    parent.children.link(col)
    return col


# --------------------------------------------------------------------------- #
#  Squelette
# --------------------------------------------------------------------------- #
def srt_matrix(scale, rotation, position, euler):
    if euler:
        rot = Euler(rotation[:3], "XYZ").to_matrix().to_4x4()
    else:
        x, y, z, w = rotation
        q = Quaternion((w, x, y, z))
        rot = (q.normalized() if q.magnitude > 1e-12 else Quaternion()).to_matrix().to_4x4()
    return (Matrix.Translation(position) @ rot
            @ Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0)))


def bone_world_matrices(skeleton):
    """Matrices monde (mathutils) de chaque os, dans l'espace du jeu."""
    worlds = []
    for b in skeleton.bones:
        local = srt_matrix(b.scale, b.rotation, b.position, b.is_euler)
        if 0 <= b.parent < len(worlds):
            worlds.append(worlds[b.parent] @ local)
        else:
            worlds.append(local)
    return worlds


def build_armature(model, worlds, st, collection, context):
    sk = model.skeleton
    arm = bpy.data.armatures.new(model.name + "_Skel")
    obj = bpy.data.objects.new(model.name + "_Skel", arm)
    collection.objects.link(obj)
    if hasattr(arm, "display_type"):
        arm.display_type = "STICK"

    heads = [_to_blender(np.array(w.to_translation())[None], st)[0] for w in worlds]
    children = [[] for _ in sk.bones]
    for i, b in enumerate(sk.bones):
        if 0 <= b.parent < len(sk.bones):
            children[b.parent].append(i)
    lengths = []
    for i in range(len(sk.bones)):
        d = [np.linalg.norm(heads[c] - heads[i]) for c in children[i]]
        d = [x for x in d if x > 1e-5]
        lengths.append(float(np.mean(d)) if d else 0.0)
    fallback = max(st.scale * 5.0, 0.01)
    for i, b in enumerate(sk.bones):
        if lengths[i] <= 1e-5:
            p = b.parent
            lengths[i] = lengths[p] * 0.6 if 0 <= p < len(lengths) and lengths[p] > 1e-5 else fallback

    view_layer = context.view_layer
    prev_active = view_layer.objects.active
    view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = []
    for i, b in enumerate(sk.bones):
        eb = arm.edit_bones.new(b.name or "Bone_%d" % i)
        eb.head = (0.0, 0.0, 0.0)
        eb.tail = (0.0, lengths[i], 0.0)
        rot = worlds[i].to_3x3().normalized()
        if st.convert_axes:
            rot = C4.to_3x3() @ rot @ C4.to_3x3().transposed()
        m = rot.to_4x4()
        m.translation = Vector(heads[i])
        eb.matrix = m
        ebs.append(eb)
    for i, b in enumerate(sk.bones):
        if 0 <= b.parent < len(ebs):
            ebs[i].parent = ebs[b.parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    view_layer.objects.active = prev_active

    # SRT d'origine (espace jeu) : utile aux animations et à l'export
    for i, b in enumerate(sk.bones):
        bone = arm.bones.get(b.name or "Bone_%d" % i)
        if bone is not None:
            bone["bfres_rest"] = list(b.scale) + list(b.rotation) + list(b.position)
            bone["bfres_euler"] = bool(b.is_euler)
            bone["bfres_index"] = i
    obj["bfres_scale"] = st.scale
    obj["bfres_convert_axes"] = st.convert_axes
    for pb in obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    return obj


# --------------------------------------------------------------------------- #
#  Textures
# --------------------------------------------------------------------------- #
class TextureLibrary:
    """Textures BNTX disponibles (par nom), décodées à la demande."""

    def __init__(self, st):
        self.st = st
        self.textures = {}
        self.images = {}
        self.files = _texture_index(st.texture_dir)

    def add_bntx(self, data, label):
        try:
            for t in bntx_mod.parse(data):
                self.textures.setdefault(t.name, t)
        except Exception as e:  # noqa: BLE001
            self.st.warnings.append("BNTX %s illisible : %s" % (label, e))

    def image(self, name, non_color):
        if not name:
            return None
        if name in self.images:
            return self.images[name]
        img = None
        tex = self.textures.get(name)
        if tex is not None and self.st.import_textures:
            try:
                px = bntx_mod.decode(tex)
                h, w = px.shape[:2]
                img = bpy.data.images.new(name, w, h, alpha=True)
                flt = (px[::-1].astype(np.float32) / 255.0).ravel()
                img.pixels.foreach_set(flt)
                img.pack()
                img["bntx_format"] = tex.format_name
            except Exception as e:  # noqa: BLE001
                self.st.warnings.append("Texture %s : %s" % (name, e))
                img = None
        if img is None:
            path = self.files.get(name.lower())
            if path:
                try:
                    img = bpy.data.images.load(path, check_existing=True)
                except RuntimeError:
                    img = None
        if img is not None:
            try:
                img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
            except TypeError:
                pass
        self.images[name] = img
        return img


def _texture_index(texture_dir):
    found = {}
    if texture_dir and os.path.isdir(texture_dir):
        for root, _, files in os.walk(texture_dir):
            for f in files:
                stem, ext = os.path.splitext(f)
                if ext.lower() in TEX_EXT:
                    found.setdefault(stem.lower(), os.path.join(root, f))
    return found


def texture_role(sampler, texname):
    s = (sampler or "").lower()
    t = (texname or "").lower()
    if s.startswith("_a") or "_alb" in t:
        return "albedo"
    if s.startswith("_n") or "_nrm" in t:
        return "normal"
    if s.startswith("_e") or "_emm" in t or "_emi" in t:
        return "emission"
    if "_rgh" in t or s.startswith("_r"):
        return "roughness"
    if "_mtl" in t or s.startswith("_m"):
        return "metallic"
    return None


def build_material(mat, texlib):
    bm = bpy.data.materials.new(mat.name or "Material")
    bm["bfres_textures"] = ", ".join(mat.textures)
    if mat.samplers:
        bm["bfres_samplers"] = ", ".join(mat.samplers)
    bm.use_nodes = True
    tree = bm.node_tree
    bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return bm

    done = set()
    y = 300
    for i, tex in enumerate(mat.textures):
        sampler = mat.samplers[i] if i < len(mat.samplers) else ""
        role = texture_role(sampler, tex)
        if role is None and i == 0:
            role = "albedo"
        if role is None or role in done:
            continue
        img = texlib.image(tex, non_color=role not in ("albedo", "emission"))
        if img is None:
            continue
        done.add(role)
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = img
        node.label = "%s (%s)" % (tex, sampler or role)
        node.location = (-500, y)
        y -= 300
        inp = bsdf.inputs.get
        if role == "albedo":
            tree.links.new(node.outputs["Color"], inp("Base Color"))
            if inp("Alpha") is not None:
                tree.links.new(node.outputs["Alpha"], inp("Alpha"))
        elif role == "normal":
            nm = tree.nodes.new("ShaderNodeNormalMap")
            nm.location = (-200, y + 300)
            tree.links.new(node.outputs["Color"], nm.inputs["Color"])
            tree.links.new(nm.outputs["Normal"], inp("Normal"))
        elif role == "emission":
            target = inp("Emission Color") or inp("Emission")
            if target is not None:
                tree.links.new(node.outputs["Color"], target)
                if inp("Emission Strength") is not None:
                    inp("Emission Strength").default_value = 1.0
        elif role == "roughness" and inp("Roughness") is not None:
            tree.links.new(node.outputs["Color"], inp("Roughness"))
        elif role == "metallic" and inp("Metallic") is not None:
            tree.links.new(node.outputs["Color"], inp("Metallic"))
    return bm


# --------------------------------------------------------------------------- #
#  Shapes
# --------------------------------------------------------------------------- #
def _assign_weights(obj, shape, attrs, used, sk):
    bones = sk.bones
    mtb = np.array(sk.matrix_to_bone, dtype=np.int64) if sk.matrix_to_bone else np.zeros(0, np.int64)
    n = len(used)
    groups = {}

    def group(bone_idx):
        name = bones[bone_idx].name or "Bone_%d" % bone_idx
        if name not in groups:
            groups[name] = obj.vertex_groups.new(name=name)
        return groups[name]

    def add(bone_ids, weights):
        valid = (bone_ids >= 0) & (bone_ids < len(bones)) & (weights > 1e-6)
        vid = np.arange(n)[valid]
        bid = bone_ids[valid]
        wq = np.round(weights[valid], 4)
        if not len(vid):
            return
        keys = np.stack([bid, (wq * 10000).astype(np.int64)], axis=1)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        for k, (b, w) in enumerate(uniq):
            group(int(b)).add(vid[inv == k].tolist(), float(w) / 10000.0, "ADD")

    if not bones:
        return
    if shape.skin_count == 0:
        if 0 <= shape.bone_index < len(bones):
            add(np.full(n, shape.bone_index, np.int64), np.ones(n))
        return

    i0 = attrs.get("_i0")
    if i0 is None or not len(mtb):
        return
    idx = [i0[used]]
    if "_i1" in attrs:
        idx.append(attrs["_i1"][used])
    idx = np.concatenate(idx, axis=1).astype(np.int64)
    w = [attrs["_w0"][used]] if "_w0" in attrs else []
    if "_w1" in attrs:
        w.append(attrs["_w1"][used])
    w = np.concatenate(w, axis=1) if w else None

    for k in range(min(shape.skin_count, idx.shape[1])):
        mi = idx[:, k]
        bone_ids = np.where((mi >= 0) & (mi < len(mtb)), mtb[np.clip(mi, 0, len(mtb) - 1)], -1)
        if shape.skin_count == 1 or w is None:
            weights = np.ones(n) if k == 0 else np.zeros(n)
        else:
            weights = w[:, k]
        add(bone_ids, weights)


def build_shape(model, shape, worlds, materials, st, collection, armature_obj):
    if shape.vertex_buffer_index >= len(model.vertex_buffers) or not shape.meshes:
        return None
    vb = model.vertex_buffers[shape.vertex_buffer_index]

    attrs = {}
    for name, a in vb.attributes.items():
        dec = bfres_mod.decode_attribute(vb, a)
        if dec is not None:
            attrs[name] = dec
    if "_p0" not in attrs:
        return None
    nvert = len(attrs["_p0"])

    tris = bfres_mod.mesh_triangles(shape.meshes[0])
    tris = tris[np.all((tris >= 0) & (tris < nvert), axis=1)]
    tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]
    if not len(tris):
        return None

    used, remap = np.unique(tris, return_inverse=True)
    faces = remap.reshape(tris.shape)

    pos = attrs["_p0"][used, :3].astype(np.float64)
    nrm = attrs["_n0"][used, :3].astype(np.float64) if "_n0" in attrs else None

    # Transformations rigides (comme BfresSwitch.cs de Switch Toolbox)
    bones = model.skeleton.bones
    np_worlds = [np.array(m) for m in worlds]
    if bones:
        if shape.skin_count == 0 and 0 <= shape.bone_index < len(bones):
            M = np_worlds[shape.bone_index]
            pos = pos @ M[:3, :3].T + M[:3, 3]
            if nrm is not None:
                nrm = nrm @ M[:3, :3].T
        elif shape.skin_count == 1 and "_i0" in attrs and model.skeleton.matrix_to_bone:
            mtb = np.array(model.skeleton.matrix_to_bone, np.int64)
            mi = attrs["_i0"][used, 0].astype(np.int64)
            bidx = np.where((mi >= 0) & (mi < len(mtb)), mtb[np.clip(mi, 0, len(mtb) - 1)],
                            shape.bone_index)
            for b in np.unique(bidx):
                if not (0 <= b < len(bones)) or bones[b].rigid_index == -1:
                    continue
                sel = bidx == b
                M = np_worlds[b]
                pos[sel] = pos[sel] @ M[:3, :3].T + M[:3, 3]
                if nrm is not None:
                    nrm[sel] = nrm[sel] @ M[:3, :3].T

    verts = _to_blender(pos, st)
    obj, me = _new_mesh_object(shape.name or "Shape", verts, faces, collection)

    if 0 <= shape.material_index < len(materials):
        me.materials.append(materials[shape.material_index])

    loop_v = np.zeros(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", loop_v)
    uv_sets = []
    for name in UV_ATTRS:
        if name in attrs:
            uv_sets.append(("UV" + name[-1], attrs[name][used, :2]))
    for packed, names in (("_g3d_02_u0_u1", ("UV0", "UV1")), ("_g3d_02_u2_u3", ("UV2", "UV3"))):
        if packed in attrs and not any(n == names[0] for n, _ in uv_sets):
            uv_sets.append((names[0], attrs[packed][used, 0:2]))
            uv_sets.append((names[1], attrs[packed][used, 2:4]))
    for name, uv in uv_sets:
        layer = me.uv_layers.new(name=name)
        if layer is None:
            break
        uv = uv.astype(np.float64).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        layer.data.foreach_set("uv", uv[loop_v].ravel())

    for name in ("_c0", "_c1"):
        if name in attrs and hasattr(me, "color_attributes"):
            col = me.color_attributes.new(name="Col" + name[-1], type="FLOAT_COLOR", domain="POINT")
            col.data.foreach_set("color", attrs[name][used].astype(np.float32).ravel())

    normals = _dir_to_blender(nrm, st) if (nrm is not None and st.import_normals) else None
    _finish_mesh(me, normals)

    obj["bfres_shape"] = shape.name
    obj["bfres_skin_count"] = shape.skin_count
    obj["bfres_attributes"] = ", ".join(sorted(vb.attributes))

    if armature_obj is not None:
        if st.import_weights:
            _assign_weights(obj, shape, attrs, used, model.skeleton)
        obj.parent = armature_obj
        mod = obj.modifiers.new("Armature", "ARMATURE")
        mod.object = armature_obj
    return obj


# --------------------------------------------------------------------------- #
#  Animations
# --------------------------------------------------------------------------- #
def _new_fcurve(action, obj, path, index, group):
    if hasattr(action, "fcurve_ensure_for_datablock"):
        return action.fcurve_ensure_for_datablock(obj, path, index=index, group_name=group)
    fc = action.fcurves.new(path, index=index, action_group=group)
    return fc


def _rest_local_game(bone, st):
    """SRT locale de repos (espace jeu)."""
    if "bfres_rest" in bone:
        v = list(bone["bfres_rest"])
        return v[0:3], v[3:7], v[7:10], bool(bone.get("bfres_euler", False))
    rest_b = bone.matrix_local if bone.parent is None else bone.parent.matrix_local.inverted() @ bone.matrix_local
    m = unconv(rest_b, st)
    t, q, s = m.decompose()
    return list(s), [q.x, q.y, q.z, q.w], list(t), False


def import_skeletal_anim(anim, arm_obj, st):
    arm = arm_obj.data
    action = bpy.data.actions.new(anim.name or "Anim")
    action.use_fake_user = True
    arm_obj.animation_data_create()
    arm_obj.animation_data.action = action

    frames = np.arange(0, max(anim.frame_count, 1) + 1, dtype=np.float64)
    matched = 0
    for ba in anim.bone_anims:
        bone = arm.bones.get(ba.name)
        if bone is None:
            continue
        matched += 1
        rs, rr, rt, r_euler = _rest_local_game(bone, st)
        euler = anim.euler

        # valeurs de base
        base_s = list(ba.base_scale) if ba.base_scale else list(rs)
        if ba.base_rotate:
            base_r = list(ba.base_rotate)
        elif euler == r_euler:
            base_r = list(rr)
        else:
            q = (Euler(rr[:3], "XYZ").to_quaternion() if r_euler
                 else Quaternion((rr[3], rr[0], rr[1], rr[2])))
            base_r = list(q.to_euler("XYZ")) + [0.0] if euler else [q.x, q.y, q.z, q.w]
        base_t = list(ba.base_translate) if ba.base_translate else list(rt)

        n = len(frames)
        S = np.tile(base_s, (n, 1))
        R = np.tile(base_r, (n, 1))
        T = np.tile(base_t, (n, 1))
        for cv in ba.curves:
            tgt = bfres_mod.ANIM_TARGETS.get(cv.target)
            if tgt is None:
                continue
            arr = {"S": S, "R": R, "T": T}[tgt[0]]
            arr[:, tgt[1]] = [cv.evaluate(f) for f in frames]

        rest_b = bone.matrix_local if bone.parent is None else \
            bone.parent.matrix_local.inverted() @ bone.matrix_local
        rest_inv = rest_b.inverted()
        locs, quats, scales = [], [], []
        prev = None
        for i in range(n):
            L = srt_matrix(S[i], R[i], T[i], euler)
            basis = rest_inv @ conv(L, st)
            loc, q, sc = basis.decompose()
            if prev is not None and prev.dot(q) < 0:
                q = -q
            prev = q
            locs.append(loc)
            quats.append(q)
            scales.append(sc)

        pb_path = 'pose.bones["%s"]' % bone.name.replace('"', '\\"')
        for prop, values, comps in (("location", locs, 3), ("rotation_quaternion", quats, 4),
                                    ("scale", scales, 3)):
            for c in range(comps):
                fc = _new_fcurve(action, arm_obj, pb_path + "." + prop, c, bone.name)
                if fc is None:
                    continue
                co = np.empty(n * 2, np.float32)
                co[0::2] = frames
                co[1::2] = [v[c] for v in values]
                fc.keyframe_points.add(n)
                fc.keyframe_points.foreach_set("co", co)
                fc.update()
    action["bfres_frame_count"] = anim.frame_count
    action["bfres_loop"] = anim.loop
    return action, matched


def import_animations(res, arm_obj, st):
    actions = []
    for anim in res.skeletal_anims:
        try:
            action, matched = import_skeletal_anim(anim, arm_obj, st)
        except Exception as e:  # noqa: BLE001
            st.warnings.append("Animation %s : %s" % (anim.name, e))
            continue
        if matched == 0:
            bpy.data.actions.remove(action)
            st.warnings.append("Animation %s : aucun os correspondant" % anim.name)
        else:
            actions.append(action)
    if actions:
        arm_obj.animation_data.action = actions[0]
        scene = bpy.context.scene
        scene.frame_start = 0
        scene.frame_end = max(scene.frame_end, int(actions[0].get("bfres_frame_count", 0)))
    return actions


# --------------------------------------------------------------------------- #
#  Entrées principales
# --------------------------------------------------------------------------- #
def import_bfres(res, label, st, parent_collection, context, texlib, anim_target=None):
    col = _child_collection(parent_collection, label)
    col["szs_bfres"] = label
    for name, data in res.external_files.items():
        if data[:4] == b"BNTX":
            texlib.add_bntx(data, name)

    created = 0
    first_arm = None
    for mi, model in enumerate(res.models):
        mcol = _child_collection(col, model.name) if len(res.models) > 1 else col
        worlds = bone_world_matrices(model.skeleton)
        arm_obj = None
        if st.import_armature and model.skeleton.bones:
            arm_obj = build_armature(model, worlds, st, mcol, context)
            arm_obj["szs_bfres"] = label
            arm_obj["szs_model"] = mi
            first_arm = first_arm or arm_obj
        materials = [build_material(m, texlib) for m in model.materials]
        for si, shape in enumerate(model.shapes):
            obj = build_shape(model, shape, worlds, materials, st, mcol, arm_obj)
            if obj is not None:
                obj["szs_bfres"] = label
                obj["szs_model"] = mi
                obj["szs_shape"] = si
                created += 1

    target = first_arm or anim_target
    if st.import_animations and res.skeletal_anims:
        if target is None:
            st.warnings.append("%s : %d animation(s) ignorée(s) (pas d'armature : sélectionnez-en une)"
                               % (label, len(res.skeletal_anims)))
        else:
            acts = import_animations(res, target, st)
            created += len(acts)
    return created


def _attr_color(attr):
    h = (zlib.crc32(int(attr).to_bytes(2, "little")) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.9)
    return (r, g, b, 1.0)


def kcl_material(code, entries):
    """Matériau Blender pour un code d'attribut KCL (avec noms SMO si disponibles)."""
    entry = entries[code] if entries is not None and 0 <= code < len(entries) else None
    if entry is not None:
        name = "KCL_%02d %s" % (code, byml_mod.attribute_label(entry))
    else:
        name = "KCL_0x%04X" % code
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = _attr_color(code)
    mat["kcl_attribute"] = int(code)
    if entry is not None and hasattr(mat, "smo_kcl"):
        mat.smo_kcl.is_kcl = True
        for k in byml_mod.SMO_CODE_KEYS:
            setattr(mat.smo_kcl, k, entry.get(k, ""))
    return mat


def import_kcl(kcl_file, label, st, parent_collection, entries=None):
    col = _child_collection(parent_collection, label)
    col["szs_kcl"] = label
    parts = []
    for i, m in enumerate(kcl_file.models):
        tris, attr = m.triangles()
        if len(tris):
            parts.append((i, tris, attr))
    if not parts:
        return 0
    if st.merge_collision:
        parts = [(None, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]))]

    mats = {}
    created = 0
    for idx, tris, attr in parts:
        flat = _to_blender(tris.reshape(-1, 3), st)
        key = np.round(flat, 5)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        faces = inv.reshape(-1, 3)
        ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
        faces, attr = faces[ok], attr[ok]
        if not len(faces):
            continue
        name = label.rsplit(".", 1)[0] + ("" if idx is None else "_%d" % idx)
        obj, me = _new_mesh_object(name, uniq, faces, col)
        codes = sorted(set(int(a) for a in attr))
        slot = {}
        for c in codes:
            if c not in mats:
                mats[c] = kcl_material(c, entries)
            slot[c] = len(me.materials)
            me.materials.append(mats[c])
        me.polygons.foreach_set("material_index", [slot[int(a)] for a in attr])
        fa = me.attributes.new("kcl_attribute", "INT", "FACE")
        fa.data.foreach_set("value", [int(a) for a in attr])
        _finish_mesh(me, None)
        me.polygons.foreach_set("use_smooth", [False] * len(me.polygons))
        obj["szs_kcl"] = label
        created += 1
    return created
