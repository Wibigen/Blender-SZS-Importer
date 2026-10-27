"""Export : objets Blender -> géométries BFRES / triangles KCL / attributs BYML -> .szs."""

import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

from . import bfres as bfres_mod
from . import bfres_writer
from . import byml as byml_mod
from . import kcl as kcl_mod
from . import sarc as sarc_mod
from .blender_build import C3, bone_world_matrices


class ExportError(Exception):
    pass


class ExportSettings:
    def __init__(self, scale=0.01, convert_axes=True, export_model=True, export_collision=True,
                 hide_missing=True, yaz0_level=1, kcl_settings=None):
        self.scale = scale
        self.convert_axes = convert_axes
        self.export_model = export_model
        self.export_collision = export_collision
        self.hide_missing = hide_missing
        self.yaz0_level = yaz0_level
        self.kcl_settings = kcl_settings or kcl_mod.KclSettings()
        self.report = []


def _to_game(points, st):
    p = np.asarray(points, np.float64) / st.scale
    return p @ C3 if st.convert_axes else p  # C3 orthogonale : C3.T^-1 = C3


def _dir_to_game(v, st):
    v = np.asarray(v, np.float64)
    if st.convert_axes:
        v = v @ C3
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return v / n


def _strip_suffix(name):
    base, dot, num = name.rpartition(".")
    return base if dot and num.isdigit() and len(num) == 3 else name


# --------------------------------------------------------------------------- #
#  Lecture d'un mesh Blender
# --------------------------------------------------------------------------- #
class MeshData:
    """Données « par sommet » (déjà dédoublonnées par UV/normale) en espace jeu."""

    def __init__(self):
        self.pos = None
        self.nrm = None
        self.tan = None
        self.bitan = None
        self.uvs = {}
        self.colors = {}
        self.weights = []  # par sommet : liste (nom de groupe, poids)
        self.tris = None


def _triangulated_copy(obj, use_modifiers=False, depsgraph=None):
    if use_modifiers and depsgraph is not None:
        ev = obj.evaluated_get(depsgraph)
        src = ev.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(src)
        ev.to_mesh_clear()
    else:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new("__szs_tmp")
    bm.to_mesh(me)
    bm.free()
    return me


def read_mesh(obj, matrix, st, need_tangents=False):
    me = _triangulated_copy(obj)
    try:
        nl = len(me.loops)
        if nl == 0:
            return None
        loop_v = np.zeros(nl, np.int64)
        me.loops.foreach_get("vertex_index", loop_v)
        co = np.zeros(len(me.vertices) * 3, np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        m = np.array(matrix)
        co = co @ m[:3, :3].T + m[:3, 3]
        nmat = np.array(matrix.to_3x3().inverted_safe().transposed())

        if hasattr(me, "corner_normals"):
            ln = np.zeros(nl * 3, np.float64)
            me.corner_normals.foreach_get("vector", ln)
        else:
            me.calc_normals_split()
            ln = np.zeros(nl * 3, np.float64)
            me.loops.foreach_get("normal", ln)
        ln = ln.reshape(-1, 3) @ nmat.T

        uv_layers = {}
        for layer in me.uv_layers:
            uv = np.zeros(nl * 2, np.float64)
            layer.data.foreach_get("uv", uv)
            uv_layers[layer.name] = uv.reshape(-1, 2)

        colors = {}
        if hasattr(me, "color_attributes"):
            for ca in me.color_attributes:
                c = np.zeros(len(ca.data) * 4, np.float64)
                ca.data.foreach_get("color", c)
                c = c.reshape(-1, 4)
                colors[ca.name] = c[loop_v] if ca.domain == "POINT" else c

        tan = bitan = None
        if need_tangents and me.uv_layers:
            try:
                me.calc_tangents(uvmap=me.uv_layers[0].name)
                tan = np.zeros(nl * 3, np.float64)
                me.loops.foreach_get("tangent", tan)
                sign = np.zeros(nl, np.float64)
                me.loops.foreach_get("bitangent_sign", sign)
                tan = tan.reshape(-1, 3) @ m[:3, :3].T
                bitan = np.cross(ln, tan) * sign[:, None]
                tan = np.c_[tan, sign]
            except RuntimeError:
                tan = bitan = None

        # Dédoublonnage : un sommet BFRES par (sommet, normale, UV, couleur)
        feats = [loop_v[:, None].astype(np.float64), np.round(ln, 4)]
        for name in sorted(uv_layers):
            feats.append(np.round(uv_layers[name], 5))
        for name in sorted(colors):
            feats.append(np.round(colors[name], 3))
        feat = np.concatenate(feats, axis=1)
        _, first, inv = np.unique(feat, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)

        me.calc_loop_triangles()
        tri_loops = np.zeros(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("loops", tri_loops)
        tris = inv[tri_loops].reshape(-1, 3)

        md = MeshData()
        src_v = loop_v[first]
        md.pos = _to_game(co[src_v], st)
        md.nrm = _dir_to_game(ln[first], st)
        for name, uv in uv_layers.items():
            u = uv[first].copy()
            u[:, 1] = 1.0 - u[:, 1]
            md.uvs[name] = u
        for name, c in colors.items():
            md.colors[name] = c[first]
        if tan is not None:
            t = tan[first]
            md.tan = np.c_[_dir_to_game(t[:, :3], st), t[:, 3]]
            md.bitan = _dir_to_game(bitan[first], st)
        groups = {g.index: g.name for g in obj.vertex_groups}
        per_v = []
        for v in me.vertices:
            per_v.append([(groups[g.group], g.weight) for g in v.groups if g.group in groups and g.weight > 0])
        md.weights = [per_v[i] for i in src_v]
        md.tris = tris
        return md
    finally:
        bpy.data.meshes.remove(me)


def _merge_meshdata(parts):
    if len(parts) == 1:
        return parts[0]
    out = MeshData()
    off = 0
    tris = []
    for p in parts:
        tris.append(p.tris + off)
        off += len(p.pos)
    out.pos = np.concatenate([p.pos for p in parts])
    out.nrm = np.concatenate([p.nrm for p in parts])
    out.tris = np.concatenate(tris)
    names = set().union(*[p.uvs.keys() for p in parts])
    for n in names:
        out.uvs[n] = np.concatenate([p.uvs.get(n, np.zeros((len(p.pos), 2))) for p in parts])
    names = set().union(*[p.colors.keys() for p in parts])
    for n in names:
        out.colors[n] = np.concatenate([p.colors.get(n, np.ones((len(p.pos), 4))) for p in parts])
    if all(p.tan is not None for p in parts):
        out.tan = np.concatenate([p.tan for p in parts])
        out.bitan = np.concatenate([p.bitan for p in parts])
    out.weights = sum((p.weights for p in parts), [])
    return out


# --------------------------------------------------------------------------- #
#  Shape BFRES
# --------------------------------------------------------------------------- #
def _weight_quant(vb):
    a = vb.attributes.get("_w0")
    if a is None:
        return None
    rc = bfres_mod._resolve_format(a.format)
    if rc is None:
        return None
    base, _, kind = bfres_mod.ATTRIB_FORMATS[rc]
    if kind == "unorm":
        return float(np.iinfo(np.dtype(base)).max)
    return None


def build_geometry(model, shape, md, st, material_index=None):
    sk = model.skeleton
    bones = sk.bones
    vb = model.vertex_buffers[shape.vertex_buffer_index]
    n = len(md.pos)
    name_to_bone = {b.name: i for i, b in enumerate(bones)}
    worlds = [np.array(w) for w in bone_world_matrices(sk)] if bones else []

    pos = md.pos.copy()
    nrm = md.nrm.copy()
    tan = md.tan.copy() if md.tan is not None else None
    bitan = md.bitan.copy() if md.bitan is not None else None
    geo = bfres_writer.ShapeGeometry()

    def to_local(sel, M):
        inv = np.linalg.inv(M)
        R = inv[:3, :3]
        pos[sel] = pos[sel] @ R.T + inv[:3, 3]
        nrm[sel] = _norm(nrm[sel] @ M[:3, :3])  # inverse transposée de inv(M)
        if tan is not None:
            tan[sel, :3] = _norm(tan[sel, :3] @ R.T)
        if bitan is not None:
            bitan[sel] = _norm(bitan[sel] @ R.T)

    model_space = pos.copy()
    skin = shape.skin_count
    idx = np.zeros((n, 8), np.float64)
    wts = np.zeros((n, 8), np.float64)

    def bone_weights(i):
        lst = [(name_to_bone[g], w) for g, w in md.weights[i] if g in name_to_bone]
        lst.sort(key=lambda x: -x[1])
        return lst

    if bones and skin == 0:
        if 0 <= shape.bone_index < len(bones):
            to_local(np.ones(n, bool), worlds[shape.bone_index])
    elif bones and skin == 1:
        dom = np.full(n, shape.bone_index, np.int64)
        for i in range(n):
            bw = bone_weights(i)
            if bw:
                dom[i] = bw[0][0]
        for b in np.unique(dom):
            bone = bones[b]
            mi = bone.rigid_index if bone.rigid_index >= 0 else bone.smooth_index
            if mi < 0:
                bone = bones[shape.bone_index]
                mi = max(bone.rigid_index, bone.smooth_index, 0)
            sel = dom == b
            idx[sel, 0] = mi
            wts[sel, 0] = 1.0
            if bones[b].rigid_index != -1:
                to_local(sel, worlds[b])
    elif bones and skin >= 2:
        q = _weight_quant(vb)
        fallback = bones[shape.bone_index].smooth_index if 0 <= shape.bone_index < len(bones) else 0
        missing = 0
        for i in range(n):
            bw = [(b, w) for b, w in bone_weights(i) if bones[b].smooth_index >= 0][:skin]
            if not bw:
                bw = [(None, 1.0)]
                missing += 1
            total = sum(w for _, w in bw) or 1.0
            ws = [w / total for _, w in bw]
            if q:
                qi = [int(round(w * q)) for w in ws]
                qi[0] += int(q) - sum(qi)
                ws = [v / q for v in qi]
            for k, ((b, _), w) in enumerate(zip(bw, ws)):
                idx[i, k] = fallback if b is None else bones[b].smooth_index
                wts[i, k] = w
        if missing:
            st.report.append("%s : %d sommet(s) sans poids -> os %s" % (
                shape.name, missing, bones[shape.bone_index].name if bones else "?"))

    # Attributs
    n_w = 0.0
    if "_n0" in vb.attributes:
        orig_n = bfres_mod.decode_attribute(vb, vb.attributes["_n0"])
        if orig_n is not None and len(orig_n):
            vals, counts = np.unique(orig_n[:, 3], return_counts=True)
            n_w = float(vals[np.argmax(counts)])
    attrs = {"_p0": np.c_[pos, np.zeros(n)], "_n0": np.c_[nrm, np.full(n, n_w)]}
    if tan is not None:
        attrs["_t0"] = tan
    if bitan is not None:
        attrs["_b0"] = np.c_[bitan, np.zeros(n)]
    uv_names = sorted(md.uvs)
    for k, name in enumerate(uv_names[:4]):
        slot = int(name[-1]) if name[-1:].isdigit() and name.upper().startswith("UV") else k
        attrs["_u%d" % slot] = np.c_[md.uvs[name], np.zeros((n, 2))]
    for packed, a, b in (("_g3d_02_u0_u1", "_u0", "_u1"), ("_g3d_02_u2_u3", "_u2", "_u3")):
        if packed in vb.attributes:
            ua = attrs.get(a, np.zeros((n, 4)))
            ub = attrs.get(b, np.zeros((n, 4)))
            attrs[packed] = np.c_[ua[:, :2], ub[:, :2]]
    for name, c in md.colors.items():
        if name.lower().startswith("col") and name[-1:].isdigit():
            attrs["_c" + name[-1]] = c
    if "_c0" not in attrs and md.colors:
        attrs["_c0"] = next(iter(md.colors.values()))
    if skin >= 1:
        attrs["_i0"] = idx[:, 0:4]
        attrs["_w0"] = wts[:, 0:4]
        attrs["_i1"] = idx[:, 4:8]
        attrs["_w1"] = wts[:, 4:8]

    bb_src = pos if skin == 0 else model_space
    mn, mx = bb_src.min(axis=0), bb_src.max(axis=0)
    center = (mn + mx) / 2.0
    geo.attrs = attrs
    geo.triangles = md.tris
    geo.vertex_count = n
    geo.bbox_center = tuple(center)
    geo.bbox_extent = tuple((mx - mn) / 2.0)
    geo.radius = float(np.linalg.norm(bb_src - center, axis=1).max())
    geo.material_index = material_index
    return geo


def _norm(v):
    l = np.linalg.norm(v, axis=1, keepdims=True)
    l[l < 1e-12] = 1.0
    return v / l


# --------------------------------------------------------------------------- #
#  Collecte de la scène
# --------------------------------------------------------------------------- #
def _collection_objects(col):
    objs = list(col.objects)
    for child in col.children:
        objs += _collection_objects(child)
    return objs


def _find_collections(col, key):
    out = []
    if key in col:
        out.append(col)
    for c in col.children:
        out += _find_collections(c, key)
    return out


def _object_matrix(obj):
    m = obj.matrix_world.copy()
    p = obj.parent
    if p is not None and p.type == "ARMATURE":
        m = p.matrix_world.inverted() @ m
    return m


def gather_bfres_geometries(root_col, files, st):
    """Retourne {nom du .bfres : nouvelles données}."""
    results = {}
    for col in _find_collections(root_col, "szs_bfres"):
        label = col["szs_bfres"]
        if label not in files:
            st.report.append("%s : absent de l'archive d'origine, ignoré" % label)
            continue
        data = files[label]
        res = bfres_mod.parse(data, lod_all=True)
        groups = {}
        unknown = []
        for obj in _collection_objects(col):
            if obj.type != "MESH":
                continue
            if "szs_shape" not in obj or obj.get("szs_bfres", label) != label:
                unknown.append(obj.name)
                continue
            groups.setdefault((int(obj["szs_model"]), int(obj["szs_shape"])), []).append(obj)
        for name in unknown:
            st.report.append("%s : nouvel objet ignoré (fusionnez-le dans un shape existant)" % name)

        geos = {}
        for (mi, si), objs in groups.items():
            if mi >= len(res.models) or si >= len(res.models[mi].shapes):
                continue
            model = res.models[mi]
            shape = model.shapes[si]
            vb = model.vertex_buffers[shape.vertex_buffer_index]
            need_t = "_t0" in vb.attributes or "_b0" in vb.attributes
            parts = []
            for o in objs:
                md = read_mesh(o, _object_matrix(o), st, need_tangents=need_t)
                if md is not None and len(md.tris):
                    parts.append(md)
            if not parts:
                g = bfres_writer.ShapeGeometry()
                g.hidden = True
                geos[(mi, si)] = g
                continue
            md = _merge_meshdata(parts)
            mat_index = None
            mats = [s.material for s in objs[0].material_slots if s.material]
            if mats:
                names = [m.name for m in model.materials]
                nm = _strip_suffix(mats[0].name)
                if nm in names:
                    mat_index = names.index(nm)
            geos[(mi, si)] = build_geometry(model, shape, md, st, mat_index)

        if st.hide_missing:
            for mi, model in enumerate(res.models):
                for si, shape in enumerate(model.shapes):
                    if (mi, si) not in geos and (mi, si) not in groups:
                        g = bfres_writer.ShapeGeometry()
                        g.hidden = True
                        geos[(mi, si)] = g
                        st.report.append("%s : shape supprimé dans Blender -> masqué" % shape.name)
        new = bfres_writer.rebuild(data, res, geos)
        errs = bfres_writer.validate(new)
        if errs:
            raise ExportError("BFRES %s invalide après reconstruction : %s" % (label, "; ".join(errs[:5])))
        results[label] = new
    return results


def _material_attribute(mat, entries, changed):
    """Code d'attribut KCL d'un matériau."""
    if mat is None:
        return None
    if entries is not None and hasattr(mat, "smo_kcl") and mat.smo_kcl.is_kcl:
        e = {k: getattr(mat.smo_kcl, k) for k in byml_mod.SMO_CODE_KEYS}
        e = {k: v for k, v in e.items() if v or k != "MaterialCodePrefix"}
        for i, ent in enumerate(entries):
            norm = {k: v for k, v in ent.items() if v or k != "MaterialCodePrefix"}
            if norm == e:
                return i
        entries.append(e)
        changed[0] = True
        return len(entries) - 1
    if "kcl_attribute" in mat:
        return int(mat["kcl_attribute"])
    nm = mat.name
    if nm.startswith("KCL_0x"):
        try:
            return int(nm[6:10], 16)
        except ValueError:
            return None
    return None


def gather_kcl(root_col, files, st, depsgraph):
    results = {}
    for col in _find_collections(root_col, "szs_kcl"):
        label = col["szs_kcl"]
        stem = label.rsplit(".", 1)[0]
        attr_name = None
        for cand in (stem + "Attribute.byml", stem.rsplit("/", 1)[0] + "Attribute.byml"):
            if cand in files:
                attr_name = cand
                break
        entries = None
        byml_meta = None
        if attr_name:
            raw = files[attr_name]
            entries = byml_mod.attribute_entries(byml_mod.parse(raw))
            byml_meta = (byml_mod.endian_of(raw), byml_mod.version_of(raw))
        changed = [False]

        all_tris, all_attr = [], []
        for obj in _collection_objects(col):
            if obj.type != "MESH":
                continue
            me = _triangulated_copy(obj, use_modifiers=True, depsgraph=depsgraph)
            try:
                if not len(me.polygons):
                    continue
                m = np.array(obj.matrix_world)
                co = np.zeros(len(me.vertices) * 3)
                me.vertices.foreach_get("co", co)
                co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
                me.calc_loop_triangles()
                vi = np.zeros(len(me.loop_triangles) * 3, np.int64)
                me.loop_triangles.foreach_get("vertices", vi)
                pidx = np.zeros(len(me.loop_triangles), np.int64)
                me.loop_triangles.foreach_get("polygon_index", pidx)
                mat_idx = np.zeros(len(me.polygons), np.int64)
                me.polygons.foreach_get("material_index", mat_idx)
                face_attr = None
                if "kcl_attribute" in me.attributes and me.attributes["kcl_attribute"].domain == "FACE":
                    face_attr = np.zeros(len(me.polygons), np.int64)
                    me.attributes["kcl_attribute"].data.foreach_get("value", face_attr)
                slot_codes = []
                for s in obj.material_slots:
                    slot_codes.append(_material_attribute(s.material, entries, changed))
                codes = np.zeros(len(me.polygons), np.int64)
                for p in range(len(me.polygons)):
                    c = slot_codes[mat_idx[p]] if mat_idx[p] < len(slot_codes) else None
                    if c is None:
                        c = int(face_attr[p]) if face_attr is not None else 0
                    codes[p] = c
                all_tris.append(_to_game(co[vi], st).reshape(-1, 3, 3))
                all_attr.append(codes[pidx])
            finally:
                bpy.data.meshes.remove(me)
        if not all_tris:
            st.report.append("%s : aucun objet de collision, fichier conservé" % label)
            continue
        tris = np.concatenate(all_tris)
        attrs = np.concatenate(all_attr)
        if attrs.max() > 0xFFFF:
            raise ExportError("Code d'attribut KCL hors limites")
        results[label] = kcl_mod.build(tris, attrs, st.kcl_settings)
        if entries is not None and changed[0]:
            results[attr_name] = byml_mod.to_binary(entries, byml_meta[0], byml_meta[1])
            st.report.append("%s : %d entrée(s) d'attributs" % (attr_name, len(entries)))
    return results


def export_szs(root_col, source_path, out_path, st, depsgraph):
    with open(source_path, "rb") as fh:
        raw = fh.read()
    files = {f.name: f.data for f in sarc_mod.open_container(raw, os.path.basename(source_path), True)}
    replacements = {}
    if st.export_model:
        replacements.update(gather_bfres_geometries(root_col, files, st))
    if st.export_collision:
        replacements.update(gather_kcl(root_col, files, st, depsgraph))
    if not replacements:
        raise ExportError("Rien à exporter : aucun objet issu de ce .szs n'a été trouvé")
    out = sarc_mod.rebuild_container(raw, replacements, "", st.yaz0_level)
    # Contrôle : l'archive produite doit se relire
    check = {f.name: f.data for f in sarc_mod.open_container(out, os.path.basename(out_path), True)}
    for k, v in replacements.items():
        if check.get(k) != v:
            raise ExportError("Vérification de l'archive échouée pour %s" % k)
    with open(out_path, "wb") as fh:
        fh.write(out)
    return sorted(replacements)
