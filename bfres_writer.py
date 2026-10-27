"""Reconstruction d'un BFRES Switch avec de nouvelles géométries.

Principe : la section principale (structures + table de chaînes) est conservée,
ses champs de géométrie sont patchés, puis la fin du fichier est reconstruite
dans l'ordre de BfresLibrary (ResFileSaver) :

    [section 1 : structures + _STR] [index buffers] [vertex buffers]
    [memory pool 0x120] [fichiers externes (BNTX...)] [_RLT]

Tous les pointeurs du fichier sont listés dans la table de relocation (_RLT).
Elle sert à repositionner ceux qui visent la fin du fichier (memory pool,
fichiers externes, buffer info) : le résultat reste un fichier relogeable valide.
"""

import struct

import numpy as np

from . import bfres as B


class BfresWriteError(Exception):
    pass


class ShapeGeometry:
    """Nouvelle géométrie d'un shape, dans l'espace de stockage du jeu."""

    def __init__(self):
        self.attrs = {}        # nom d'attribut -> (N, 4) float
        self.triangles = None  # (T, 3) int
        self.vertex_count = 0
        self.bbox_center = (0.0, 0.0, 0.0)
        self.bbox_extent = (0.0, 0.0, 0.0)
        self.radius = 0.0
        self.material_index = None
        self.hidden = False


def _align(x, a):
    return (x + a - 1) // a * a


# --------------------------------------------------------------------------- #
#  Encodage des attributs (inverse de bfres.decode_attribute)
# --------------------------------------------------------------------------- #
def encode_attribute(values, code):
    rc = B._resolve_format(code)
    if rc is None:
        raise BfresWriteError("Format d'attribut inconnu 0x%04X" % code)
    base, comps, kind = B.ATTRIB_FORMATS[rc]
    base = np.dtype(base)
    v = np.asarray(values, np.float64)
    n = len(v)

    if kind == "unorm":
        mx = np.iinfo(base).max
        out = np.round(np.clip(v[:, :comps], 0, 1) * mx).astype(base)
    elif kind == "snorm":
        mx = np.iinfo(base).max
        out = np.round(np.clip(v[:, :comps], -1, 1) * mx).astype(base)
    elif kind == "int":
        info = np.iinfo(base)
        out = np.clip(np.round(v[:, :comps]), info.min, info.max).astype(base)
    elif kind == "float":
        out = v[:, :comps].astype(base)
    elif kind == "4_4":
        a = np.clip(np.round(v[:, 0] * 127), 0, 15).astype(np.uint8)
        b = np.clip(np.round(v[:, 1] * 127), 0, 15).astype(np.uint8)
        out = (a | (b << 4)).reshape(n, 1)
    else:
        if kind == "1010102_snorm":
            xyz = np.round(np.clip(v[:, :3], -1, 1) * 511).astype(np.int64) & 0x3FF
            w = np.clip(np.round(v[:, 3]), 0, 3).astype(np.int64)
        elif kind == "1010102_unorm":
            xyz = np.round(np.clip(v[:, :3], 0, 1) * 1023).astype(np.int64)
            w = np.round(np.clip(v[:, 3], 0, 1) * 3).astype(np.int64)
        elif kind == "1010102_sint":
            xyz = np.clip(np.round(v[:, :3]), -512, 511).astype(np.int64) & 0x3FF
            w = np.clip(np.round(v[:, 3]), 0, 3).astype(np.int64)
        else:
            xyz = np.clip(np.round(v[:, :3]), 0, 1023).astype(np.int64)
            w = np.clip(np.round(v[:, 3]), 0, 3).astype(np.int64)
        packed = xyz[:, 0] | (xyz[:, 1] << 10) | (xyz[:, 2] << 20) | (w << 30)
        out = packed.astype(np.uint32).reshape(n, 1)
    return np.ascontiguousarray(out).view(np.uint8).reshape(n, -1)


def _default_values(name, n):
    v = np.zeros((n, 4), np.float64)
    if name.startswith("_c"):
        v[:] = 1.0
    elif name.startswith("_w"):
        v[:, 0] = 1.0
    elif name == "_t0":
        v[:, 0] = 1.0
        v[:, 3] = 1.0
    elif name == "_b0":
        v[:, 1] = 1.0
    return v


def build_vertex_buffers(vb, geo):
    """Retourne la liste des nouveaux buffers (bytes) d'un vertex buffer."""
    n = geo.vertex_count
    bufs = [np.zeros((n, max(stride, 1)), np.uint8) for _, stride in vb.buffers]
    for name, a in vb.attributes.items():
        if a.buffer_index >= len(bufs):
            continue
        vals = geo.attrs.get(name)
        if vals is None or len(vals) != n:
            vals = _default_values(name, n)
        try:
            enc = encode_attribute(vals, a.format)
        except BfresWriteError:
            continue
        size = enc.shape[1]
        dest = bufs[a.buffer_index]
        if a.offset + size <= dest.shape[1]:
            dest[:, a.offset:a.offset + size] = enc
    return [b.tobytes() for b in bufs]


def build_indices(tris, first_vertex=0):
    flat = np.asarray(tris, np.int64).reshape(-1) - first_vertex
    if len(flat) and flat.max() > 0xFFFF:
        return 2, flat.astype("<u4").tobytes(), len(flat)
    return 1, flat.astype("<u2").tobytes(), len(flat)


# --------------------------------------------------------------------------- #
#  Table de relocation
# --------------------------------------------------------------------------- #
class _Rlt:
    def __init__(self, data, off):
        if bytes(data[off:off + 4]) != b"_RLT":
            raise BfresWriteError("Table de relocation _RLT introuvable à 0x%X" % off)
        self.off = off
        count = struct.unpack_from("<I", data, off + 8)[0]
        self.sections = []
        for i in range(count):
            p = off + 0x10 + i * 0x18
            _ptr, pos, size, eidx, ecount = struct.unpack_from("<QIIII", data, p)
            self.sections.append([pos, size, eidx, ecount])
        ent_start = off + 0x10 + count * 0x18
        total = sum(s[3] for s in self.sections)
        self.entries = [struct.unpack_from("<IHBB", data, ent_start + i * 8) for i in range(total)]
        self.raw = bytes(data[off:ent_start + total * 8])

    def pointer_fields(self):
        for pos, structs, offsets, padding in self.entries:
            stride = (offsets + padding) * 8
            for s in range(structs):
                for k in range(offsets):
                    yield pos + s * stride + k * 8

    def serialize(self):
        out = bytearray(self.raw)
        for i, (pos, size, eidx, ecount) in enumerate(self.sections):
            struct.pack_into("<QIIII", out, 0x10 + i * 0x18, 0, pos, size, eidx, ecount)
        return out


# --------------------------------------------------------------------------- #
#  Reconstruction
# --------------------------------------------------------------------------- #
def rebuild(data, res, geometries):
    """data : octets d'origine ; res : bfres.parse(data) ;
    geometries : {(index modèle, index shape): ShapeGeometry}."""
    data = bytes(data)
    if not res.rlt_off:
        raise BfresWriteError("BFRES sans table de relocation")
    rlt = _Rlt(data, res.rlt_off)
    sec1_end = rlt.sections[0][1] if rlt.sections else res.buffer_offset
    data_align = 1 << res.alignment_log2 if res.alignment_log2 < 16 else 0x1000

    out = bytearray(data[:sec1_end])

    def put(fmt, pos, *vals):
        struct.pack_into("<" + fmt, out, pos, *vals)

    # --- Décisions par mesh / vertex buffer
    vb_new = {}      # (mi, vbi) -> (liste de buffers, nb sommets)
    mesh_new = {}    # id(mesh) -> (format, octets, nb index)
    for (mi, si), geo in geometries.items():
        mdl = res.models[mi]
        shp = mdl.shapes[si]
        vbi = shp.vertex_buffer_index
        if geo.hidden:
            for m in shp.meshes:
                mesh_new[id(m)] = (1, b"\x00" * 6, 3)
            continue
        users = [s for s in mdl.shapes if s.vertex_buffer_index == vbi]
        if len(users) > 1:
            raise BfresWriteError("Le shape %s partage son vertex buffer : non modifiable" % shp.name)
        vb = mdl.vertex_buffers[vbi]
        vb_new[(mi, vbi)] = (build_vertex_buffers(vb, geo), geo.vertex_count)
        fmt, ib, cnt = build_indices(geo.triangles)
        for m in shp.meshes:
            mesh_new[id(m)] = (fmt, ib, cnt)

    # --- Index buffers
    buf_start = _align(len(out), data_align)
    out += b"\x00" * (buf_start - len(out))
    shared = {}
    for mdl in res.models:
        for shp in mdl.shapes:
            for m in shp.meshes:
                if id(m) in mesh_new:
                    fmt, ib, cnt = mesh_new[id(m)]
                    key = (id(ib),)
                    if key not in shared:
                        while len(out) % 8:
                            out.append(0)
                        shared[key] = len(out) - buf_start
                        out += ib
                    off = shared[key]
                    put("I", m.pos + 0x20, off)
                    put("I", m.pos + 0x24, 0x03)  # triangles
                    put("I", m.pos + 0x28, fmt)
                    put("I", m.pos + 0x2C, cnt)
                    put("I", m.pos + 0x30, 0)
                    if m.buffer_size_struct:
                        put("I", m.buffer_size_struct, len(ib))
                    if m.submesh_array and m.num_submesh:
                        for k in range(m.num_submesh):
                            put("II", m.submesh_array + k * 8, 0, cnt if k == 0 else 0)
                else:
                    while len(out) % 8:
                        out.append(0)
                    put("I", m.pos + 0x20, len(out) - buf_start)
                    out += m.index_data
    index_end = len(out)

    # --- Vertex buffers
    first_align = res.models[0].vertex_buffers[0].align if res.models and res.models[0].vertex_buffers else 8
    while len(out) % first_align:
        out.append(0)
    vtx_start = len(out)
    for mi, mdl in enumerate(res.models):
        delta_vertices = 0
        for vbi, vb in enumerate(mdl.vertex_buffers):
            new = vb_new.get((mi, vbi))
            buffers = new[0] if new else [b for b, _ in vb.buffers]
            count = new[1] if new else vb.vertex_count
            delta_vertices += count - vb.vertex_count
            while len(out) % vb.align:
                out.append(0)
            put("i", vb.f_buffer_offset, len(out) - buf_start)
            put("I", vb.f_vertex_count, count)
            for k, blob in enumerate(buffers):
                while len(out) % vb.align:
                    out.append(0)
                out += blob
                if vb.size_array:
                    put("I", vb.size_array + k * 0x10, len(blob))
        if mdl.f_total_vertex_count and delta_vertices:
            old_total = struct.unpack_from("<I", data, mdl.f_total_vertex_count)[0]
            put("I", mdl.f_total_vertex_count, max(0, old_total + delta_vertices))
    vtx_end = len(out)

    # --- Memory pool
    new_mempool = 0
    if res.mempool_off:
        new_mempool = _align(len(out), data_align)
        out += b"\x00" * (new_mempool - len(out))
        out += data[res.mempool_off:res.mempool_off + 0x120]
        if res.buffer_info_off:
            put("I", res.buffer_info_off + 4, new_mempool - buf_start)
    elif res.buffer_info_off:
        put("I", res.buffer_info_off + 4, vtx_end - buf_start)

    # --- Fichiers externes (copiés en bloc, alignement interne conservé)
    ext_start = min((o for o, _ in res.ext_ranges), default=0)
    new_ext = 0
    if res.ext_ranges:
        ext_end = res.rlt_off if res.rlt_off > ext_start else max(o + s for o, s in res.ext_ranges)
        ext_al = min(ext_start & -ext_start, 0x1000) if ext_start else 0x1000
        new_ext = _align(len(out), max(ext_al, 0x100))
        out += b"\x00" * (new_ext - len(out))
        out += data[ext_start:ext_end]

    # --- Pointeurs vers la fin du fichier
    regions = []
    if res.buffer_offset:
        regions.append((res.buffer_offset, buf_start))
    if res.mempool_off:
        regions.append((res.mempool_off, new_mempool))
    if res.ext_ranges:
        regions.append((ext_start, new_ext))
    regions.append((res.rlt_off, None))  # remplacé plus bas
    regions.sort()

    new_rlt = _align(len(out), 0x100)

    def remap(v):
        if v < sec1_end:
            return v
        for i, (start, new_start) in enumerate(regions):
            end = regions[i + 1][0] if i + 1 < len(regions) else len(data) + 1
            if start <= v < end:
                ns = new_rlt if new_start is None else new_start
                return ns + (v - start)
        return v

    for p in rlt.pointer_fields():
        if p + 8 <= len(out):
            v = struct.unpack_from("<Q", out, p)[0]
            if v:
                struct.pack_into("<Q", out, p, remap(v))

    # --- Sections de la table de relocation
    secs = rlt.sections
    if len(secs) >= 5:
        # comme BfresLibrary : chaque section court jusqu'au début de la suivante
        secs[1][0], secs[1][1] = buf_start, vtx_start - buf_start
        vtx_stop = new_mempool if res.mempool_off else vtx_end
        secs[2][0], secs[2][1] = vtx_start, vtx_stop - vtx_start
        if res.mempool_off:
            secs[3][0], secs[3][1] = new_mempool, 0x120
        else:
            secs[3][0] = remap(secs[3][0])
        if res.ext_ranges:
            secs[4][0], secs[4][1] = new_ext, new_rlt - new_ext
        else:
            secs[4][0], secs[4][1] = new_rlt, 0
    else:
        for s in secs[1:]:
            s[0] = remap(s[0])

    out += b"\x00" * (new_rlt - len(out))
    rlt_bytes = rlt.serialize()
    struct.pack_into("<I", rlt_bytes, 4, new_rlt)
    out += rlt_bytes

    # --- En-têtes
    put("I", 0x18, new_rlt)
    put("I", 0x1C, len(out))
    str_pos = out.rfind(b"_STR", 0, sec1_end)
    if str_pos >= 0:
        old = struct.unpack_from("<I", data, str_pos + 8)[0]
        if old == res.rlt_off - str_pos:
            put("I", str_pos + 8, new_rlt - str_pos)

    # --- Boîtes englobantes, rayons, matériau
    for (mi, si), geo in geometries.items():
        shp = res.models[mi].shapes[si]
        if geo.material_index is not None:
            put("H", shp.f_material_index, int(geo.material_index))
        if geo.hidden:
            continue
        nb = sum(m.num_submesh + 1 for m in shp.meshes)
        if shp.bbox_array and nb:
            for k in range(nb):
                put("6f", shp.bbox_array + k * 24, *geo.bbox_center, *geo.bbox_extent)
        if shp.radius_array and shp.meshes:
            if res.version[0] >= 10:
                nr = len(shp.skin_bone_indices) or len(shp.meshes)
                for k in range(nr):
                    put("4f", shp.radius_array + k * 16, *geo.bbox_center, geo.radius)
            else:
                for k in range(len(shp.meshes)):
                    put("f", shp.radius_array + k * 4, geo.radius)
    return bytes(out)


def validate(data):
    """Vérifications structurelles d'un BFRES reconstruit. Retourne une liste d'erreurs."""
    errors = []
    try:
        res = B.parse(data)
    except Exception as e:  # noqa: BLE001
        return ["parse: %s" % e]
    size = struct.unpack_from("<I", data, 0x1C)[0]
    if size != len(data):
        errors.append("taille d'en-tête %d != %d" % (size, len(data)))
    try:
        rlt = _Rlt(data, res.rlt_off)
    except BfresWriteError as e:
        return errors + [str(e)]
    for p in rlt.pointer_fields():
        v = struct.unpack_from("<Q", data, p)[0]
        if v >= len(data):
            errors.append("pointeur hors fichier à 0x%X -> 0x%X" % (p, v))
    for m in res.models:
        for s in m.shapes:
            if s.vertex_buffer_index >= len(m.vertex_buffers):
                errors.append("shape %s : vertex buffer invalide" % s.name)
                continue
            vb = m.vertex_buffers[s.vertex_buffer_index]
            for mesh in s.meshes:
                idx = B.mesh_indices(mesh)
                if len(idx) and idx.max() >= vb.vertex_count:
                    errors.append("shape %s : index hors limites" % s.name)
            for a in vb.attributes.values():
                if B.decode_attribute(vb, a) is None:
                    continue
    return errors
