"""Parseur BFRES Switch (FRES + 4 espaces), lecture seule.

L'ordre et la taille des champs sont calqués sur BfresLibrary
(KillzXGaming/BfresLibrary, dossier Switch/), la lib utilisée par Switch Toolbox :
  ResFileParser.cs, ModelParser.cs, ShapeParser.cs, VertexBufferParser.cs,
  MaterialParser(.V10).cs, Shared/Models/Skeleton/*.cs, Shared/Models/Shape/Mesh.cs

Versions gérées : 0.5 (SMO, MK8DX, Splatoon 2...), 0.7/0.8/0.9 et 0.10 sans
buffers/strings externes. Les tableaux de structures sont lus séquentiellement,
exactement comme LoadList<T>() le fait dans la lib C#.
"""

import struct

import numpy as np


class BfresError(Exception):
    pass


# --------------------------------------------------------------------------- #
#  Lecteur binaire
# --------------------------------------------------------------------------- #
class Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0
        self.size = len(data)

    def _u(self, fmt, n):
        v = struct.unpack_from("<" + fmt, self.d, self.p)[0]
        self.p += n
        return v

    def u8(self):
        return self._u("B", 1)

    def u16(self):
        return self._u("H", 2)

    def i16(self):
        return self._u("h", 2)

    def u32(self):
        return self._u("I", 4)

    def i32(self):
        return self._u("i", 4)

    def u64(self):
        return self._u("Q", 8)

    def f32s(self, n):
        v = struct.unpack_from("<%df" % n, self.d, self.p)
        self.p += 4 * n
        return v

    def skip(self, n):
        self.p += n

    def off(self):
        return self.u64()

    def signature(self, magic):
        got = bytes(self.d[self.p:self.p + 4])
        if got != magic:
            raise BfresError(
                "Signature %r attendue à 0x%X, trouvé %r" % (magic, self.p, got)
            )
        self.p += 4

    def header_block(self):  # LoadHeaderBlock : u32 + i64
        self.p += 12

    # -- chaînes : offset absolu -> u16 longueur + texte zéro-terminé
    def string_at(self, off):
        if off == 0 or off + 2 > self.size:
            return ""
        start = off + 2
        end = self.d.find(b"\x00", start)
        if end < 0:
            end = self.size
        return bytes(self.d[start:end]).decode("utf-8", "replace")

    def string(self):
        return self.string_at(self.u64())

    def dict_keys(self, off):
        """ResDict Switch : u32 '_DIC', i32 nb noeuds (hors racine),
        puis (n+1) noeuds de 16 octets : u32 ref, u16 gauche, u16 droite, u64 clé."""
        if off == 0:
            return []
        count = struct.unpack_from("<i", self.d, off + 4)[0]
        keys = []
        for i in range(1, count + 1):
            key_off = struct.unpack_from("<Q", self.d, off + 8 + i * 16 + 8)[0]
            keys.append(self.string_at(key_off))
        return keys

    def dict_values(self):
        """Équivalent de LoadDictValues<T>() : offset des valeurs puis du dict."""
        values = self.off()
        dct = self.off()
        return self.dict_keys(dct), values


# --------------------------------------------------------------------------- #
#  Structures
# --------------------------------------------------------------------------- #
class Bone:
    __slots__ = ("name", "index", "parent", "smooth_index", "rigid_index",
                 "flags", "scale", "rotation", "position")

    @property
    def is_euler(self):
        return (self.flags & 0x7000) == 0x1000  # BoneFlagsRotation.EulerXYZ


class Skeleton:
    def __init__(self):
        self.bones = []
        self.matrix_to_bone = []  # MatrixToBoneList (smooth puis rigid)


class VertexAttrib:
    __slots__ = ("name", "format", "offset", "buffer_index", "pos")


class VertexBuffer:
    def __init__(self):
        self.attributes = {}
        self.vertex_count = 0
        self.skin_count = 0
        self.buffers = []  # [(bytes, stride)]
        # positions dans le fichier (pour l'export)
        self.pos = 0
        self.f_buffer_offset = 0
        self.f_vertex_count = 0
        self.size_array = 0
        self.stride_array = 0
        self.align = 8


class Mesh:
    __slots__ = ("primitive", "index_format", "index_count", "first_vertex", "index_data",
                 "pos", "buffer_size_struct", "submesh_array", "num_submesh", "submeshes")


class Shape:
    def __init__(self):
        self.name = ""
        self.material_index = 0
        self.bone_index = 0
        self.vertex_buffer_index = 0
        self.skin_count = 0
        self.meshes = []
        self.skin_bone_indices = []
        self.pos = 0
        self.f_material_index = 0
        self.bbox_array = 0
        self.radius_array = 0
        self.vertex_buffer_ptr = 0


class Material:
    def __init__(self):
        self.name = ""
        self.textures = []
        self.samplers = []  # noms des samplers (_a0, _n0...) dans le même ordre


class Model:
    def __init__(self):
        self.name = ""
        self.skeleton = Skeleton()
        self.vertex_buffers = []
        self.shapes = []
        self.materials = []
        self.f_total_vertex_count = 0


class ResFile:
    def __init__(self):
        self.name = ""
        self.version = (0, 0, 0)
        self.models = []
        self.external_files = {}
        self.skeletal_anims = []
        # disposition (export)
        self.alignment_log2 = 12
        self.buffer_info_off = 0
        self.buffer_offset = 0
        self.mempool_off = 0
        self.rlt_off = 0
        self.ext_ranges = []  # [(offset données, taille)]


# --------------------------------------------------------------------------- #
#  Formats d'attributs Switch (SwitchAttribFormat, lu en big-endian sur 16 bits)
#  -> (dtype numpy de base, nb composantes, type)
# --------------------------------------------------------------------------- #
_U8, _I8, _U16, _I16, _F16 = np.uint8, np.int8, np.dtype("<u2"), np.dtype("<i2"), np.dtype("<f2")
_U32, _I32, _F32 = np.dtype("<u4"), np.dtype("<i4"), np.dtype("<f4")

ATTRIB_FORMATS = {
    # 8 bits
    0x0102: (_U8, 1, "unorm"), 0x0202: (_I8, 1, "snorm"), 0x0302: (_U8, 1, "int"),
    0x0402: (_I8, 1, "int"), 0x0802: (_U8, 1, "int"), 0x0A02: (_I8, 1, "int"),
    0x0001: (_U8, 1, "4_4"),
    # 16 bits (16 x 1)  -- valeurs telles que dans BfresLibrary
    0x010A: (_U16, 1, "unorm"), 0x020A: (_U16, 1, "int"), 0x030A: (_I16, 1, "snorm"),
    0x040A: (_I16, 1, "int"), 0x050A: (_F16, 1, "float"), 0x0803: (_U16, 1, "int"),
    0x0A03: (_I16, 1, "int"),
    # 16 bits (8 x 2)
    0x0109: (_U8, 2, "unorm"), 0x0209: (_I8, 2, "snorm"), 0x0309: (_U8, 2, "int"),
    0x0409: (_I8, 2, "int"), 0x0804: (_U8, 2, "int"), 0x0A04: (_I8, 2, "int"),
    # 32 bits (16 x 2)
    0x0112: (_U16, 2, "unorm"), 0x0212: (_I16, 2, "snorm"), 0x0312: (_U16, 2, "int"),
    0x0412: (_I16, 2, "int"), 0x0512: (_F16, 2, "float"), 0x0807: (_U16, 2, "int"),
    0x0A07: (_I16, 2, "int"),
    # 32 bits (8 x 4)
    0x010B: (_U8, 4, "unorm"), 0x020B: (_I8, 4, "snorm"), 0x030B: (_U8, 4, "int"),
    0x040B: (_I8, 4, "int"), 0x080B: (_U8, 4, "int"), 0x0A0B: (_I8, 4, "int"),
    # 32 bits (10 x 3 + 2)
    0x000B: (_U32, 1, "1010102_unorm"), 0x090B: (_U32, 1, "1010102_uint"),
    0x020E: (_U32, 1, "1010102_snorm"), 0x099B: (_U32, 1, "1010102_sint"),
    # 64 bits (16 x 4)
    0x0115: (_U16, 4, "unorm"), 0x0215: (_I16, 4, "snorm"), 0x0315: (_U16, 4, "int"),
    0x0415: (_I16, 4, "int"), 0x0515: (_F16, 4, "float"), 0x080E: (_U16, 4, "int"),
    0x0A0E: (_I16, 4, "int"),
    # 32 bits x N
    0x0314: (_U32, 1, "int"), 0x0416: (_I32, 1, "int"), 0x0516: (_F32, 1, "float"),
    0x0317: (_U32, 2, "int"), 0x0417: (_I32, 2, "int"), 0x0517: (_F32, 2, "float"),
    0x0318: (_U32, 3, "int"), 0x0418: (_I32, 3, "int"), 0x0518: (_F32, 3, "float"),
    0x0319: (_U32, 4, "int"), 0x0419: (_I32, 4, "int"), 0x0519: (_F32, 4, "float"),
}


def _resolve_format(code):
    if code in ATTRIB_FORMATS:
        return code
    swapped = ((code & 0xFF) << 8) | (code >> 8)
    if swapped in ATTRIB_FORMATS:
        return swapped
    return None


def decode_attribute(vb, attr):
    """Retourne un tableau float32 (N, 4) ou None si le format est inconnu."""
    code = _resolve_format(attr.format)
    if code is None or attr.buffer_index >= len(vb.buffers):
        return None
    base, comps, kind = ATTRIB_FORMATS[code]
    base = np.dtype(base)
    elem_size = base.itemsize * comps

    data, stride = vb.buffers[attr.buffer_index]
    count = vb.vertex_count
    if count == 0:
        return np.zeros((0, 4), np.float32)
    if stride == 0:
        stride = elem_size
    needed = attr.offset + (count - 1) * stride + elem_size
    if needed > len(data):
        count = max(0, (len(data) - attr.offset - elem_size) // stride + 1)

    raw = np.frombuffer(data, dtype=np.uint8)
    view = np.lib.stride_tricks.as_strided(
        raw[attr.offset:], shape=(count, elem_size), strides=(stride, 1), writeable=False
    )
    vals = np.ascontiguousarray(view).view(base).reshape(count, comps)

    out = np.zeros((count, 4), np.float32)
    if kind == "unorm":
        out[:, :comps] = vals / float(np.iinfo(base).max)
    elif kind == "snorm":
        out[:, :comps] = np.maximum(vals / float(np.iinfo(base).max), -1.0)
    elif kind in ("int", "float"):
        out[:, :comps] = vals
    elif kind == "4_4":
        v = vals[:, 0].astype(np.int32)
        out[:, 0] = (v & 0xF) / 127.0
        out[:, 1] = (v >> 4) / 127.0
    else:  # 10_10_10_2
        v = vals[:, 0].astype(np.int64)
        x, y, z, w = v & 0x3FF, (v >> 10) & 0x3FF, (v >> 20) & 0x3FF, (v >> 30) & 0x3
        if kind in ("1010102_snorm", "1010102_sint"):
            x, y, z = [((c ^ 0x200) - 0x200) for c in (x, y, z)]
        if kind == "1010102_snorm":
            out[:, 0], out[:, 1], out[:, 2], out[:, 3] = x / 511.0, y / 511.0, z / 511.0, w
        elif kind == "1010102_unorm":
            out[:, 0], out[:, 1], out[:, 2], out[:, 3] = x / 1023.0, y / 1023.0, z / 1023.0, w / 3.0
        else:
            out[:, 0], out[:, 1], out[:, 2], out[:, 3] = x, y, z, w
    return out


def mesh_indices(mesh):
    """Indices (int64) + first_vertex, comme Switch Toolbox."""
    dt = {0: np.uint8, 1: np.dtype("<u2"), 2: np.dtype("<u4")}.get(mesh.index_format)
    if dt is None:
        raise BfresError("Format d'index inconnu : %d" % mesh.index_format)
    dt = np.dtype(dt)
    n = min(mesh.index_count, len(mesh.index_data) // dt.itemsize)
    idx = np.frombuffer(mesh.index_data, dtype=dt, count=n).astype(np.int64)
    return idx + mesh.first_vertex


def mesh_triangles(mesh):
    """Retourne un tableau (T, 3) de triangles (gère les strips)."""
    idx = mesh_indices(mesh)
    if mesh.primitive == 0x03:  # Triangles
        t = len(idx) // 3
        return idx[:t * 3].reshape(t, 3)
    if mesh.primitive == 0x04:  # TriangleStrip
        tris = []
        for i in range(len(idx) - 2):
            a, b, c = idx[i], idx[i + 1], idx[i + 2]
            tris.append((a, b, c) if i % 2 == 0 else (b, a, c))
        return np.array(tris, dtype=np.int64).reshape(-1, 3)
    return np.zeros((0, 3), np.int64)


# --------------------------------------------------------------------------- #
#  Parseurs (miroir de BfresLibrary)
# --------------------------------------------------------------------------- #
class _Ctx:
    def __init__(self, r, major, buffer_offset):
        self.r = r
        self.major = major
        self.buffer_offset = buffer_offset


def _read_bone(c):
    r, v = c.r, c.major
    b = Bone()
    b.name = r.string()
    r.dict_values()  # UserData
    if v > 9:
        r.skip(8)
    if v in (8, 9):
        r.skip(16)
    b.index = r.u16()
    b.parent = r.i16()
    b.smooth_index = r.i16()
    b.rigid_index = r.i16()
    r.i16()  # billboard
    r.u16()  # nb user data
    b.flags = r.u32()
    b.scale = r.f32s(3)
    b.rotation = r.f32s(4)
    b.position = r.f32s(3)
    return b


def _read_skeleton(c, off):
    r, v = c.r, c.major
    sk = Skeleton()
    if off == 0:
        return sk
    save = r.p
    r.p = off
    r.signature(b"FSKL")
    if v >= 9:
        r.u32()
    else:
        r.header_block()
    bone_dict = r.off()
    bone_array = r.off()
    matrix_to_bone = r.off()
    r.off()  # inverse model matrices
    r.u64()  # user pointer
    if v == 8:
        r.skip(16)
    if v >= 9:
        r.u64()  # mirror table
    if v < 9:
        r.u32()  # flags
    num_bone = r.u16()
    num_smooth = r.u16()
    num_rigid = r.u16()
    r.skip(6)

    names = r.dict_keys(bone_dict)
    count = len(names) or num_bone
    r.p = bone_array
    for i in range(count):
        bone = _read_bone(c)
        if not bone.name and i < len(names):
            bone.name = names[i]
        sk.bones.append(bone)

    n = num_smooth + num_rigid
    if matrix_to_bone and n:
        sk.matrix_to_bone = list(struct.unpack_from("<%dH" % n, r.d, matrix_to_bone))
    r.p = save
    return sk


def _read_vertex_buffer(c):
    r, v = c.r, c.major
    vb = VertexBuffer()
    vb.pos = r.p
    r.signature(b"FVTX")
    if v >= 9:
        r.u32()
    else:
        r.header_block()
    attr_keys, attr_values = r.dict_values()
    r.off()  # memory pool
    r.off()  # unk
    if v > 2:
        r.off()  # unk2
    size_off = r.off()
    stride_off = r.off()
    r.u64()  # padding
    vb.f_buffer_offset = r.p
    buffer_offset = r.i32()
    num_attrib = r.u8()
    num_buffer = r.u8()
    r.u16()  # index
    vb.f_vertex_count = r.p
    vb.vertex_count = r.u32()
    vb.skin_count = r.u16() & 0xFF
    raw_align = r.u16()  # v10+ : GPUBufferAlignent, sinon padding
    align = raw_align if (v >= 10 and raw_align > 0) else 8  # défaut BfresLibrary = 8
    vb.align = align
    vb.size_array = size_off
    vb.stride_array = stride_off

    save = r.p
    # Attributs (16 octets chacun)
    r.p = attr_values
    for i in range(len(attr_keys) or num_attrib):
        a = VertexAttrib()
        a.pos = r.p
        a.name = r.string()
        a.format = struct.unpack_from(">H", r.d, r.p)[0]  # big-endian, cf. VertexAttrib.cs
        r.skip(4)
        a.offset = r.u16()
        a.buffer_index = r.u16() & 0xFF
        if not a.name and i < len(attr_keys):
            a.name = attr_keys[i]
        vb.attributes[a.name] = a

    strides = [struct.unpack_from("<I", r.d, stride_off + i * 0x10)[0] for i in range(num_buffer)]
    sizes = [struct.unpack_from("<I", r.d, size_off + i * 0x10)[0] for i in range(num_buffer)]

    pos = c.buffer_offset + buffer_offset
    for i in range(num_buffer):
        pos = (pos + align - 1) // align * align  # loader.Align() : absolu
        vb.buffers.append((bytes(r.d[pos:pos + sizes[i]]), strides[i]))
        pos += sizes[i]
    r.p = save
    return vb


def _read_mesh(c):
    r = c.r
    m = Mesh()
    m.pos = r.p
    m.submesh_array = r.off()
    r.off()  # memory pool
    r.off()  # buffer
    buffer_size_off = r.off()
    face_buffer_offset = r.u32()
    m.primitive = r.u32()
    m.index_format = r.u32()
    m.index_count = r.u32()
    m.first_vertex = r.u32()
    m.num_submesh = r.u16()
    r.u16()
    m.buffer_size_struct = buffer_size_off
    m.submeshes = [struct.unpack_from("<II", r.d, m.submesh_array + i * 8)
                   for i in range(m.num_submesh)] if m.submesh_array else []
    size = struct.unpack_from("<I", r.d, buffer_size_off)[0] if buffer_size_off else 0
    start = c.buffer_offset + face_buffer_offset
    m.index_data = bytes(r.d[start:start + size])
    return m


def _read_shape(c, lod_all):
    r, v = c.r, c.major
    s = Shape()
    s.pos = r.p
    r.signature(b"FSHP")
    if v >= 9:
        r.u32()
    else:
        r.header_block()
    s.name = r.string()
    s.vertex_buffer_ptr = r.off()
    mesh_array = r.off()
    skin_bone_list = r.off()
    r.dict_values()  # key shapes
    s.bbox_array = r.off()
    if v > 2:
        s.radius_array = r.off()
        r.u64()  # user pointer
    else:
        r.u64()
        r.skip(4)
    if v < 9:
        r.u32()  # flags
    r.u16()  # index
    s.f_material_index = r.p
    s.material_index = r.u16()
    s.bone_index = r.u16()
    s.vertex_buffer_index = r.u16()
    num_skin_bone = r.u16()
    s.skin_count = r.u8()
    num_mesh = r.u8()
    r.u8()  # keys
    r.u8()  # target attribs
    if v <= 2 or v >= 9:
        r.skip(2)
    else:
        r.skip(6)

    save = r.p
    r.p = mesh_array
    for i in range(num_mesh):
        s.meshes.append(_read_mesh(c))
    if num_skin_bone and skin_bone_list:
        s.skin_bone_indices = list(struct.unpack_from("<%dH" % num_skin_bone, r.d, skin_bone_list))
    r.p = save
    return s


def _read_material(c):
    r, v = c.r, c.major
    m = Material()
    r.signature(b"FMAT")
    if v >= 9:
        r.u32()
    else:
        r.header_block()
    m.name = r.string()

    if v >= 10:
        r.off()  # shader info
        r.off()  # texture array
        tex_names = r.off()
        r.off()  # sampler array
        m.samplers = r.dict_values()[0]
        r.skip(8 * 5)  # render info x3, source param, source param indices
        r.u64()  # réservé
        r.dict_values()  # user data
        r.skip(8 * 4)  # volatile, user ptr, sampler slots, tex slots
        r.u16()
        r.u8()  # nb samplers
        num_tex = r.u8()
        r.skip(2 + 2 + 2 + 2 + 4)
    else:
        r.dict_values()  # render infos
        r.off()  # shader assign
        r.off()  # texture array
        tex_names = r.off()
        r.off()  # sampler array
        m.samplers = r.dict_values()[0]
        r.dict_values()  # shader params
        r.off()  # source param
        r.dict_values()  # user data
        r.skip(8 * 4)  # volatile, user ptr, sampler slots, tex slots
        if v < 9:
            r.u32()  # flags
        r.u16()  # index
        r.u16()  # nb render info
        num_tex = r.u8()
        r.u8()  # nb samplers
        r.skip(2 * 5)  # shader params, volatile, source size, raw size, user data
        if v < 9:
            r.u32()

    if tex_names and num_tex:
        for i in range(num_tex):
            off = struct.unpack_from("<Q", r.d, tex_names + i * 8)[0]
            m.textures.append(r.string_at(off))
    return m


def _read_model(c, lod_all):
    r, v = c.r, c.major
    mdl = Model()
    r.signature(b"FMDL")
    if v >= 9:
        r.u32()
    else:
        r.header_block()
    mdl.name = r.string()
    r.string()  # path
    skeleton_off = r.off()
    vertex_array = r.off()
    shape_keys, shape_values = r.dict_values()
    if v == 9:
        mat_values = r.off()
        pad = r.off()
        mat_dict = r.off() or pad
    else:
        mat_values = r.off()
        mat_dict = r.off()
        if v >= 10:
            r.off()  # shader assign
    r.dict_values()  # user data
    r.u64()  # user pointer
    num_vb = r.u16()
    num_shape = r.u16()
    num_mat = r.u16()
    if v >= 9:
        r.skip(2 + 2 + 2 + 4)
    else:
        r.skip(2)
        mdl.f_total_vertex_count = r.p
        r.skip(4 + 4)
    save = r.p

    mdl.skeleton = _read_skeleton(c, skeleton_off)

    r.p = vertex_array
    for _ in range(num_vb):
        mdl.vertex_buffers.append(_read_vertex_buffer(c))

    r.p = shape_values
    for i in range(len(shape_keys) or num_shape):
        shp = _read_shape(c, lod_all)
        if not shp.name and i < len(shape_keys):
            shp.name = shape_keys[i]
        mdl.shapes.append(shp)

    mat_keys = r.dict_keys(mat_dict)
    r.p = mat_values
    for i in range(len(mat_keys) or num_mat):
        mat = _read_material(c)
        if not mat.name and i < len(mat_keys):
            mat.name = mat_keys[i]
        mdl.materials.append(mat)

    r.p = save
    return mdl


# --------------------------------------------------------------------------- #
#  Animations squelettiques (FSKA) — SkeletalAnim.cs / BoneAnim.cs / AnimCurve.cs
# --------------------------------------------------------------------------- #
# Offsets des cibles dans BoneAnimData
ANIM_TARGETS = {0x04: ("S", 0), 0x08: ("S", 1), 0x0C: ("S", 2),
                0x10: ("T", 0), 0x14: ("T", 1), 0x18: ("T", 2),
                0x20: ("R", 0), 0x24: ("R", 1), 0x28: ("R", 2), 0x2C: ("R", 3)}


class AnimCurve:
    __slots__ = ("frames", "keys", "curve_type", "target", "start", "end", "scale", "offset")

    def evaluate(self, frame):
        f = self.frames
        n = len(f)
        if n == 0:
            return self.offset
        sc = self.scale if self.scale > 0 else 1.0
        k = self.keys
        if frame <= f[0]:
            i = 0
        else:
            i = int(np.searchsorted(f, frame, side="right") - 1)
            i = min(max(i, 0), n - 1)
        start = f[i]
        end = f[i + 1] if i + 1 < n else self.end
        t = 0.0 if end <= start else min(max((frame - start) / (end - start), 0.0), 1.0)
        ct = self.curve_type
        if ct == 0x00:  # Cubic
            c0, c1, c2, c3 = k[i]
            return (c0 * sc + self.offset) + c1 * sc * t + c2 * sc * t * t + c3 * sc * t * t * t
        if ct == 0x10:  # Linear
            return (k[i][0] * sc + self.offset) + k[i][1] * sc * t
        return k[i][0] * sc + self.offset  # Baked / Step


class BoneAnim:
    def __init__(self):
        self.name = ""
        self.flags = 0
        self.base_scale = None
        self.base_rotate = None
        self.base_translate = None
        self.curves = []


class SkeletalAnim:
    def __init__(self):
        self.name = ""
        self.flags = 0
        self.frame_count = 0
        self.bone_anims = []

    @property
    def loop(self):
        return bool(self.flags & 0x4)

    @property
    def euler(self):
        return (self.flags & 0x7000) == 0x1000


def _read_curve(c):
    r = c.r
    cv = AnimCurve()
    frames_off = r.off()
    keys_off = r.off()
    flags = r.u16()
    num_key = r.u16()
    cv.target = r.u32()
    cv.start, cv.end, cv.scale, cv.offset, _delta = r.f32s(5)
    r.skip(4)
    frame_type = flags & 0x3
    key_type = flags & 0xC
    cv.curve_type = flags & 0x70
    per_key = 4 if cv.curve_type == 0x00 else (2 if cv.curve_type == 0x10 else 1)

    if frame_type == 0:
        frames = np.frombuffer(r.d, "<f4", count=num_key, offset=frames_off).astype(np.float64)
    elif frame_type == 1:
        frames = np.frombuffer(r.d, "<i2", count=num_key, offset=frames_off).astype(np.float64) / 32.0
    else:
        frames = np.frombuffer(r.d, np.uint8, count=num_key, offset=frames_off).astype(np.float64)

    cnt = num_key * per_key
    if key_type == 0x0:
        dt = "<u4" if cv.curve_type in (0x40, 0x60) else "<f4"
    elif key_type == 0x4:
        dt = "<i2"
    else:
        dt = np.int8
    keys = np.frombuffer(r.d, dt, count=cnt, offset=keys_off).astype(np.float64).reshape(num_key, per_key)
    cv.frames = frames
    cv.keys = keys
    return cv


def _read_bone_anim(c):
    r, v = c.r, c.major
    ba = BoneAnim()
    ba.name = r.string()
    curve_off = r.off()
    base_off = r.off()
    if v >= 9:
        r.skip(16)
    ba.flags = r.u32()
    r.u8()  # begin rotate
    r.u8()  # begin translate
    num_curve = r.u8()
    r.u8()  # begin base translate
    r.i32()  # begin curve
    r.i32()

    save = r.p
    if base_off:
        p = base_off
        if ba.flags & (1 << 3):
            ba.base_scale = struct.unpack_from("<3f", r.d, p)
            p += 12
        if ba.flags & (1 << 4):
            ba.base_rotate = struct.unpack_from("<4f", r.d, p)
            p += 16
        if ba.flags & (1 << 5):
            ba.base_translate = struct.unpack_from("<3f", r.d, p)
    r.p = curve_off
    for _ in range(num_curve):
        ba.curves.append(_read_curve(c))
    r.p = save
    return ba


def _read_skeletal_anim(c):
    r, v = c.r, c.major
    a = SkeletalAnim()
    r.signature(b"FSKA")
    if v >= 9:
        a.flags = r.u32()
    else:
        r.header_block()
    a.name = r.string()
    r.string()  # path
    r.off()  # bind skeleton
    r.off()  # bind indices
    bone_array = r.off()
    r.dict_values()  # user data
    if v < 9:
        a.flags = r.u32()
    a.frame_count = r.i32()
    r.i32()  # nb courbes
    r.u32()  # baked size
    num_bone_anim = r.u16()
    r.u16()
    if v < 9:
        r.u32()
    save = r.p
    r.p = bone_array
    for _ in range(num_bone_anim):
        a.bone_anims.append(_read_bone_anim(c))
    r.p = save
    return a


def parse(data, lod_all=False):
    """Analyse un BFRES Switch et retourne un ResFile."""
    if bytes(data[:4]) != b"FRES":
        raise BfresError("Ce n'est pas un fichier BFRES (signature FRES absente)")
    if bytes(data[4:8]) != b"    ":
        raise BfresError(
            "BFRES Wii U détecté (pas d'espaces après 'FRES') : seul le format Switch est géré"
        )

    r = Reader(data)
    res = ResFile()
    r.p = 8
    version = r.u32()
    major = (version >> 16) & 0xFFFF
    res.version = (major, (version >> 8) & 0xFF, version & 0xFF)
    bom = bytes(data[0x0C:0x0E])
    if bom != b"\xFF\xFE":
        raise BfresError("BFRES Switch big-endian inattendu (BOM %r)" % bom)

    if major >= 10:
        flags = data[0xEE]
        if flags & 0x01:
            raise BfresError("BFRES v10 avec table de chaînes externe (TotK) non géré")
        if flags & 0x02:
            raise BfresError("BFRES v10 avec buffers GPU externes (TotK) non géré")

    res.alignment_log2 = data[0x0E]
    res.rlt_off = struct.unpack_from("<I", data, 0x18)[0]
    r.p = 0x20
    res.name = r.string()
    model_values = r.off()
    model_dict = r.off()
    if major >= 9:
        r.skip(32)
    ska_keys, ska_values = r.dict_values()
    for _ in range(4):  # anims matériau, visibilité os, shape, scène
        r.dict_values()
    res.mempool_off = r.off()
    buffer_info = r.off()
    res.buffer_info_off = buffer_info
    ext_keys, ext_values = r.dict_values()

    buffer_offset = 0
    if buffer_info:
        buffer_offset = struct.unpack_from("<q", data, buffer_info + 8)[0]
    res.buffer_offset = buffer_offset

    for i, key in enumerate(ext_keys):
        doff, dsize = struct.unpack_from("<QQ", data, ext_values + i * 16)
        res.external_files[key] = bytes(data[doff:doff + dsize])
        res.ext_ranges.append((doff, dsize))

    c = _Ctx(r, major, buffer_offset)
    model_keys = r.dict_keys(model_dict)
    r.p = model_values
    for i in range(len(model_keys)):
        mdl = _read_model(c, lod_all)
        if not mdl.name:
            mdl.name = model_keys[i]
        res.models.append(mdl)

    if ska_keys:
        save = r.p
        r.p = ska_values
        for i in range(len(ska_keys)):
            try:
                anim = _read_skeletal_anim(c)
            except (BfresError, struct.error, IndexError):
                break
            if not anim.name:
                anim.name = ska_keys[i]
            res.skeletal_anims.append(anim)
        r.p = save
    return res
