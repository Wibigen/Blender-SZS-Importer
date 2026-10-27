"""Génère des fichiers de test synthétiques (BFRES Switch v0.5.0.3 complet avec
_RLT, memory pool, BNTX embarqué, animation FSKA ; KCL v2 ; Attribute.byml),
emballés en SARC + Yaz0 via oead (implémentation indépendante).

Usage : python make_test_files.py <dossier> [chemin vers BNTX-Extractor]
"""

import math
import os
import struct
import sys

import numpy as np
import oead

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
if len(sys.argv) > 2:
    sys.path.append(sys.argv[2])
import swizzle as ab_swizzle  # noqa: E402  (BNTX-Extractor, AboodXD)


class Blob:
    def __init__(self):
        self.b = bytearray()
        self.strings = {}
        self.ptrs = []  # (position, section)

    def alloc(self, size, align=8):
        while len(self.b) % align:
            self.b.append(0)
        off = len(self.b)
        self.b.extend(b"\x00" * size)
        return off

    def put(self, off, fmt, *vals):
        struct.pack_into("<" + fmt, self.b, off, *vals)

    def ptr(self, off, value, section=0):
        """Écrit un pointeur 64 bits et l'enregistre pour la _RLT."""
        struct.pack_into("<Q", self.b, off, value)
        if value:
            self.ptrs.append((off, section))

    def raw(self, data, align=8):
        off = self.alloc(len(data), align)
        self.b[off:off + len(data)] = data
        return off

    def string(self, s):
        if s in self.strings:
            return self.strings[s]
        enc = s.encode()
        off = self.raw(struct.pack("<H", len(enc)) + enc + b"\x00", 2)
        self.strings[s] = off
        return off

    def strptr(self, off, s):
        self.ptr(off, self.string(s))

    def dict(self, keys):
        n = len(keys)
        off = self.alloc(8 + (n + 1) * 16)
        self.b[off:off + 4] = b"_DIC"
        self.put(off + 4, "i", n)
        self.put(off + 8, "IHH", 0xFFFFFFFF, 1, 0)
        self.strptr(off + 16, "")
        for i, k in enumerate(keys):
            self.put(off + 8 + (i + 1) * 16, "IHH", 0, 0, 0)
            self.strptr(off + 8 + (i + 1) * 16 + 8, k)
        return off


# --------------------------------------------------------------------------- #
def build_bntx():
    """BNTX avec 2 textures : BC1 64x64 (block linear) et RGBA8 32x32."""
    rng = np.random.default_rng(7)
    tex = []
    # Texture 1 : BC1 aléatoire
    bc1 = rng.integers(0, 256, (64 // 4) * (64 // 4) * 8, dtype=np.uint8).tobytes()
    sw1 = bytes(ab_swizzle.swizzle(64, 64, 4, 4, 8, 1, 512, 1, bc1))
    tex.append(("Body_alb", 0x1a06, 64, 64, 1, sw1, bc1))
    # Texture 2 : RGBA8 dégradé
    g = np.zeros((32, 32, 4), np.uint8)
    g[..., 0] = np.arange(32)[None, :] * 8
    g[..., 1] = np.arange(32)[:, None] * 8
    g[..., 3] = 255
    lin = g.tobytes()
    sw2 = bytes(ab_swizzle.swizzle(32, 32, 1, 1, 4, 1, 512, 2, lin))
    tex.append(("Body_nrm", 0x0b01, 32, 32, 2, sw2, lin))

    out = bytearray(0x20)
    out[0:8] = b"BNTX\x00\x00\x00\x00"
    struct.pack_into("<I", out, 8, 0x00040000)
    out[0x0C:0x0E] = b"\xFF\xFE"
    out += b"NX  " + struct.pack("<I", len(tex)) + b"\x00" * 28  # rempli plus bas
    info_ptrs = len(out)
    out += b"\x00" * (8 * len(tex))
    brti = []
    for name, fmt, w, h, sr, data, _ in tex:
        while len(out) % 8:
            out.append(0)
        p = len(out)
        brti.append(p)
        out += b"\x00" * 0xA0
        out[p:p + 4] = b"BRTI"
        struct.pack_into("<BBHHH", out, p + 0x10, 1, 2, 0, 0, 1)
        struct.pack_into("<I", out, p + 0x1C, fmt)
        struct.pack_into("<iiii", out, p + 0x24, w, h, 1, 1)
        struct.pack_into("<i", out, p + 0x34, sr)
        struct.pack_into("<iiIi", out, p + 0x50, len(data), 512, 0x05040302, 1)
    names = []
    for name, *_ in tex:
        while len(out) % 2:
            out.append(0)
        names.append(len(out))
        out += struct.pack("<H", len(name)) + name.encode() + b"\x00"
    ptr_arrays = []
    for _ in tex:
        while len(out) % 8:
            out.append(0)
        ptr_arrays.append(len(out))
        out += b"\x00" * 8
    datas = []
    for t in tex:
        while len(out) % 0x200:
            out.append(0)
        datas.append(len(out))
        out += t[5]
    for i, p in enumerate(brti):
        struct.pack_into("<qqq", out, p + 0x60, names[i], 0, ptr_arrays[i])
        struct.pack_into("<q", out, ptr_arrays[i], datas[i])
        struct.pack_into("<q", out, info_ptrs + i * 8, p)
    struct.pack_into("<q", out, 0x28, info_ptrs)
    struct.pack_into("<I", out, 0x1C, len(out))
    return bytes(out), tex


def build_bfres(bntx):
    B = Blob()
    hdr = B.alloc(0xC0)

    # ----- données GPU
    pos_a = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]], np.float32)
    sn = lambda x: int(round(x * 511)) & 0x3FF
    n_packed = sn(0) | (sn(0) << 10) | (sn(1) << 20)
    uv_a = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float16)
    idx_a = np.array([0, 1, 2, 0, 2, 3], np.uint16)
    buf0 = pos_a.tobytes()
    buf1 = bytearray()
    for i in range(4):
        buf1 += struct.pack("<I", n_packed) + uv_a[i].tobytes() + bytes([0, 1, 0, 0]) + bytes([128, 127, 0, 0])
    buf1 += bytes([255, 255, 255, 255]) * 0  # (couleurs absentes)
    stride1 = 16
    pos_b = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], np.float32)
    idx_b = np.array([0, 1, 2], np.uint16)

    # ----- squelette
    bones = [("Root", -1, (0, 0, 0, 1), (0, 0, 0), 0x0000, 0, -1),
             ("Arm", 0, (0, 0, math.pi / 2, 0), (0, 50, 0), 0x1000, 1, 0)]
    skel = B.alloc(0x48)
    bone_arr = B.alloc(0x50 * len(bones))
    for i, (name, parent, rot, pos, flags, smooth, rigid) in enumerate(bones):
        o = bone_arr + i * 0x50
        B.strptr(o, name)
        B.put(o + 0x18, "HhhhhHI", i, parent, smooth, rigid, -1, 0, flags)
        B.put(o + 0x28, "3f", 1, 1, 1)
        B.put(o + 0x34, "4f", *rot)
        B.put(o + 0x44, "3f", *pos)
    mtb = B.raw(struct.pack("<3H", 0, 1, 1))
    B.b[skel:skel + 4] = b"FSKL"
    B.ptr(skel + 0x10, B.dict([b[0] for b in bones]))
    B.ptr(skel + 0x18, bone_arr)
    B.ptr(skel + 0x20, mtb)
    B.put(skel + 0x38, "IHHH", 0x1100, len(bones), 2, 1)

    vb_fields = []

    def vertex_buffer(o, attribs, buffers, count, skin):
        av = B.alloc(16 * len(attribs))
        for i, (name, fmt, off, bi) in enumerate(attribs):
            a = av + i * 16
            B.strptr(a, name)
            struct.pack_into(">H", B.b, a + 8, fmt)
            B.put(a + 12, "HH", off, bi)
        sizes = B.alloc(16 * len(buffers))
        strides = B.alloc(16 * len(buffers))
        for i, (data, stride) in enumerate(buffers):
            B.put(sizes + i * 16, "II", len(data), 0)
            B.put(strides + i * 16, "I", stride)
        B.b[o:o + 4] = b"FVTX"
        B.ptr(o + 0x10, av)
        B.ptr(o + 0x18, B.dict([a[0] for a in attribs]))
        vb_fields.append((o + 0x20, buffers))  # memory pool + offset GPU à remplir
        B.ptr(o + 0x38, sizes)
        B.ptr(o + 0x40, strides)
        B.put(o + 0x50, "iBBHIHH", 0, len(attribs), len(buffers), 0, count, skin, 0)

    vtx_arr = B.alloc(0x60 * 2)
    vertex_buffer(vtx_arr, [("_p0", 0x0518, 0, 0), ("_n0", 0x020E, 0, 1), ("_u0", 0x0512, 4, 1),
                            ("_i0", 0x030B, 8, 1), ("_w0", 0x010B, 12, 1)],
                  [(buf0, 12), (bytes(buf1), stride1)], 4, 2)
    vertex_buffer(vtx_arr + 0x60, [("_p0", 0x0518, 0, 0)], [(pos_b.tobytes(), 12)], 3, 0)

    mesh_fields = []

    def mesh(o, idx):
        bsz = B.alloc(16)
        B.put(bsz, "II", len(idx.tobytes()), 0)
        sub = B.alloc(8)
        B.put(sub, "II", 0, len(idx))
        B.ptr(o + 0x00, sub)
        mesh_fields.append((o, idx.tobytes()))
        B.ptr(o + 0x18, bsz)
        B.put(o + 0x24, "IIIIHH", 3, 1, len(idx), 0, 1, 0)

    mesh_a = B.alloc(0x38)
    mesh(mesh_a, idx_a)
    mesh_b = B.alloc(0x38)
    mesh(mesh_b, idx_b)

    shp_arr = B.alloc(0x70 * 2)
    for i, (name, m, mat, bone, vbi, skin) in enumerate(
            [("Body", mesh_a, 0, 0, 0, 2), ("Hand", mesh_b, 1, 1, 1, 0)]):
        o = shp_arr + i * 0x70
        B.b[o:o + 4] = b"FSHP"
        B.strptr(o + 0x10, name)
        B.ptr(o + 0x18, vtx_arr + vbi * 0x60)
        B.ptr(o + 0x20, m)
        bb = B.alloc(24 * 2)
        B.ptr(o + 0x40, bb)
        rad = B.alloc(4)
        B.ptr(o + 0x48, rad)
        B.put(o + 0x58, "IHHHHHBBBB", 0, i, mat, bone, vbi, 0, skin, 1, 0, 0)

    mat_arr = B.alloc(0xB8 * 2)
    for i, (name, texs) in enumerate([("Mat_Body", ["Body_alb", "Body_nrm"]), ("Mat_Hand", [])]):
        o = mat_arr + i * 0xB8
        B.b[o:o + 4] = b"FMAT"
        B.strptr(o + 0x10, name)
        if texs:
            tn = B.alloc(8 * len(texs))
            for k, t in enumerate(texs):
                B.strptr(tn + k * 8, t)
            B.ptr(o + 0x38, tn)
            B.ptr(o + 0x48, B.alloc(16 * len(texs)))  # valeurs des samplers
            B.ptr(o + 0x50, B.dict(["_a0", "_n0"]))
        B.put(o + 0xA4, "HHBB", i, 0, len(texs), len(texs))

    mdl = B.alloc(0x78)
    B.b[mdl:mdl + 4] = b"FMDL"
    B.strptr(mdl + 0x10, "TestModel")
    B.strptr(mdl + 0x18, "")
    B.ptr(mdl + 0x20, skel)
    B.ptr(mdl + 0x28, vtx_arr)
    B.ptr(mdl + 0x30, shp_arr)
    B.ptr(mdl + 0x38, B.dict(["Body", "Hand"]))
    B.ptr(mdl + 0x40, mat_arr)
    B.ptr(mdl + 0x48, B.dict(["Mat_Body", "Mat_Hand"]))
    B.put(mdl + 0x68, "HHHHI", 2, 2, 2, 0, 7)

    # ----- animation FSKA : "Wave" (Arm : rotation Z cubique, translation X linéaire)
    ska = B.alloc(0x60)
    B.b[ska:ska + 4] = b"FSKA"
    B.strptr(ska + 0x10, "Wave")
    B.strptr(ska + 0x18, "")
    bone_anims = B.alloc(40)
    B.ptr(ska + 0x30, bone_anims)
    B.put(ska + 0x48, "IiiIHH", 0x1000 | 0x4, 30, 2, 0, 1, 0)
    B.strptr(bone_anims, "Arm")
    curves = B.alloc(0x30 * 2)
    base = B.alloc(16)
    B.put(base, "4f", 0, 0, math.pi / 2, 0)
    B.ptr(bone_anims + 8, curves)
    B.ptr(bone_anims + 16, base)
    B.put(bone_anims + 24, "IBBBBii", (1 << 4) | (1 << 11) | (1 << 13), 0, 0, 2, 0, 0, 0)
    # courbe 1 : RZ cubique, 2 clés (0 et 30) : pi/2 -> pi/2 + 1
    fr = B.raw(struct.pack("<2f", 0, 30))
    ky = B.raw(struct.pack("<8f", math.pi / 2, 1.0, 0.0, 0.0, math.pi / 2 + 1, 0, 0, 0))
    B.ptr(curves, fr)
    B.ptr(curves + 8, ky)
    B.put(curves + 16, "HHIfffff", 0x0000, 2, 0x28, 0, 30, 1.0, 0.0, 0)
    # courbe 2 : TX linéaire : 0 -> 20
    fr2 = B.raw(struct.pack("<2f", 0, 30))
    ky2 = B.raw(struct.pack("<4f", 0.0, 20.0, 20.0, 0.0))
    B.ptr(curves + 0x30, fr2)
    B.ptr(curves + 0x38, ky2)
    B.put(curves + 0x30 + 16, "HHIfffff", 0x0010, 2, 0x10, 0, 30, 1.0, 0.0, 0)

    # ----- fichiers externes + en-tête
    ext_vals = B.alloc(16)
    ext_dict = B.dict(["textures.bntx"])
    binfo = B.alloc(0x20)

    B.b[hdr:hdr + 8] = b"FRES    "
    B.put(hdr + 8, "I", 0x00050003)
    B.b[hdr + 0x0C:hdr + 0x0E] = b"\xFF\xFE"
    B.put(hdr + 0x0E, "BB", 0x0C, 0)
    B.strptr(hdr + 0x20, "TestModel")
    B.ptr(hdr + 0x28, mdl)
    B.ptr(hdr + 0x30, B.dict(["TestModel"]))
    B.ptr(hdr + 0x38, ska)
    B.ptr(hdr + 0x40, B.dict(["Wave"]))
    B.ptr(hdr + 0x90, binfo)
    B.ptr(hdr + 0x98, ext_vals)
    B.ptr(hdr + 0xA0, ext_dict)

    # ----- table de chaînes
    while len(B.b) % 4:
        B.b.append(0)
    str_pos = len(B.b)
    B.b += b"_STR" + b"\x00" * 12
    # (les chaînes sont déjà dans le blob : on les recopie pas, le parseur n'en dépend pas)
    sec1_end = len(B.b)

    # ----- buffers GPU (ordre BfresLibrary : index puis vertex)
    def align(a):
        while len(B.b) % a:
            B.b.append(0)

    align(0x1000)
    buf_start = len(B.b)
    for o, ib in mesh_fields:
        align(8)
        B.put(o + 0x20, "I", len(B.b) - buf_start)
        B.b += ib
    idx_end = len(B.b)
    for f, buffers in vb_fields:
        align(8)
        B.put(f + 0x30, "i", len(B.b) - buf_start)
        for data, _ in buffers:
            align(8)
            B.b += data
    vtx_end = len(B.b)
    align(0x1000)
    mempool = len(B.b)
    B.b += b"\x00" * 0x120
    B.ptr(hdr + 0x88, mempool, 3)
    for f, _ in vb_fields:
        B.ptr(f, mempool, 3)
    for o, _ in mesh_fields:
        B.ptr(o + 0x08, mempool, 3)
    B.put(binfo, "II", 0, mempool - buf_start)
    B.ptr(binfo + 8, buf_start)
    align(0x1000)
    ext_start = len(B.b)
    B.b += bntx
    B.ptr(ext_vals, ext_start, 4)
    B.put(ext_vals + 8, "Q", len(bntx))
    align(0x100)
    rlt = len(B.b)

    # ----- _RLT (une entrée par pointeur)
    by_sec = {s: sorted(p for p, ss in B.ptrs if ss == s) for s in range(5)}
    secs = [(0, sec1_end), (buf_start, idx_end - buf_start), (idx_end, vtx_end - idx_end),
            (mempool, 0x120), (ext_start, rlt - ext_start)]
    out = bytearray(b"_RLT") + struct.pack("<III", rlt, 5, 0)
    idx = 0
    for s in range(5):
        out += struct.pack("<QIIII", 0, secs[s][0], secs[s][1], idx, len(by_sec[s]))
        idx += len(by_sec[s])
    for s in range(5):
        for p in by_sec[s]:
            out += struct.pack("<IHBB", p, 1, 1, 0)
    B.b += out
    B.put(hdr + 0x18, "I", rlt)
    B.put(hdr + 0x1C, "I", len(B.b))
    B.put(str_pos + 8, "I", rlt - str_pos)
    return bytes(B.b)


def build_kcl():
    tris = [((0, 0, 0), (100, 0, 0), (0, 0, -100), 0),
            ((100, 0, 0), (100, 0, -100), (0, 0, -100), 0),
            ((0, 0, 0), (0, 100, 0), (100, 0, 0), 1)]
    positions, normals, prisms = [], [], []
    for a, b, c, attr in tris:
        a, b, c = map(np.array, (a, b, c))
        fn = np.cross(b - a, c - a); fn = fn / np.linalg.norm(fn)
        na = np.cross(fn, c - a); na /= np.linalg.norm(na)
        nb = -np.cross(fn, b - a); nb /= np.linalg.norm(nb)
        nc = np.cross(fn, b - c); nc /= np.linalg.norm(nc)
        length = float(np.dot(b - a, nc))
        pi = len(positions); positions.append(a)
        ni = len(normals); normals += [fn, na, nb, nc]
        prisms.append((length, pi, ni, ni + 1, ni + 2, ni + 3, attr, len(prisms)))
    model = bytearray()
    pos_off = 0x3C
    nrm_off = pos_off + 12 * len(positions)
    prism_off = nrm_off + 12 * len(normals)
    oct_off = prism_off + 20 * len(prisms)
    model += struct.pack("<iiii", pos_off, nrm_off, prism_off, oct_off)
    model += struct.pack("<f3f3I3If", 40.0, -1, -1, -101, 0xFFFFFF80, 0xFFFFFF80, 0xFFFFFF80, 7, 0, 0, 25.0)
    for p in positions: model += struct.pack("<3f", *p)
    for n in normals: model += struct.pack("<3f", *n)
    for pr in prisms: model += struct.pack("<f6HI", *pr)
    model += struct.pack("<I", 0x80000002) + struct.pack("<4H", 0, 1, 2, 0xFFFF)
    head = bytearray(b"\x02\x02\x00\x00")
    model_off = 0x38 + 32 + 4
    head += struct.pack("<III", 0x38, 0x38 + 32, 1)
    head += struct.pack("<3f3f3II", 0, 0, -100, 100, 100, 0, 7, 7, 7, len(prisms))
    head += struct.pack("<8I", *([0x80000000] * 8))
    head += struct.pack("<I", model_off)
    return bytes(head + model)


def build_attribute_byml():
    root = oead.byml.Array([
        oead.byml.Hash({"CameraCode": "", "FloorCode": "Ground", "MaterialCode": "Stone", "WallCode": ""}),
        oead.byml.Hash({"CameraCode": "CameraThrough", "FloorCode": "", "MaterialCode": "Metal", "WallCode": "Wall"}),
    ])
    return bytes(oead.byml.to_binary(root, False, 3))


def pack(files, path):
    w = oead.SarcWriter(oead.Endianness.Little)
    for name, data in files.items():
        w.files[name] = data
    _, sarc = w.write()
    with open(path, "wb") as fh:
        fh.write(oead.yaz0.compress(sarc, data_alignment=0x1000))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    bntx_bytes, tex_info = build_bntx()
    fres, kclb, attr = build_bfres(bntx_bytes), build_kcl(), build_attribute_byml()
    pack({"TestModel.bfres": fres, "TestModel.kcl": kclb, "TestModelAttribute.byml": attr,
          "InitModel.byml": bytes(oead.byml.to_binary(oead.byml.Hash({"X": oead.S32(1)}), False, 3))},
         os.path.join(OUT, "TestModel.szs"))
    pack({"Only.kcl": kclb, "Info.byml": attr}, os.path.join(OUT, "NoBfres.szs"))
    pack({"TestModel.bfres": fres}, os.path.join(OUT, "ModelOnly.szs"))
    with open(os.path.join(OUT, "tex_ref.npy"), "wb") as fh:
        np.save(fh, np.frombuffer(tex_info[0][6], np.uint8))
    print("ok", len(fres), len(kclb))
