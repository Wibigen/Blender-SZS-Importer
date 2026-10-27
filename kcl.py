"""Parseur KCL (collisions) — calqué sur KclLibrary (KillzXGaming), utilisée
par Switch Toolbox.

V2 (Switch / Wii U, SMO) :
  0x00 u32 version 0x02020000 (octets 02 02 00 00 quel que soit l'endian)
  0x04 u32 offset octree   (= 0x38 : sert à détecter l'endian)
  0x08 u32 offset tableau des offsets de modèles
  0x0C u32 nb modèles
  0x10 min, 0x1C max (vec3f), 0x28 shift (vec3u), 0x34 nb prismes
Modèle : u32 pos, normales, prismes, octree (relatifs au modèle), f32 épaisseur,
         vec3f min, vec3u masque, vec3u shift, [f32 rayon si pos > 56]
Prisme V2 (20 o) : f32 hauteur, u16 pos, dir, n1, n2, n3, attribut, u32 index global
Prisme V1 (16 o) : idem sans index global, tableau décalé de 0x10 (1-indexé).

Reconstruction du triangle (KCLModel.GetTriangle) :
  A = P[pos]
  cA = N[n1] x N[dir] ; cB = N[n2] x N[dir]
  B = A + cB * (h / (cB . N[n3]))
  C = A + cA * (h / (cA . N[n3]))
"""

import struct

import numpy as np


class KclError(Exception):
    pass


class KclModel:
    def __init__(self):
        self.positions = np.zeros((0, 3), np.float32)
        self.normals = np.zeros((0, 3), np.float32)
        self.prisms = None  # tableau structuré numpy

    def triangles(self):
        """Retourne (sommets (T,3,3) float64, attributs (T,) uint16)."""
        p = self.prisms
        if p is None or len(p) == 0:
            return np.zeros((0, 3, 3)), np.zeros((0,), np.uint16)

        P = self.positions.astype(np.float64)
        N = self.normals.astype(np.float64)
        npos, nnrm = len(P), len(N)
        ok = (
            (p["pos"] < npos) & (p["dir"] < nnrm) & (p["n1"] < nnrm)
            & (p["n2"] < nnrm) & (p["n3"] < nnrm)
        )
        p = p[ok]

        A = P[p["pos"]]
        nd, n1, n2, n3 = N[p["dir"]], N[p["n1"]], N[p["n2"]], N[p["n3"]]
        h = p["length"].astype(np.float64)[:, None]
        cA = np.cross(n1, nd)
        cB = np.cross(n2, nd)
        dA = np.einsum("ij,ij->i", cA, n3)[:, None]
        dB = np.einsum("ij,ij->i", cB, n3)[:, None]
        dA[np.abs(dA) < 1e-12] = 1e-12
        dB[np.abs(dB) < 1e-12] = 1e-12
        B = A + cB * (h / dB)
        C = A + cA * (h / dA)

        tris = np.stack([A, B, C], axis=1)
        # Orientation : la normale du triangle doit suivre la normale de face.
        face_n = np.cross(B - A, C - A)
        flip = np.einsum("ij,ij->i", face_n, nd) < 0
        tris[flip] = tris[flip][:, [0, 2, 1]]
        return tris, p["attr"].copy()


class KclFile:
    def __init__(self):
        self.version = 2
        self.endian = "<"
        self.models = []


def is_kcl(data) -> bool:
    return len(data) >= 0x38 and bytes(data[:4]) == b"\x02\x02\x00\x00"


def _prism_dtype(e, v2):
    fields = [("length", e + "f4"), ("pos", e + "u2"), ("dir", e + "u2"),
              ("n1", e + "u2"), ("n2", e + "u2"), ("n3", e + "u2"), ("attr", e + "u2")]
    if v2:
        fields.append(("gindex", e + "u4"))
    return np.dtype(fields)


def _read_model(data, base, e, v2):
    pos_off, nrm_off, prism_off, oct_off = struct.unpack_from(e + "iiii", data, base)
    if not v2:
        prism_off += 0x10  # tableau 1-indexé en V1

    m = KclModel()
    npos = max(0, (nrm_off - pos_off) // 12)
    nnrm = max(0, (prism_off - nrm_off) // 12)
    psize = 20 if v2 else 16
    nprism = max(0, (oct_off - prism_off) // psize)

    m.positions = np.frombuffer(data, dtype=np.dtype(e + "f4"), count=npos * 3,
                                offset=base + pos_off).reshape(npos, 3)
    m.normals = np.frombuffer(data, dtype=np.dtype(e + "f4"), count=nnrm * 3,
                              offset=base + nrm_off).reshape(nnrm, 3)
    m.prisms = np.frombuffer(data, dtype=_prism_dtype(e, v2), count=nprism,
                             offset=base + prism_off)
    return m


# --------------------------------------------------------------------------- #
#  Écriture (KCLFile.Replace / KCLModel / PolygonOctree de KclLibrary)
# --------------------------------------------------------------------------- #
class KclSettings:
    def __init__(self, thickness=30.0, sphere_radius=25.0, max_tris_in_cube=10,
                 min_cube_size=32, max_cube_size=0x100000, max_root_size=2048,
                 padding=50.0, big_endian=False):
        self.thickness = thickness
        self.sphere_radius = sphere_radius
        self.max_tris_in_cube = max_tris_in_cube
        self.min_cube_size = min_cube_size
        self.max_cube_size = max_cube_size
        self.max_root_size = max_root_size
        self.padding = padding
        self.big_endian = big_endian


MAX_MODEL_PRISMS = 65535 // 4


def _next2exp(v):
    v = float(v)
    if v <= 1:
        return 0
    return int(np.ceil(np.log2(v)))


def _tri_box_overlap(tris, bmin, size):
    """Test conservatif triangle/cube (AABB + plan). tris (M,3,3)."""
    bmax = bmin + size
    tmin = tris.min(axis=1)
    tmax = tris.max(axis=1)
    ok = np.all(tmax >= bmin, axis=1) & np.all(tmin <= bmax, axis=1)
    if not ok.any():
        return ok
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    half = np.broadcast_to(np.asarray(size, np.float64) / 2.0, (3,))
    c = bmin + half
    s = np.einsum("ij,ij->i", n, c[None, :] - tris[:, 0])
    r = (np.abs(n) * half[None, :]).sum(axis=1)
    return ok & (np.abs(s) <= r * (1 + 1e-6) + 1e-9)


class _PolyNode:
    __slots__ = ("children", "indices")


def _build_poly_octree(tris, idx, pos, size, st, depth=0):
    node = _PolyNode()
    node.children = None
    blow = st.sphere_radius * 2 if st.sphere_radius > 0 else 50
    newsize = size + blow
    newpos = pos + size / 2.0 - newsize / 2.0
    inside = idx[_tri_box_overlap(tris[idx], newpos, newsize)] if len(idx) else idx
    half = size / 2.0
    if len(inside) > st.max_tris_in_cube and half >= st.min_cube_size and depth < 24:
        node.children = []
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    cp = pos + half * np.array((x, y, z), np.float64)
                    node.children.append(_build_poly_octree(tris, inside, cp, half, st, depth + 1))
        node.indices = None
    else:
        node.indices = tuple(int(i) for i in np.sort(inside))
    return node


def _build_model(tris, attrs, base_index, st, e):
    """Construit les octets d'un KCLModel (V2)."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    d = np.cross(v1 - v0, v2 - v0)
    keep = np.einsum("ij,ij->i", d, d) >= 0.01
    tris, attrs, v0, v1, v2, d = tris[keep], attrs[keep], v0[keep], v1[keep], v2[keep], d[keep]
    if not len(tris):
        return None, 0

    def norm(a):
        l = np.linalg.norm(a, axis=1, keepdims=True)
        l[l < 1e-20] = 1.0
        return a / l

    d = norm(d)
    na = norm(np.cross(d, v2 - v0))
    nb = norm(-np.cross(d, v1 - v0))
    nc = norm(np.cross(d, v1 - v2))
    length = np.einsum("ij,ij->i", v1 - v0, nc)

    pos32 = v0.astype(np.float32)
    upos, pinv = np.unique(pos32, axis=0, return_inverse=True)
    alln = np.concatenate([d, na, nb, nc]).astype(np.float32)
    unrm, ninv = np.unique(alln, axis=0, return_inverse=True)
    ninv = ninv.reshape(-1)
    pinv = pinv.reshape(-1)
    T = len(tris)
    if len(upos) > 0xFFFF or len(unrm) > 0xFFFF:
        raise KclError("Trop de sommets/normales dans un modèle KCL")

    prisms = np.zeros(T, dtype=_prism_dtype(e, True))
    prisms["length"] = length
    prisms["pos"] = pinv
    prisms["dir"] = ninv[0:T]
    prisms["n1"] = ninv[T:2 * T]
    prisms["n2"] = ninv[2 * T:3 * T]
    prisms["n3"] = ninv[3 * T:4 * T]
    prisms["attr"] = attrs
    prisms["gindex"] = base_index + np.arange(T)

    # Octree
    mn = tris.reshape(-1, 3).min(axis=0) - st.padding
    mx = tris.reshape(-1, 3).max(axis=0) + st.padding
    size = mx - mn
    ex = [_next2exp(s) for s in size]
    csp = min(_next2exp(min(size)), _next2exp(st.max_root_size))
    cube = float(1 << csp)
    shift = (csp, ex[0] - csp, ex[0] - csp + ex[1] - csp)
    mask = tuple((0xFFFFFFFF << x) & 0xFFFFFFFF for x in ex)
    counts = [max(1, (1 << x) // (1 << csp)) for x in ex]

    all_idx = np.arange(T)
    roots = []
    for z in range(counts[2]):
        for y in range(counts[1]):
            for x in range(counts[0]):
                cp = mn + cube * np.array((x, y, z), np.float64)
                roots.append(_build_poly_octree(tris, all_idx, cp, cube, st))

    # --- sérialisation de l'octree (ordre de Nintendo : parcours en largeur)
    total_nodes = 0
    q = [roots]
    pool = {}
    pool_size = 0
    while q:
        arr = q.pop(0)
        total_nodes += len(arr)
        for nd in arr:
            if nd.children is None:
                if nd.indices and nd.indices not in pool:
                    pool[nd.indices] = pool_size
                    pool_size += (len(nd.indices) + 1) * 2
            else:
                q.append(nd.children)
    if pool_size == 0:
        pool[(0,)] = 0
        pool_size = 4
    empty_off = pool_size - 2
    list_base = total_nodes * 4

    oct_bytes = bytearray()
    q = [roots]
    while q:
        arr = q.pop(0)
        start = len(oct_bytes)
        for nd in arr:
            if nd.children is None:
                lp = list_base + (pool[nd.indices] if nd.indices else empty_off)
                key = 0x80000000 | (lp - start - 2)
            else:
                key = (len(arr) + len(q) * 8) * 4
                q.append(nd.children)
            oct_bytes += struct.pack(e + "I", key)
    for ind in pool:
        oct_bytes += struct.pack(e + "%dH" % len(ind), *ind) + struct.pack(e + "H", 0xFFFF)

    header = 0x3C
    pos_off = header
    nrm_off = pos_off + 12 * len(upos)
    prism_off = nrm_off + 12 * len(unrm)
    oct_off = prism_off + 20 * T
    out = bytearray()
    out += struct.pack(e + "iiii", pos_off, nrm_off, prism_off, oct_off)
    out += struct.pack(e + "f", st.thickness)
    out += struct.pack(e + "3f", *mn.astype(np.float32))
    out += struct.pack(e + "3I", *mask)
    out += struct.pack(e + "3I", *shift)
    out += struct.pack(e + "f", st.sphere_radius)
    out += upos.astype(e + "f4").tobytes()
    out += unrm.astype(e + "f4").tobytes()
    out += prisms.tobytes()
    out += oct_bytes
    while len(out) % 4:
        out.append(0)
    return bytes(out), T


def _model_division(tris, idx, pos, box, level):
    """Retourne une liste de 8 groupes {children|indices}, comme CreateModelDivision."""
    groups = []
    for z in range(2):
        for y in range(2):
            for x in range(2):
                cp = pos + box * np.array((x, y, z), np.float64)
                inside = idx[_tri_box_overlap(tris[idx], cp, box)] if len(idx) else idx
                if len(inside) >= MAX_MODEL_PRISMS and level < 6:
                    groups.append({"children": _model_division(tris, inside, cp, box / 2.0, level + 1)})
                else:
                    groups.append({"indices": inside})
    return groups


def build(tris, attrs, settings=None):
    """tris (T,3,3) en espace jeu, attrs (T,) -> octets KCL V2."""
    st = settings or KclSettings()
    e = ">" if st.big_endian else "<"
    tris = np.asarray(tris, np.float64)
    attrs = np.asarray(attrs, np.int64).astype(np.uint16)
    if len(tris) == 0:
        raise KclError("Aucun triangle de collision à exporter")

    mn = tris.reshape(-1, 3).min(axis=0) - st.padding
    mx = tris.reshape(-1, 3).max(axis=0) + st.padding
    size = mx - mn
    ex = [_next2exp(s) for s in size]
    box = np.array([1 << x for x in ex], np.float64)

    models = []

    def make_model(idx):
        data, count = _build_model(tris[idx], attrs[idx], sum(m[1] for m in models), st, e)
        if data is None:
            return None
        models.append((data, count))
        return len(models) - 1

    # Octree des modèles : 8 racines
    if len(tris) < MAX_MODEL_PRISMS:
        mi = make_model(np.arange(len(tris)))
        root = [{"model": mi} for _ in range(8)]
    else:
        def resolve(groups):
            out = []
            for g in groups:
                if "children" in g:
                    out.append({"children": resolve(g["children"])})
                elif len(g["indices"]):
                    out.append({"model": make_model(g["indices"])})
                else:
                    out.append({"model": None})
            return out
        root = resolve(_model_division(tris, np.arange(len(tris)), mn, box / 2.0, 0))

    # Sérialisation de l'octree des modèles (offsets en u32 relatifs au tableau parent)
    oct_words = []
    q = [(root, 0)]
    arrays = []
    while q:
        arr, _ = q.pop(0)
        arrays.append(arr)
        for nd in arr:
            if "children" in nd:
                q.append((nd["children"], 0))
    starts = []
    acc = 0
    for arr in arrays:
        starts.append(acc)
        acc += len(arr)
    child_of = {}
    k = 1
    for ai, arr in enumerate(arrays):
        for nd in arr:
            if "children" in nd:
                child_of[id(nd)] = k
                k += 1
    for ai, arr in enumerate(arrays):
        for nd in arr:
            if "children" in nd:
                oct_words.append((starts[child_of[id(nd)]] - starts[ai]) & 0x3FFFFFFF)
            elif nd.get("model") is None:
                oct_words.append(0xC0000000)
            else:
                oct_words.append(0x80000000 | nd["model"])

    prism_total = sum(m[1] for m in models)
    head_size = 0x38
    oct_off = head_size
    arr_off = oct_off + 4 * len(oct_words)
    model_start = arr_off + 4 * len(models)
    offsets = []
    pos = model_start
    for data, _ in models:
        offsets.append(pos)
        pos += len(data)

    out = bytearray(b"\x02\x02\x00\x00")
    out += struct.pack(e + "III", oct_off, arr_off, len(models))
    out += struct.pack(e + "3f", *mn.astype(np.float32))
    out += struct.pack(e + "3f", *mx.astype(np.float32))
    out += struct.pack(e + "3I", *ex)
    out += struct.pack(e + "I", prism_total)
    out += struct.pack(e + "%dI" % len(oct_words), *oct_words)
    out += struct.pack(e + "%dI" % len(offsets), *offsets)
    for data, _ in models:
        out += data
    return bytes(out)


def parse(data):
    if len(data) < 0x38:
        raise KclError("Fichier KCL trop petit")

    k = KclFile()
    is_v2 = bytes(data[:4]) == b"\x02\x02\x00\x00"

    if is_v2:
        k.version = 2
        k.endian = ">" if struct.unpack_from(">I", data, 4)[0] == 0x38 else "<"
        e = k.endian
        _, oct_off, model_arr, model_count = struct.unpack_from(e + "IIII", data, 0)
        if model_count > 0x10000:
            raise KclError("Nombre de modèles KCL incohérent (%d)" % model_count)
        offsets = struct.unpack_from(e + "%dI" % model_count, data, model_arr)
        for off in offsets:
            k.models.append(_read_model(data, off, e, True))
    else:
        # V1 (Wii / GC) : le fichier est un unique modèle.
        k.version = 1
        first_be = struct.unpack_from(">I", data, 0)[0]
        k.endian = ">" if first_be in (56, 60) else "<"
        k.models.append(_read_model(data, 0, k.endian, False))
    return k
