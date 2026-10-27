"""BYML (versions 2 à 7) — lecture et écriture.

Types Python :
  str, bool, int (i32), float (f32), list, dict, None
  et les enveloppes UInt, Int64, UInt64, Double pour garder le type d'origine.

En-tête : 'BY' (BE) ou 'YB' (LE), u16 version, u32 table des clés,
          u32 table des chaînes, u32 racine.
Noeuds  : 0xA0 chaîne, 0xA1 binaire, 0xC0 tableau, 0xC1 dictionnaire,
          0xC2 table de chaînes, 0xC3 table de chemins, 0xD0 bool, 0xD1 int,
          0xD2 float, 0xD3 uint, 0xD4 int64, 0xD5 uint64, 0xD6 double, 0xFF null.
"""

import struct


class BymlError(Exception):
    pass


class UInt(int):
    pass


class Int64(int):
    pass


class UInt64(int):
    pass


class Double(float):
    pass


T_STR, T_BIN, T_ARR, T_HASH, T_STRTAB, T_PATH = 0xA0, 0xA1, 0xC0, 0xC1, 0xC2, 0xC3
T_BOOL, T_INT, T_FLOAT, T_UINT, T_I64, T_U64, T_DBL, T_NULL = 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xFF


def is_byml(data) -> bool:
    return len(data) >= 0x10 and bytes(data[:2]) in (b"BY", b"YB")


# --------------------------------------------------------------------------- #
#  Lecture
# --------------------------------------------------------------------------- #
class _Reader:
    def __init__(self, data):
        self.d = data
        self.e = ">" if bytes(data[:2]) == b"BY" else "<"
        self.version = self.u16(2)
        self.keys = self.string_table(self.u32(4))
        self.strings = self.string_table(self.u32(8))

    def u8(self, o):
        return self.d[o]

    def u16(self, o):
        return struct.unpack_from(self.e + "H", self.d, o)[0]

    def u24(self, o):
        b = bytes(self.d[o:o + 3])
        return int.from_bytes(b, "big" if self.e == ">" else "little")

    def u32(self, o):
        return struct.unpack_from(self.e + "I", self.d, o)[0]

    def string_table(self, off):
        if off == 0:
            return []
        if self.d[off] != T_STRTAB:
            raise BymlError("Table de chaînes invalide à 0x%X" % off)
        n = self.u24(off + 1)
        out = []
        for i in range(n):
            s = off + self.u32(off + 4 + i * 4)
            e = self.d.index(b"\x00", s)
            out.append(bytes(self.d[s:e]).decode("utf-8", "replace"))
        return out

    def value(self, t, raw):
        if t == T_STR:
            return self.strings[raw]
        if t == T_BOOL:
            return raw != 0
        if t == T_INT:
            return struct.unpack(self.e + "i", struct.pack(self.e + "I", raw))[0]
        if t == T_FLOAT:
            return struct.unpack(self.e + "f", struct.pack(self.e + "I", raw))[0]
        if t == T_UINT:
            return UInt(raw)
        if t == T_I64:
            return Int64(struct.unpack_from(self.e + "q", self.d, raw)[0])
        if t == T_U64:
            return UInt64(struct.unpack_from(self.e + "Q", self.d, raw)[0])
        if t == T_DBL:
            return Double(struct.unpack_from(self.e + "d", self.d, raw)[0])
        if t == T_NULL:
            return None
        if t == T_BIN:
            size = self.u32(raw)
            return bytes(self.d[raw + 4:raw + 4 + size])
        if t in (T_ARR, T_HASH):
            return self.node(raw)
        raise BymlError("Type de noeud BYML inconnu 0x%02X" % t)

    def node(self, off):
        t = self.d[off]
        n = self.u24(off + 1)
        if t == T_ARR:
            types = self.d[off + 4:off + 4 + n]
            vals = off + 4 + ((n + 3) & ~3)
            return [self.value(types[i], self.u32(vals + i * 4)) for i in range(n)]
        if t == T_HASH:
            out = {}
            for i in range(n):
                e = off + 4 + i * 8
                key = self.keys[self.u24(e)]
                out[key] = self.value(self.d[e + 3], self.u32(e + 4))
            return out
        raise BymlError("Noeud racine BYML inattendu 0x%02X" % t)


def parse(data):
    if not is_byml(data):
        raise BymlError("Signature BYML absente")
    r = _Reader(data)
    root = r.u32(0x0C)
    if root and r.d[root] == T_PATH:  # anciens fichiers v1 : table de chemins
        root = r.u32(0x10)
    if root == 0:
        return None
    return r.node(root)


def version_of(data):
    return struct.unpack_from((">" if bytes(data[:2]) == b"BY" else "<") + "H", data, 2)[0]


def endian_of(data):
    return ">" if bytes(data[:2]) == b"BY" else "<"


# --------------------------------------------------------------------------- #
#  Écriture
# --------------------------------------------------------------------------- #
def _collect(node, keys, strings):
    if isinstance(node, dict):
        for k, v in node.items():
            keys.add(k)
            _collect(v, keys, strings)
    elif isinstance(node, list):
        for v in node:
            _collect(v, keys, strings)
    elif isinstance(node, str):
        strings.add(node)


def _type_of(v):
    if v is None:
        return T_NULL
    if isinstance(v, bool):
        return T_BOOL
    if isinstance(v, str):
        return T_STR
    if isinstance(v, dict):
        return T_HASH
    if isinstance(v, list):
        return T_ARR
    if isinstance(v, (bytes, bytearray)):
        return T_BIN
    if isinstance(v, UInt64):
        return T_U64
    if isinstance(v, Int64):
        return T_I64
    if isinstance(v, UInt):
        return T_UINT
    if isinstance(v, Double):
        return T_DBL
    if isinstance(v, int):
        return T_INT
    if isinstance(v, float):
        return T_FLOAT
    raise BymlError("Type Python non sérialisable en BYML : %r" % type(v))


def to_binary(root, endian="<", version=3):
    e = endian
    keys, strings = set(), set()
    _collect(root, keys, strings)
    keys = sorted(keys, key=lambda s: s.encode())
    strings = sorted(strings, key=lambda s: s.encode())
    key_idx = {k: i for i, k in enumerate(keys)}
    str_idx = {s: i for i, s in enumerate(strings)}

    out = bytearray(0x10)

    def align4():
        while len(out) % 4:
            out.append(0)

    def u24(v):
        return v.to_bytes(3, "big" if e == ">" else "little")

    def write_strtab(items):
        start = len(out)
        enc = [s.encode("utf-8") + b"\x00" for s in items]
        out.append(T_STRTAB)
        out.extend(u24(len(items)))
        table = 4 + 4 * (len(items) + 1)
        offs = []
        pos = table
        for s in enc:
            offs.append(pos)
            pos += len(s)
        offs.append(pos)
        for o in offs:
            out.extend(struct.pack(e + "I", o))
        for s in enc:
            out.extend(s)
        align4()
        return start

    key_off = write_strtab(keys) if keys else 0
    str_off = write_strtab(strings) if strings else 0

    pending = []  # (position du champ valeur, objet à écrire plus tard)

    def raw_value(v, field_pos):
        t = _type_of(v)
        if t == T_STR:
            return str_idx[v]
        if t == T_BOOL:
            return 1 if v else 0
        if t == T_INT:
            return struct.unpack(e + "I", struct.pack(e + "i", v))[0]
        if t == T_FLOAT:
            return struct.unpack(e + "I", struct.pack(e + "f", v))[0]
        if t == T_UINT:
            return int(v) & 0xFFFFFFFF
        if t == T_NULL:
            return 0
        pending.append((field_pos, v))
        return 0

    def write_container(v):
        align4()
        start = len(out)
        if isinstance(v, list):
            out.append(T_ARR)
            out.extend(u24(len(v)))
            out.extend(bytes(_type_of(x) for x in v))
            align4()
            for x in v:
                p = len(out)
                out.extend(struct.pack(e + "I", raw_value(x, p)))
        else:
            items = sorted(v.items(), key=lambda kv: key_idx[kv[0]])
            out.append(T_HASH)
            out.extend(u24(len(items)))
            for k, x in items:
                out.extend(u24(key_idx[k]))
                out.append(_type_of(x))
                p = len(out)
                out.extend(struct.pack(e + "I", raw_value(x, p)))
        return start

    root_off = write_container(root) if root is not None else 0
    while pending:
        batch, pending[:] = list(pending), []
        for field_pos, v in batch:
            t = _type_of(v)
            if t in (T_ARR, T_HASH):
                off = write_container(v)
            else:
                align4()
                off = len(out)
                if t == T_I64:
                    out.extend(struct.pack(e + "q", v))
                elif t == T_U64:
                    out.extend(struct.pack(e + "Q", v))
                elif t == T_DBL:
                    out.extend(struct.pack(e + "d", v))
                elif t == T_BIN:
                    out.extend(struct.pack(e + "I", len(v)) + bytes(v))
            struct.pack_into(e + "I", out, field_pos, off)

    out[0:2] = b"BY" if e == ">" else b"YB"
    struct.pack_into(e + "HIII", out, 2, version, key_off, str_off, root_off)
    align4()
    return bytes(out)


# --------------------------------------------------------------------------- #
#  Attributs de collision SMO (<nom>Attribute.byml)
# --------------------------------------------------------------------------- #
SMO_CODE_KEYS = ("FloorCode", "WallCode", "CameraCode", "MaterialCode", "MaterialCodePrefix")

SMO_CODES = {
    "CameraCode": ["", "CameraInvalidThrough", "CameraNoThrough", "CameraNoThroughAlways", "CameraThrough"],
    "FloorCode": ["", "Bed", "Chair", "ClimbSlope", "DamageFire", "DamageFire2D", "Fence", "GrabCeil",
                  "Ground", "IgnoreTouch", "Jump", "JumpSmall", "Needle", "Poison", "Poison2D", "Pole",
                  "Pole10", "Pole20", "Pole30Plus", "Press", "Rolling", "SandSink", "Skate", "Slide", "Slip"],
    "MaterialCode": ["", "Carpet", "Cloth", "Cloud", "ColNoEffect", "ExStarCube", "FlowerForest",
                     "FlowerPeach", "Glass", "Gravel", "Ice", "Kawara", "LavaMarble", "LavaPink", "LavaRed",
                     "LavaWhite", "Lawn", "LawnCap", "LawnCapTower", "LawnDarkGreen", "LawnDeep",
                     "LawnDeepForest", "LawnDeepWaterfall", "LawnForest", "LawnWaterfall", "Marble",
                     "MarioCube", "Metal", "MetalHeavy", "MetalLawn", "NoCode", "NoCollide", "PoisonWater",
                     "Puddle", "Salt", "Sand", "SandDesert", "SandLake", "SandMoon", "SandSea", "Snow",
                     "SnowDeep", "Soil", "SoilSoft", "SqueakWood", "Stone", "StoneRough", "Tatami",
                     "WoodThick", "WoodThin"],
    "MaterialCodePrefix": ["", "InWater", "Shallow", "Wet"],
    "WallCode": ["", "NoAction", "NoCeilGrab", "NoCeilSquat", "NoClimbPole", "NoClingTongue",
                 "NoTraceStick", "NoWallGrab", "OnlyWallHitDown", "ReflectStick",
                 "ReflectStickNoWallGrab", "ThroughStick", "Wall"],
}


def attribute_entries(root):
    """Normalise la racine d'un *Attribute.byml en liste de dicts {code: str}."""
    if not isinstance(root, list):
        return None
    entries = []
    for item in root:
        if not isinstance(item, dict):
            return None
        entries.append({k: (v if isinstance(v, str) else str(v)) for k, v in item.items()})
    return entries


def attribute_label(entry):
    parts = []
    for k in ("FloorCode", "MaterialCodePrefix", "MaterialCode", "WallCode", "CameraCode"):
        v = entry.get(k, "")
        if v:
            parts.append(v)
    return "/".join(parts) or "Default"
