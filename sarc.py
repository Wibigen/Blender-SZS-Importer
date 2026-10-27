"""Lecture d'archives SARC (contenu d'un .szs une fois décompressé).

Structure (exelix / Switch Toolbox SARCExt) :
  SARC header (0x14) : 'SARC', u16 taille en-tête, u16 BOM, u32 taille fichier,
                       u32 offset données, u16 version, u16 réservé
  SFAT (0x0C)        : 'SFAT', u16 taille en-tête, u16 nb noeuds, u32 clé de hash
  noeuds (0x10 chacun): u32 hash, u32 attributs, u32 début, u32 fin
                       attributs & 0x01000000 -> nom présent à SFNT + (attr & 0xFFFF) * 4
  SFNT (0x08)        : 'SFNT', u16 taille en-tête, u16 réservé, puis noms (zéro-terminés)
"""

import struct

from . import yaz0


class SarcError(Exception):
    pass


def is_sarc(data) -> bool:
    return len(data) >= 0x14 and bytes(data[:4]) == b"SARC"


def guess_extension(data) -> str:
    head = bytes(data[:8])
    if head[:4] == b"FRES":
        return ".bfres"
    if head[:4] == b"SARC":
        return ".sarc"
    if head[:4] == b"Yaz0":
        return ".szs"
    if head[:4] == b"BNTX":
        return ".bntx"
    if head[:2] in (b"BY", b"YB"):
        return ".byml"
    if head[:4] == b"\x02\x02\x00\x00":
        return ".kcl"
    return ".bin"


class SarcFile:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def __repr__(self):
        return "<SarcFile %s (%d octets)>" % (self.name, len(self.data))


def parse(data):
    """Retourne la liste des SarcFile contenus dans l'archive."""
    if not is_sarc(data):
        raise SarcError("Signature SARC absente")

    bom = bytes(data[6:8])
    if bom == b"\xFE\xFF":
        e = ">"
    elif bom == b"\xFF\xFE":
        e = "<"
    else:
        raise SarcError("BOM SARC invalide : %r" % bom)

    header_size = struct.unpack_from(e + "H", data, 4)[0]
    data_offset = struct.unpack_from(e + "I", data, 0x0C)[0]

    sfat = header_size
    if bytes(data[sfat:sfat + 4]) != b"SFAT":
        raise SarcError("Section SFAT introuvable")
    sfat_hsize, node_count = struct.unpack_from(e + "HH", data, sfat + 4)
    nodes_start = sfat + sfat_hsize

    sfnt = nodes_start + node_count * 0x10
    if bytes(data[sfnt:sfnt + 4]) != b"SFNT":
        raise SarcError("Section SFNT introuvable")
    sfnt_hsize = struct.unpack_from(e + "H", data, sfnt + 4)[0]
    names_start = sfnt + sfnt_hsize

    files = []
    for i in range(node_count):
        off = nodes_start + i * 0x10
        name_hash, attr, start, end = struct.unpack_from(e + "IIII", data, off)
        blob = bytes(data[data_offset + start:data_offset + end])

        if attr & 0x01000000:
            name_pos = names_start + (attr & 0xFFFF) * 4
            zero = data.index(b"\x00", name_pos)
            name = bytes(data[name_pos:zero]).decode("utf-8", "replace")
        else:
            name = "%08X%s" % (name_hash, guess_extension(blob))

        sf = SarcFile(name, blob)
        sf.offset = data_offset + start  # position absolue d'origine
        files.append(sf)
    return files


# --------------------------------------------------------------------------- #
#  Écriture
# --------------------------------------------------------------------------- #
HASH_KEY = 0x65


def name_hash(name, key=HASH_KEY):
    h = 0
    for c in name.encode("utf-8"):
        if c >= 0x80:
            c -= 0x100  # caractère signé, comme le jeu
        h = (h * key + c) & 0xFFFFFFFF
    return h


def guess_alignment(data):
    """Alignements utilisés par SARCExt (exelix) / Switch Toolbox."""
    head = bytes(data[:8])
    if head[:4] in (b"SARC", b"FRES", b"Gfx2", b"AAHS", b"BAHS", b"FFNT"):
        return 0x2000
    if head[:4] in (b"BNTX", b"BNSH", b"FSHA"):
        return 0x1000
    if head[:3] == b"Yaz" or head[:2] in (b"YB", b"BY") or head[:4] in (b"CFNT", b"CGFX"):
        return 0x80
    if head[:4] == b"\x02\x02\x00\x00":  # KCL
        return 0x80
    if head[1:4] in (b"STM", b"WAV") or head[:4] == b"FSTP":
        return 0x20
    if head[:4] == b"AAMP":
        return 8
    return 4


def _align(x, a):
    return (x + a - 1) // a * a


def build(files, endian="<", min_alignment=4):
    """files : liste de SarcFile (ou (nom, data[, alignement])). Retourne les octets du SARC."""
    e = endian
    entries = []
    for f in files:
        if isinstance(f, SarcFile):
            name, data, al = f.name, f.data, getattr(f, "alignment", 0)
        else:
            name, data = f[0], f[1]
            al = f[2] if len(f) > 2 else 0
        al = max(min_alignment, guess_alignment(data), al or 0)
        entries.append((name_hash(name), name, bytes(data), al))
    entries.sort(key=lambda x: (x[0], x[1]))

    # SFNT
    names = bytearray()
    name_offs = []
    for _, name, _, _ in entries:
        name_offs.append(len(names))
        names += name.encode("utf-8") + b"\x00"
        while len(names) % 4:
            names.append(0)

    header_size = 0x14 + 0x0C + 0x10 * len(entries) + 0x08 + len(names)
    max_align = max([e_[3] for e_ in entries] + [4])
    data_start = _align(header_size, max_align)

    blob = bytearray()
    ranges = []
    for _, _, data, al in entries:
        start = _align(data_start + len(blob), al) - data_start
        blob += b"\x00" * (start - len(blob))
        blob += data
        ranges.append((start, start + len(data)))

    out = bytearray()
    out += b"SARC" + struct.pack(e + "H", 0x14)
    out += b"\xFF\xFE" if e == "<" else b"\xFE\xFF"
    out += struct.pack(e + "IIHH", data_start + len(blob), data_start, 0x0100, 0)
    out += b"SFAT" + struct.pack(e + "HHI", 0x0C, len(entries), HASH_KEY)
    for (h, _, _, _), noff, (s, t) in zip(entries, name_offs, ranges):
        out += struct.pack(e + "IIII", h, 0x01000000 | (noff // 4), s, t)
    out += b"SFNT" + struct.pack(e + "HH", 0x08, 0)
    out += names
    out += b"\x00" * (data_start - len(out))
    out += blob
    return bytes(out)


def detect_endian(data):
    return ">" if bytes(data[6:8]) == b"\xFE\xFF" else "<"


def rebuild_container(raw, replacements, prefix="", yaz0_level=1):
    """Reconstruit une archive (Yaz0 ? -> SARC) en remplaçant des fichiers.

    replacements : {chemin aplati (comme open_container) : nouvelles données}
    Les sous-archives touchées sont reconstruites récursivement ;
    le reste est copié à l'identique.
    """
    compressed = yaz0.is_yaz0(raw)
    # L'alignement de l'en-tête Yaz0 est celui du buffer alloué par le jeu
    # pour décompresser : il DOIT être conservé (BNTX/BFRES exigent 0x1000).
    y_align = struct.unpack_from(">I", raw, 8)[0] if compressed else 0
    data = yaz0.decompress(raw) if compressed else raw

    if not is_sarc(data):
        # archive qui contient directement un fichier (ex : BFRES compressé)
        stem = prefix.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0] if prefix else None
        for k, v in replacements.items():
            if k.startswith(prefix) and "/" not in k[len(prefix):]:
                data = v
                break
        return yaz0.compress(data, yaz0_level, y_align) if compressed else bytes(data)

    endian = detect_endian(data)
    data_offset = struct.unpack_from(endian + "I", data, 0x0C)[0]
    out_files = []
    for f in parse(data):
        path = prefix + f.name
        new = replacements.get(path)
        inner_prefix = path + "/"
        touched_inside = any(k.startswith(inner_prefix) for k in replacements)
        if new is not None:
            blob = new
        elif touched_inside and (yaz0.is_yaz0(f.data) or is_sarc(f.data)):
            blob = rebuild_container(f.data, replacements, inner_prefix, yaz0_level)
        else:
            blob = f.data
        # alignement d'origine du fichier (bit de poids faible de sa position), plafonné
        off = getattr(f, "offset", 0)
        orig_al = min(off & -off, 0x2000) if off else 0
        out_files.append((f.name, blob, orig_al))
    built = build(out_files, endian)
    if compressed and y_align == 0 and any(b[:4] in (b"FRES", b"BNTX") for _, b, _ in out_files):
        y_align = 0x1000
    return yaz0.compress(built, yaz0_level, y_align) if compressed else built


def open_container(raw, name="archive", recursive=True, _depth=0):
    """Décompresse (Yaz0) puis ouvre un SARC. Si le contenu n'est pas un SARC
    (ex : .szs qui contient directement un BFRES), il est retourné comme
    fichier unique. Les SARC / Yaz0 imbriqués sont dépliés si recursive=True,
    avec un chemin du type 'sous_archive.szs/fichier.bfres'."""
    data = raw
    if yaz0.is_yaz0(data):
        data = yaz0.decompress(data)

    if not is_sarc(data):
        base = name.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0]
        return [SarcFile(stem + guess_extension(data), data)]

    result = []
    for f in parse(data):
        inner = f.data
        nested = recursive and _depth < 8 and (yaz0.is_yaz0(inner) or is_sarc(inner))
        if nested:
            try:
                for sub in open_container(inner, f.name, True, _depth + 1):
                    result.append(SarcFile(f.name + "/" + sub.name, sub.data))
                continue
            except (SarcError, yaz0.Yaz0Error):
                pass
        result.append(f)
    return result
