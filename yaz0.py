"""Décompression Yaz0 (SZS).

Format (repris de la lib Yaz0 d'AboodXD utilisée par Switch Toolbox) :
    0x00  'Yaz0'
    0x04  u32 BE  taille décompressée
    0x08  u32 BE  alignement (0 sur la plupart des jeux)
    0x0C  u32     réservé
    0x10  flux : 1 octet de code (8 bits, MSB d'abord)
          bit = 1 -> 1 octet copié tel quel
          bit = 0 -> référence arrière sur 2 ou 3 octets
"""

import struct


class Yaz0Error(Exception):
    pass


def is_yaz0(data) -> bool:
    return len(data) >= 16 and bytes(data[:4]) == b"Yaz0"


def decompress(data) -> bytes:
    if not is_yaz0(data):
        raise Yaz0Error("En-tête Yaz0 absent")

    size = struct.unpack_from(">I", data, 4)[0]
    out = bytearray(size)
    src = 16
    dst = 0
    n = len(data)

    try:
        while dst < size:
            code = data[src]
            src += 1
            for _ in range(8):
                if dst >= size:
                    break
                if code & 0x80:
                    out[dst] = data[src]
                    dst += 1
                    src += 1
                else:
                    b1 = data[src]
                    b2 = data[src + 1]
                    src += 2
                    dist = (((b1 & 0x0F) << 8) | b2) + 1
                    length = b1 >> 4
                    if length == 0:
                        length = data[src] + 0x12
                        src += 1
                    else:
                        length += 2

                    if length > size - dst:
                        length = size - dst
                    start = dst - dist
                    if start < 0:
                        raise Yaz0Error("Référence arrière invalide (fichier corrompu)")

                    if dist >= length:
                        out[dst:dst + length] = out[start:start + length]
                    else:
                        # Chevauchement : le motif de 'dist' octets se répète.
                        pattern = bytes(out[start:dst])
                        reps = length // dist + 1
                        out[dst:dst + length] = (pattern * reps)[:length]
                    dst += length
                code = (code << 1) & 0xFF
    except IndexError:
        raise Yaz0Error(
            "Flux Yaz0 tronqué : %d/%d octets décompressés (source %d/%d)"
            % (dst, size, src, n)
        )

    return bytes(out)


def _match_len(data, a, b, limit):
    """Longueur du préfixe commun data[a:] / data[b:] (recherche dichotomique sur tranches)."""
    lo, hi = 0, limit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if data[a:a + mid] == data[b:b + mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def compress(data, level=1, alignment=0, progress=None):
    """Compresse en Yaz0.

    level 0 : aucun motif (flux de littéraux, valide et instantané)
    level 1 : LZ glouton (chaînes de hachage, 8 candidats)
    level 2 : LZ glouton plus poussé (32 candidats)
    """
    data = bytes(data)
    n = len(data)
    out = bytearray(b"Yaz0")
    out += struct.pack(">II", n, alignment)
    out += b"\x00" * 4

    if level <= 0:
        for i in range(0, n, 8):
            chunk = data[i:i + 8]
            out.append((0xFF << (8 - len(chunk))) & 0xFF)
            out += chunk
        return bytes(out)

    depth = 8 if level == 1 else 32
    window = 0x1000
    max_len = 0x111  # 273
    table = {}
    pos = 0
    group_pos = 0
    code = 0
    bit = 0
    group = bytearray()
    next_report = 1 << 20

    def flush():
        nonlocal code, bit, group
        out.append(code)
        out.extend(group)
        code = 0
        bit = 0
        group = bytearray()

    while pos < n:
        best_len = 0
        best_dist = 0
        if pos + 3 <= n:
            key = data[pos:pos + 3]
            cands = table.get(key)
            if cands:
                limit = min(max_len, n - pos)
                tried = 0
                for c in reversed(cands):
                    dist = pos - c
                    if dist > window:
                        break
                    if data[c + best_len:c + best_len + 1] == data[pos + best_len:pos + best_len + 1] or best_len < 3:
                        L = _match_len(data, c, pos, limit)
                        if L > best_len:
                            best_len, best_dist = L, dist
                            if L == limit:
                                break
                    tried += 1
                    if tried >= depth:
                        break
                if len(cands) > 64:
                    del cands[:-32]

        if best_len >= 3:
            d = best_dist - 1
            if best_len >= 0x12:
                group += bytes(((d >> 8) & 0x0F, d & 0xFF, best_len - 0x12))
            else:
                group += bytes((((best_len - 2) << 4) | ((d >> 8) & 0x0F), d & 0xFF))
            step = best_len
        else:
            code |= 0x80 >> bit
            group.append(data[pos])
            step = 1

        end = pos + step
        for p in range(pos, min(end, n - 2)):
            k = data[p:p + 3]
            lst = table.get(k)
            if lst is None:
                table[k] = [p]
            else:
                lst.append(p)
        pos = end
        bit += 1
        if bit == 8:
            flush()
        if progress and pos >= next_report:
            progress(pos / n)
            next_report += 1 << 20

    if bit:
        flush()
    return bytes(out)
