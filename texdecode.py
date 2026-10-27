"""Décodeurs de textures (numpy, vectorisés) : BC1/2/3/4/5/7, RGBA8, RGB565, R8, RG8.

Sortie : tableau uint8 (H, W, 4) RGBA, ligne 0 = haut de l'image.
Formules alignées sur texture2ddecoder (bcn.cpp) pour BC1-BC5 et BC7.
"""

import numpy as np


class TexDecodeError(Exception):
    pass


# --------------------------------------------------------------------------- #
#  Utilitaires
# --------------------------------------------------------------------------- #
def _blocks_to_image(px, bw, bh, width, height):
    """px : (bh*bw, 16, 4) -> image (H, W, 4)."""
    img = px.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return img[:height, :width]


def _rgb565(c):
    c = c.astype(np.uint32)
    r = ((c >> 8) & 0xF8) | (c >> 13)
    g = ((c >> 3) & 0xFC) | ((c >> 9) & 3)
    b = ((c << 3) & 0xF8) | ((c >> 2) & 7)
    return r, g, b


def _bc1_colors(blocks8, force_4color=False):
    """blocks8 : (N, 8) uint8 -> (N, 16, 4) uint8."""
    c0 = blocks8[:, 0].astype(np.uint32) | (blocks8[:, 1].astype(np.uint32) << 8)
    c1 = blocks8[:, 2].astype(np.uint32) | (blocks8[:, 3].astype(np.uint32) << 8)
    r0, g0, b0 = _rgb565(c0)
    r1, g1, b1 = _rgb565(c1)
    n = len(blocks8)
    pal = np.zeros((n, 4, 4), np.uint32)
    pal[:, 0] = np.stack([r0, g0, b0, np.full(n, 255)], 1)
    pal[:, 1] = np.stack([r1, g1, b1, np.full(n, 255)], 1)
    four = (c0 > c1) | force_4color
    p2_4 = np.stack([(r0 * 2 + r1) // 3, (g0 * 2 + g1) // 3, (b0 * 2 + b1) // 3, np.full(n, 255)], 1)
    p3_4 = np.stack([(r0 + r1 * 2) // 3, (g0 + g1 * 2) // 3, (b0 + b1 * 2) // 3, np.full(n, 255)], 1)
    p2_3 = np.stack([(r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2, np.full(n, 255)], 1)
    p3_3 = np.stack([np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, 255)], 1).astype(np.uint32)
    pal[:, 2] = np.where(four[:, None], p2_4, p2_3)
    pal[:, 3] = np.where(four[:, None], p3_4, p3_3)

    idx = blocks8[:, 4:8].copy().view("<u4").reshape(n)
    sel = (idx[:, None] >> (2 * np.arange(16, dtype=np.uint32))) & 3
    return np.take_along_axis(pal, sel[:, :, None].astype(np.int64).repeat(4, 2), 1).astype(np.uint8)


def _bc3_alpha(blocks8):
    """Bloc alpha BC3/BC4 (8 octets) -> (N, 16) uint8."""
    n = len(blocks8)
    a0 = blocks8[:, 0].astype(np.uint32)
    a1 = blocks8[:, 1].astype(np.uint32)
    pal = np.zeros((n, 8), np.uint32)
    pal[:, 0], pal[:, 1] = a0, a1
    gt = a0 > a1
    for i in range(1, 7):
        pal[:, i + 1] = np.where(gt, (a0 * (7 - i) + a1 * i) // 7, 0)
    for i in range(1, 5):
        pal[:, i + 1] = np.where(gt, pal[:, i + 1], (a0 * (5 - i) + a1 * i) // 5)
    pal[:, 6] = np.where(gt, pal[:, 6], 0)
    pal[:, 7] = np.where(gt, pal[:, 7], 255)
    bits = np.zeros(n, np.uint64)
    for i in range(6):
        bits |= blocks8[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
    sel = (bits[:, None] >> (np.uint64(3) * np.arange(16, dtype=np.uint64))) & np.uint64(7)
    return np.take_along_axis(pal, sel.astype(np.int64), 1).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  BCn
# --------------------------------------------------------------------------- #
def decode_bc1(data, w, h):
    bw, bh = (w + 3) // 4, (h + 3) // 4
    blk = np.frombuffer(data, np.uint8, count=bw * bh * 8).reshape(-1, 8)
    return _blocks_to_image(_bc1_colors(blk), bw, bh, w, h)


def decode_bc2(data, w, h):
    bw, bh = (w + 3) // 4, (h + 3) // 4
    blk = np.frombuffer(data, np.uint8, count=bw * bh * 16).reshape(-1, 16)
    px = _bc1_colors(blk[:, 8:], force_4color=True)
    a = blk[:, :8].astype(np.uint8)
    nib = np.stack([a & 0xF, a >> 4], 2).reshape(-1, 16)
    px[:, :, 3] = nib * 17
    return _blocks_to_image(px, bw, bh, w, h)


def decode_bc3(data, w, h):
    bw, bh = (w + 3) // 4, (h + 3) // 4
    blk = np.frombuffer(data, np.uint8, count=bw * bh * 16).reshape(-1, 16)
    px = _bc1_colors(blk[:, 8:])
    px[:, :, 3] = _bc3_alpha(blk[:, :8])
    return _blocks_to_image(px, bw, bh, w, h)


def decode_bc4(data, w, h, signed=False):
    bw, bh = (w + 3) // 4, (h + 3) // 4
    blk = np.frombuffer(data, np.uint8, count=bw * bh * 8).reshape(-1, 8)
    if signed:
        blk = (blk.astype(np.int16) + 128).astype(np.uint8)  # approximation SNORM -> UNORM
    r = _bc3_alpha(blk)
    px = np.zeros((len(blk), 16, 4), np.uint8)
    px[:, :, 0] = r
    px[:, :, 3] = 255
    return _blocks_to_image(px, bw, bh, w, h)


def decode_bc5(data, w, h, signed=False):
    bw, bh = (w + 3) // 4, (h + 3) // 4
    blk = np.frombuffer(data, np.uint8, count=bw * bh * 16).reshape(-1, 16)
    if signed:
        blk = (blk.astype(np.int16) + 128).astype(np.uint8)
    px = np.zeros((len(blk), 16, 4), np.uint8)
    px[:, :, 0] = _bc3_alpha(blk[:, :8])
    px[:, :, 1] = _bc3_alpha(blk[:, 8:])
    px[:, :, 3] = 255
    return _blocks_to_image(px, bw, bh, w, h)


# ---- BC7 ------------------------------------------------------------------- #
_P2 = [0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80, 0xc800, 0xffec, 0xfe80, 0xe800,
       0xffe8, 0xff00, 0xfff0, 0xf000, 0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce,
       0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c, 0xaaaa, 0xf0f0, 0x5a5a, 0x33cc,
       0x3c3c, 0x55aa, 0x9696, 0xa55a, 0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
       0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c, 0x9336, 0x9cc6, 0x817e, 0xe718,
       0xccf0, 0x0fcc, 0x7744, 0xee22]
_A2 = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15,
       2, 8, 2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6,
       6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15]
_P3 = [0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
       0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
       0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
       0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
       0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424, 0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
       0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
       0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
       0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000, 0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254]
_A3 = [[3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15,
        8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3],
       [15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8,
        15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8]]
_FACTORS = {2: [0, 21, 43, 64], 3: [0, 9, 18, 27, 37, 46, 55, 64],
            4: [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64]}
# numSubsets, partitionBits, rotationBits, indexSelBits, colorBits, alphaBits, endpointPBits, sharedPBits, ib0, ib1
_MODES = [(3, 4, 0, 0, 4, 0, 1, 0, 3, 0), (2, 6, 0, 0, 6, 0, 0, 1, 3, 0), (3, 6, 0, 0, 5, 0, 0, 0, 2, 0),
          (2, 6, 0, 0, 7, 0, 1, 0, 2, 0), (1, 0, 2, 1, 5, 6, 0, 0, 2, 3), (1, 0, 2, 0, 7, 8, 0, 0, 2, 2),
          (1, 0, 0, 0, 7, 7, 1, 0, 4, 0), (2, 6, 0, 0, 5, 5, 1, 0, 2, 0)]

# Tables (partition, pixel) -> sous-ensemble / ancre
_SUB2 = np.array([[(p >> i) & 1 for i in range(16)] for p in _P2], np.int64)
_SUB3 = np.array([[(p >> (2 * i)) & 3 for i in range(16)] for p in _P3], np.int64)
_ANCH2 = np.zeros((64, 16), np.int64)
_ANCH3 = np.zeros((64, 16), np.int64)
for _p in range(64):
    _ANCH2[_p, 0] = 1
    _ANCH2[_p, _A2[_p]] = 1
    _ANCH3[_p, 0] = 1
    _ANCH3[_p, _A3[0][_p]] = 1
    _ANCH3[_p, _A3[1][_p]] = 1


class _Bits:
    """Lecture de bits dans des blocs de 128 bits (vectorisé, positions par bloc)."""

    def __init__(self, blk):
        self.lo = blk[:, :8].copy().view("<u8").reshape(-1)
        self.hi = blk[:, 8:].copy().view("<u8").reshape(-1)

    def get(self, pos, n):
        pos = np.asarray(pos, np.int64)
        n = np.asarray(n, np.int64)
        pos = np.broadcast_to(pos, self.lo.shape).astype(np.int64)
        n = np.broadcast_to(n, self.lo.shape).astype(np.int64)
        out = np.zeros(self.lo.shape, np.uint64)
        lo_part = pos < 64
        p = pos.astype(np.uint64)
        sh_lo = np.where(lo_part, p, 0).astype(np.uint64)
        v_lo = self.lo >> sh_lo
        spill = lo_part & (pos + n > 64)
        sh_hi = np.where(spill, 64 - pos, 0).astype(np.uint64)
        v_hi_spill = np.where(spill, self.hi << np.where(spill, sh_hi, 0).astype(np.uint64), 0).astype(np.uint64)
        sh_h = np.where(~lo_part, pos - 64, 0).astype(np.uint64)
        v_hi = self.hi >> sh_h
        out = np.where(lo_part, v_lo | v_hi_spill, v_hi)
        mask = (np.uint64(1) << n.astype(np.uint64)) - np.uint64(1)
        return (out & mask).astype(np.int64)


def _expand(v, bits):
    v = (v << (8 - bits)) & 0xFF
    return v | (v >> bits)


def _decode_bc7_mode(blk, mode):
    ns, pb, rb, isb, cb, ab, epb, spb, ib0, ib1 = _MODES[mode]
    n = len(blk)
    bits = _Bits(blk)
    pos = mode + 1
    part = bits.get(pos, pb) if pb else np.zeros(n, np.int64)
    pos += pb
    rot = bits.get(pos, rb) if rb else np.zeros(n, np.int64)
    pos += rb
    isel = bits.get(pos, isb) if isb else np.zeros(n, np.int64)
    pos += isb

    mpb = epb or spb
    ep = np.zeros((n, 4, ns * 2), np.int64)  # canal, extrémité
    for ch in range(3):
        for e in range(ns * 2):
            ep[:, ch, e] = bits.get(pos, cb) << mpb
            pos += cb
    if ab:
        for e in range(ns * 2):
            ep[:, 3, e] = bits.get(pos, ab) << mpb
            pos += ab
    else:
        ep[:, 3, :] = 0xFF
    if mpb:
        for s in range(ns):
            pa = bits.get(pos, mpb)
            pos += mpb
            if spb:
                pbv = pa
            else:
                pbv = bits.get(pos, mpb)
                pos += mpb
            for ch in range(4):
                ep[:, ch, 2 * s] |= pa
                ep[:, ch, 2 * s + 1] |= pbv
    cbits = cb + mpb
    for ch in range(3):
        ep[:, ch] = _expand(ep[:, ch], cbits)
    if ab:
        ep[:, 3] = _expand(ep[:, 3], ab + mpb)

    # Sous-ensembles et ancres par pixel
    if ns == 1:
        sub = np.zeros((n, 16), np.int64)
        anchor = np.zeros((n, 16), np.int64)
        anchor[:, 0] = 1
    elif ns == 2:
        sub = _SUB2[part]
        anchor = _ANCH2[part]
    else:
        sub = _SUB3[part]
        anchor = _ANCH3[part]

    def read_indices(start, ib):
        widths = ib - anchor
        offs = start[:, None] + np.concatenate([np.zeros((n, 1), np.int64), np.cumsum(widths, 1)[:, :-1]], 1)
        out = np.zeros((n, 16), np.int64)
        for i in range(16):
            out[:, i] = bits.get(offs[:, i], widths[:, i])
        return out, start + widths.sum(1)

    idx0, end0 = read_indices(np.full(n, pos), ib0)
    if ib1:
        idx1, _ = read_indices(end0, ib1)
    else:
        idx1 = idx0

    f0 = np.array(_FACTORS[ib0], np.int64)
    f1 = np.array(_FACTORS[ib1 or ib0], np.int64)
    fc_c = np.where(isel[:, None] == 0, f0[idx0], f1[idx1])
    fc_a = np.where(isel[:, None] == 0, f1[idx1] if ib1 else f0[idx0], f0[idx0])
    if not ib1:
        fc_a = fc_c

    e0 = np.take_along_axis(ep, (2 * sub)[:, None, :].repeat(4, 1), 2)
    e1 = np.take_along_axis(ep, (2 * sub + 1)[:, None, :].repeat(4, 1), 2)
    px = np.zeros((n, 4, 16), np.int64)
    for ch in range(3):
        px[:, ch] = (e0[:, ch] * (64 - fc_c) + e1[:, ch] * fc_c + 32) >> 6
    px[:, 3] = (e0[:, 3] * (64 - fc_a) + e1[:, 3] * fc_a + 32) >> 6
    px = px.transpose(0, 2, 1)  # (n, 16, 4)

    for r, ch in ((1, 0), (2, 1), (3, 2)):
        m = rot == r
        if m.any():
            tmp = px[m, :, 3].copy()
            px[m, :, 3] = px[m, :, ch]
            px[m, :, ch] = tmp
    return px.astype(np.uint8)


def decode_bc7(data, w, h):
    bw, bh = (w + 3) // 4, (h + 3) // 4
    blk = np.frombuffer(data, np.uint8, count=bw * bh * 16).reshape(-1, 16)
    first = blk[:, 0].astype(np.int64)
    mode = np.full(len(blk), 8, np.int64)
    for m in range(7, -1, -1):
        mode = np.where((first >> m) & 1 == 1, m, mode)
    px = np.zeros((len(blk), 16, 4), np.uint8)
    for m in range(8):
        sel = mode == m
        if sel.any():
            px[sel] = _decode_bc7_mode(blk[sel], m)
    return _blocks_to_image(px, bw, bh, w, h)


# --------------------------------------------------------------------------- #
#  Formats non compressés
# --------------------------------------------------------------------------- #
def decode_rgba8(data, w, h):
    return np.frombuffer(data, np.uint8, count=w * h * 4).reshape(h, w, 4).copy()


def decode_bgra8(data, w, h):
    return decode_rgba8(data, w, h)[:, :, [2, 1, 0, 3]]


def decode_rgb565(data, w, h):
    c = np.frombuffer(data, "<u2", count=w * h)
    r, g, b = _rgb565(c)
    # R5G6B5 Switch : R dans les bits de poids faible
    r, b = b, r
    out = np.stack([r, g, b, np.full_like(r, 255)], 1).astype(np.uint8)
    return out.reshape(h, w, 4)


def decode_r8(data, w, h):
    r = np.frombuffer(data, np.uint8, count=w * h).reshape(h, w)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 0] = r
    out[..., 3] = 255
    return out


def decode_rg8(data, w, h):
    rg = np.frombuffer(data, np.uint8, count=w * h * 2).reshape(h, w, 2)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :2] = rg
    out[..., 3] = 255
    return out
