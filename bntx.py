"""BNTX (textures Switch) : lecture, déswizzle Tegra X1 (block linear), décodage.

Structures d'après BNTX-Extractor (AboodXD) et Switch Toolbox :
  en-tête BNTX (0x20) puis 'NX  ' : u32 nb, q tableau de ptrs BRTI, q bloc data, q dict, u32
  BRTI : 0x12 u16 tile mode (1 = linéaire), 0x16 u16 nb mips, 0x1C u32 format,
         0x24 largeur, 0x28 hauteur, 0x30 nb faces, 0x34 log2 hauteur de bloc,
         0x50 taille, 0x54 alignement, 0x58 sélecteurs de canaux, 0x60 nom, 0x70 ptrs mips
"""

import struct

import numpy as np

from . import texdecode as td


class BntxError(Exception):
    pass


FORMAT_NAMES = {
    0x0b: "RGBA8", 0x0c: "BGRA8", 0x07: "RGB565", 0x02: "R8", 0x09: "RG8",
    0x1a: "BC1", 0x1b: "BC2", 0x1c: "BC3", 0x1d: "BC4", 0x1e: "BC5", 0x1f: "BC6H", 0x20: "BC7",
}
BLOCK = {0x1a: 4, 0x1b: 4, 0x1c: 4, 0x1d: 4, 0x1e: 4, 0x1f: 4, 0x20: 4}
BPP = {0x0b: 4, 0x0c: 4, 0x07: 2, 0x02: 1, 0x09: 2, 0x1a: 8, 0x1b: 16, 0x1c: 16,
       0x1d: 8, 0x1e: 16, 0x1f: 16, 0x20: 16}
ASTC_BLOCKS = {0x2d: (4, 4), 0x2e: (5, 4), 0x2f: (5, 5), 0x30: (6, 5), 0x31: (6, 6), 0x32: (8, 5),
               0x33: (8, 6), 0x34: (8, 8), 0x35: (10, 5), 0x36: (10, 6), 0x37: (10, 8),
               0x38: (10, 10), 0x39: (12, 10), 0x3a: (12, 12)}


class Texture:
    def __init__(self):
        self.name = ""
        self.width = 0
        self.height = 0
        self.format = 0
        self.tile_linear = False
        self.block_height_log2 = 0
        self.comp_sel = 0x05040302
        self.num_faces = 1
        self.data = b""

    @property
    def type_id(self):
        return self.format >> 8

    @property
    def is_srgb(self):
        return (self.format & 0xFF) == 0x06

    @property
    def format_name(self):
        t = self.type_id
        if t in ASTC_BLOCKS:
            return "ASTC%dx%d" % ASTC_BLOCKS[t]
        return FORMAT_NAMES.get(t, "0x%04X" % self.format) + ("_SRGB" if self.is_srgb else "")


def is_bntx(data):
    return len(data) >= 0x40 and bytes(data[:4]) == b"BNTX"


def parse(data):
    if not is_bntx(data):
        raise BntxError("Signature BNTX absente")
    bom = bytes(data[0x0C:0x0E])
    e = "<" if bom == b"\xFF\xFE" else ">"
    if bytes(data[0x20:0x24]) != b"NX  ":
        raise BntxError("Section NX introuvable")
    count, info_ptrs = struct.unpack_from(e + "Iq", data, 0x24)

    textures = []
    for i in range(count):
        p = struct.unpack_from(e + "q", data, info_ptrs + i * 8)[0]
        if bytes(data[p:p + 4]) != b"BRTI":
            continue
        t = Texture()
        tile = struct.unpack_from(e + "H", data, p + 0x12)[0]
        t.tile_linear = tile == 1
        t.format = struct.unpack_from(e + "I", data, p + 0x1C)[0]
        t.width, t.height = struct.unpack_from(e + "ii", data, p + 0x24)
        t.num_faces, t.block_height_log2 = struct.unpack_from(e + "ii", data, p + 0x30)
        size, _align, t.comp_sel = struct.unpack_from(e + "iiI", data, p + 0x50)
        name_off, _parent, ptrs = struct.unpack_from(e + "qqq", data, p + 0x60)
        nlen = struct.unpack_from(e + "H", data, name_off)[0]
        t.name = bytes(data[name_off + 2:name_off + 2 + nlen]).decode("utf-8", "replace")
        data_off = struct.unpack_from(e + "q", data, ptrs)[0]
        t.data = bytes(data[data_off:data_off + size])
        textures.append(t)
    return textures


# --------------------------------------------------------------------------- #
#  Déswizzle
# --------------------------------------------------------------------------- #
def _div_up(n, d):
    return (n + d - 1) // d


def _round_up(x, y):
    return ((x - 1) | (y - 1)) + 1


def block_linear_addresses(width_blocks, height_blocks, bpp, block_height):
    """Adresse (en octets) de chaque élément (y, x) — Tegra X1 TRM."""
    y = np.arange(height_blocks, dtype=np.int64)[:, None]
    x = np.arange(width_blocks, dtype=np.int64)[None, :]
    gobs_w = _div_up(width_blocks * bpp, 64)
    xb = x * bpp
    gob = ((y // (8 * block_height)) * 512 * block_height * gobs_w
           + (xb // 64) * 512 * block_height
           + (y % (8 * block_height) // 8) * 512)
    return (gob + ((xb % 64) // 32) * 256 + ((y % 8) // 2) * 64
            + ((xb % 32) // 16) * 32 + (y % 2) * 16 + (xb % 16))


def deswizzle(tex):
    t = tex.type_id
    blk = BLOCK.get(t, 1)
    bpp = BPP[t]
    wb, hb = _div_up(tex.width, blk), _div_up(tex.height, blk)
    src = np.frombuffer(tex.data, np.uint8)
    if tex.tile_linear:
        pitch = _round_up(wb * bpp, 32)
        addr = (np.arange(hb)[:, None] * pitch + np.arange(wb)[None, :] * bpp).astype(np.int64)
    else:
        bh = 1 << max(0, min(tex.block_height_log2, 5))
        addr = block_linear_addresses(wb, hb, bpp, bh)
    idx = addr.reshape(-1, 1) + np.arange(bpp)
    valid = idx < len(src)
    out = np.zeros(idx.shape, np.uint8)
    out[valid] = src[idx[valid]]
    return out.tobytes()


# --------------------------------------------------------------------------- #
#  Décodage complet
# --------------------------------------------------------------------------- #
def _apply_comp_sel(img, comp_sel):
    if comp_sel == 0:
        return img
    src = [img[..., 0], img[..., 1], img[..., 2], img[..., 3]]
    out = np.empty_like(img)
    for ch in range(4):
        sel = (comp_sel >> (8 * ch)) & 0xFF
        if sel == 0:
            out[..., ch] = 0
        elif sel == 1:
            out[..., ch] = 255
        elif 2 <= sel <= 5:
            out[..., ch] = src[sel - 2]
        else:
            out[..., ch] = src[ch]
    return out


def decode(tex):
    """Retourne une image uint8 (H, W, 4) RGBA (haut en premier)."""
    t = tex.type_id
    if t in ASTC_BLOCKS:
        raise BntxError("%s : format %s non géré" % (tex.name, tex.format_name))
    if t not in BPP or t == 0x1f:
        raise BntxError("%s : format %s non géré" % (tex.name, tex.format_name))
    raw = deswizzle(tex)
    w, h = tex.width, tex.height
    signed = (tex.format & 0xFF) == 0x02
    if t == 0x1a:
        img = td.decode_bc1(raw, w, h)
    elif t == 0x1b:
        img = td.decode_bc2(raw, w, h)
    elif t == 0x1c:
        img = td.decode_bc3(raw, w, h)
    elif t == 0x1d:
        img = td.decode_bc4(raw, w, h, signed)
    elif t == 0x1e:
        img = td.decode_bc5(raw, w, h, signed)
    elif t == 0x20:
        img = td.decode_bc7(raw, w, h)
    elif t == 0x0b:
        img = td.decode_rgba8(raw, w, h)
    elif t == 0x0c:
        img = td.decode_bgra8(raw, w, h)
    elif t == 0x07:
        img = td.decode_rgb565(raw, w, h)
    elif t == 0x02:
        img = td.decode_r8(raw, w, h)
    else:
        img = td.decode_rg8(raw, w, h)
    img = _apply_comp_sel(img, tex.comp_sel)
    if t == 0x1e:  # BC5 = normales XY : reconstruit Z pour Blender
        x = img[..., 0].astype(np.float32) / 127.5 - 1.0
        y = img[..., 1].astype(np.float32) / 127.5 - 1.0
        z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
        img = img.copy()
        img[..., 2] = np.clip((z * 0.5 + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        img[..., 3] = 255
    return img
