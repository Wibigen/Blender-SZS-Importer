"""Import / export Nintendo Switch .szs (Yaz0 + SARC) : BFRES (modèles, textures BNTX,
animations FSKA) et KCL (collisions + attributs SMO).

Made by WibiDev.
Formats repris de Switch Toolbox / BfresLibrary / KclLibrary (KillzXGaming).
"""

bl_info = {
    "name": "SZS Importer / Exporter (BFRES / KCL) - Super Mario Odyssey",
    "author": "WibiDev",
    "version": (2, 1, 0),
    "blender": (3, 6, 0),
    "location": "Fichier > Importer / Exporter > Nintendo SZS (.szs)",
    "description": "Importe modèles, textures, animations et collisions d'un .szs et reconstruit un .szs valide",
    "category": "Import-Export",
}

import os
import time

if "bpy" in locals():
    import importlib
    from . import (yaz0, sarc, byml, bntx, texdecode, bfres, bfres_writer, kcl,
                   blender_build, blender_export)
    for _m in (yaz0, sarc, byml, texdecode, bntx, bfres, bfres_writer, kcl,
               blender_build, blender_export):
        importlib.reload(_m)

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import bfres, blender_build, blender_export, byml, kcl, sarc, yaz0


class SzsImportError(Exception):
    pass


def _is_bfres(f):
    return f.name.lower().endswith(".bfres") or f.data[:4] == b"FRES"


def _is_kcl(f):
    return f.name.lower().endswith(".kcl") or kcl.is_kcl(f.data)


def load_szs(path, recursive=True):
    with open(path, "rb") as fh:
        raw = fh.read()
    if not (yaz0.is_yaz0(raw) or sarc.is_sarc(raw) or raw[:4] == b"FRES"):
        raise SzsImportError("%s n'est ni un Yaz0, ni un SARC, ni un BFRES" % os.path.basename(path))
    try:
        return sarc.open_container(raw, os.path.basename(path), recursive)
    except (yaz0.Yaz0Error, sarc.SarcError) as e:
        raise SzsImportError("Archive illisible : %s" % e)


def extract_files(files, out_dir, only_models=True):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for f in files:
        if only_models and not (_is_bfres(f) or _is_kcl(f) or f.data[:4] == b"BNTX"):
            continue
        dest = os.path.join(out_dir, *f.name.replace("\\", "/").split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(f.data)
        written.append(dest)
    return written


def _texture_archives(path):
    """Archives de textures voisines (SMO : <Nom>Texture.szs)."""
    folder = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = []
    for cand in (stem + "Texture.szs", stem + "_Texture.szs", stem + "Tex.szs"):
        p = os.path.join(folder, cand)
        if os.path.isfile(p) and os.path.abspath(p) != os.path.abspath(path):
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
#  Propriétés de collision SMO (matériaux)
# --------------------------------------------------------------------------- #
def _code_search(key):
    def search(self, context, edit_text):
        vals = [v for v in byml.SMO_CODES[key] if v]
        t = (edit_text or "").lower()
        return [v for v in vals if t in v.lower()] or vals
    return search


def _code_prop(key, desc):
    kw = dict(name=key, default="", description=desc)
    try:
        return StringProperty(search=_code_search(key), search_options={"SUGGESTION"}, **kw)
    except TypeError:  # Blender < 3.3
        return StringProperty(**kw)


class SMO_KclCodes(bpy.types.PropertyGroup):
    is_kcl: BoolProperty(name="Collision SMO", default=False,
                         description="Ce matériau décrit une entrée de <nom>Attribute.byml")
    FloorCode: _code_prop("FloorCode", "Comportement du sol (Ground, Slide, Poison…)")
    WallCode: _code_prop("WallCode", "Comportement des murs (Wall, NoWallGrab…)")
    CameraCode: _code_prop("CameraCode", "Collision de la caméra")
    MaterialCode: _code_prop("MaterialCode", "Matière (sons / effets : Stone, Lawn, Metal…)")
    MaterialCodePrefix: _code_prop("MaterialCodePrefix", "Préfixe de matière (Wet, InWater…)")


class MATERIAL_PT_smo_kcl(bpy.types.Panel):
    bl_label = "Collision SMO (KCL)"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        m = context.material
        return m is not None and (m.smo_kcl.is_kcl or "kcl_attribute" in m or m.name.startswith("KCL_"))

    def draw(self, context):
        m = context.material
        layout = self.layout
        layout.prop(m.smo_kcl, "is_kcl")
        col = layout.column()
        col.enabled = m.smo_kcl.is_kcl
        for k in byml.SMO_CODE_KEYS:
            col.prop(m.smo_kcl, k)
        if "kcl_attribute" in m:
            layout.label(text="Index d'attribut d'origine : %d" % int(m["kcl_attribute"]))


# --------------------------------------------------------------------------- #
#  Import
# --------------------------------------------------------------------------- #
class IMPORT_OT_szs(bpy.types.Operator, ImportHelper):
    """Importer un .szs Nintendo (modèles, textures, animations, collisions)"""

    bl_idname = "import_scene.nintendo_szs"
    bl_label = "Importer SZS"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    filename_ext = ".szs"
    filter_glob: StringProperty(default="*.szs;*.sarc;*.bfres;*.kcl", options={"HIDDEN"})

    mode: EnumProperty(
        name="Charger",
        items=(
            ("MODEL", "Modèle (BFRES)", "Modèles, textures et animations du .bfres"),
            ("COLLISION", "Collisions (KCL)", "Collisions du .kcl"),
            ("BOTH", "Tout", "Modèle + textures + animations + collisions"),
        ),
        default="BOTH",
    )
    scale: FloatProperty(name="Échelle", default=0.01, min=1e-6, max=1000.0, soft_min=0.001, soft_max=1.0,
                         description="Unités du jeu -> mètres (SMO : 1 unité = 1 cm, donc 0.01)")
    convert_axes: BoolProperty(name="Y-up -> Z-up", default=True)
    import_armature: BoolProperty(name="Squelette", default=True)
    import_weights: BoolProperty(name="Poids (vertex groups)", default=True)
    import_normals: BoolProperty(name="Normales personnalisées", default=True)
    import_textures: BoolProperty(name="Textures BNTX", default=True,
                                  description="Décode les textures embarquées et les relie aux matériaux")
    search_texture_szs: BoolProperty(name="Chercher <Nom>Texture.szs", default=True,
                                     description="Charge aussi les textures d'une archive voisine (SMO)")
    import_animations: BoolProperty(name="Animations", default=True,
                                    description="Crée une action par animation squelettique (FSKA)")
    merge_collision: BoolProperty(name="Fusionner les modèles KCL", default=True)
    recursive: BoolProperty(name="Ouvrir les SARC imbriqués", default=True)
    texture_dir: StringProperty(name="Dossier textures (option)", subtype="DIR_PATH", default="",
                                description="PNG/DDS exportés, utilisés si une texture n'est pas dans un BNTX")
    extract: BoolProperty(name="Extraire sur le disque", default=False)
    extract_all: BoolProperty(name="Extraire tous les fichiers", default=False)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "mode")
        box = layout.box()
        box.label(text="Transformation")
        box.prop(self, "scale")
        box.prop(self, "convert_axes")
        if self.mode in {"MODEL", "BOTH"}:
            box = layout.box()
            box.label(text="Modèle")
            box.prop(self, "import_armature")
            row = box.row()
            row.enabled = self.import_armature
            row.prop(self, "import_weights")
            box.prop(self, "import_normals")
            box.prop(self, "import_textures")
            row = box.row()
            row.enabled = self.import_textures
            row.prop(self, "search_texture_szs")
            box.prop(self, "import_animations")
            box.prop(self, "texture_dir")
        if self.mode in {"COLLISION", "BOTH"}:
            box = layout.box()
            box.label(text="Collisions")
            box.prop(self, "merge_collision")
        box = layout.box()
        box.label(text="Archive")
        box.prop(self, "recursive")
        box.prop(self, "extract")
        row = box.row()
        row.enabled = self.extract
        row.prop(self, "extract_all")
        layout.label(text="Made by WibiDev")

    def execute(self, context):
        t0 = time.time()
        try:
            n, warnings = self._run(context)
        except SzsImportError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        except (bfres.BfresError, kcl.KclError, byml.BymlError) as e:
            self.report({"ERROR"}, "Erreur de format : %s" % e)
            return {"CANCELLED"}
        for w in warnings[:10]:
            self.report({"WARNING"}, w)
        self.report({"INFO"}, "%d élément(s) importé(s) en %.2fs" % (n, time.time() - t0))
        return {"FINISHED"}

    def _run(self, context):
        path = self.filepath
        files = load_szs(path, self.recursive)
        stem = os.path.splitext(os.path.basename(path))[0]
        by_name = {f.name: f for f in files}

        bfres_files = [f for f in files if _is_bfres(f)]
        kcl_files = [f for f in files if _is_kcl(f)]

        if not bfres_files:
            names = ", ".join(f.name for f in files[:12]) or "(vide)"
            raise SzsImportError("Aucun .bfres dans %s. Contenu : %s" % (os.path.basename(path), names))
        if self.mode == "COLLISION" and not kcl_files:
            raise SzsImportError("Aucun .kcl dans %s (le .bfres est présent mais pas de collisions)"
                                 % os.path.basename(path))

        if self.extract:
            out_dir = os.path.join(os.path.dirname(path), stem + "_extrait")
            written = extract_files(files, out_dir, only_models=not self.extract_all)
            self.report({"INFO"}, "%d fichier(s) extrait(s) dans %s" % (len(written), out_dir))

        st = blender_build.Settings(
            scale=self.scale, convert_axes=self.convert_axes,
            import_armature=self.import_armature, import_weights=self.import_weights,
            import_normals=self.import_normals, merge_collision=self.merge_collision,
            texture_dir=bpy.path.abspath(self.texture_dir) if self.texture_dir else "",
            import_textures=self.import_textures, import_animations=self.import_animations,
        )

        # Tout est analysé AVANT de toucher la scène.
        parsed_models = []
        parsed_kcl = []
        if self.mode in {"MODEL", "BOTH"}:
            for f in bfres_files:
                res = bfres.parse(f.data)
                if not any(m.shapes for m in res.models) and not res.skeletal_anims:
                    st.warnings.append("%s ne contient ni mesh ni animation" % f.name)
                parsed_models.append((f.name, res))
        if self.mode in {"COLLISION", "BOTH"}:
            if not kcl_files:
                st.warnings.append("Pas de .kcl dans l'archive : collisions ignorées")
            for f in kcl_files:
                entries = None
                st_name = f.name.rsplit(".", 1)[0] + "Attribute.byml"
                if st_name in by_name:
                    try:
                        entries = byml.attribute_entries(byml.parse(by_name[st_name].data))
                    except byml.BymlError as e:
                        st.warnings.append("%s : %s" % (st_name, e))
                parsed_kcl.append((f.name, kcl.parse(f.data), entries))

        texlib = blender_build.TextureLibrary(st)
        if self.mode in {"MODEL", "BOTH"} and self.import_textures:
            for f in files:
                if f.data[:4] == b"BNTX":
                    texlib.add_bntx(f.data, f.name)
            if self.search_texture_szs:
                for p in _texture_archives(path):
                    try:
                        for f in load_szs(p, True):
                            if f.data[:4] == b"BNTX":
                                texlib.add_bntx(f.data, f.name)
                            elif _is_bfres(f):
                                for n, d in bfres.parse(f.data).external_files.items():
                                    if d[:4] == b"BNTX":
                                        texlib.add_bntx(d, n)
                    except Exception as e:  # noqa: BLE001
                        st.warnings.append("%s : %s" % (os.path.basename(p), e))

        anim_target = context.active_object if (context.active_object and
                                                context.active_object.type == "ARMATURE") else None
        if context.active_object and context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        root = bpy.data.collections.new(stem)
        context.scene.collection.children.link(root)
        root["szs_source"] = os.path.abspath(path)
        root["szs_scale"] = self.scale
        root["szs_convert_axes"] = self.convert_axes

        created = 0
        for name, res in parsed_models:
            created += blender_build.import_bfres(res, name, st, root, context, texlib, anim_target)
        for name, kf, entries in parsed_kcl:
            created += blender_build.import_kcl(kf, name, st, root, entries)

        if created == 0:
            n_anim = sum(len(res.skeletal_anims) for _, res in parsed_models)
            if n_anim:
                raise SzsImportError("%d animation(s) trouvée(s) mais aucune armature : sélectionnez "
                                     "l'armature cible (objet actif) puis relancez l'import" % n_anim)
            raise SzsImportError("Fichiers lus mais aucun maillage exploitable n'a été trouvé")
        return created, st.warnings


# --------------------------------------------------------------------------- #
#  Export
# --------------------------------------------------------------------------- #
def _szs_collections(self, context):
    items = []
    for c in bpy.data.collections:
        if "szs_source" in c:
            items.append((c.name, c.name, c["szs_source"]))
    return items or [("NONE", "(aucune collection importée)", "")]


def _active_szs_collection(context):
    obj = context.active_object
    if obj is not None:
        for c in bpy.data.collections:
            if "szs_source" in c and obj.name in c.all_objects:
                return c
    for c in bpy.data.collections:
        if "szs_source" in c:
            return c
    return None


class EXPORT_OT_szs(bpy.types.Operator, ExportHelper):
    """Reconstruire un .szs complet (modèles + collisions) à partir d'une collection importée"""

    bl_idname = "export_scene.nintendo_szs"
    bl_label = "Exporter SZS"
    bl_options = {"REGISTER"}

    filename_ext = ".szs"
    filter_glob: StringProperty(default="*.szs", options={"HIDDEN"})

    collection: EnumProperty(name="Collection", items=_szs_collections,
                             description="Collection créée par l'import (contient le chemin du .szs d'origine)")
    source_path: StringProperty(name="SZS d'origine", subtype="FILE_PATH", default="",
                                description="Vide = celui mémorisé à l'import")
    export_model: BoolProperty(name="Modèles (BFRES)", default=True)
    export_collision: BoolProperty(name="Collisions (KCL)", default=True)
    hide_missing: BoolProperty(name="Masquer les shapes supprimés", default=True,
                               description="Un shape sans objet dans Blender devient invisible dans le jeu")
    compression: EnumProperty(
        name="Compression Yaz0",
        items=(("1", "Normale", "LZ (≈ taille d'origine)"),
               ("2", "Maximale", "Plus lente, un peu plus petite"),
               ("0", "Aucune", "Instantané, fichier plus gros (toujours valide)")),
        default="1",
    )
    kcl_thickness: FloatProperty(name="Épaisseur des prismes", default=30.0, min=0.0)
    kcl_sphere_radius: FloatProperty(name="Rayon de sphère", default=25.0, min=0.0)
    kcl_max_tris: IntProperty(name="Triangles max par cube", default=10, min=1, max=1024)
    kcl_min_cube: IntProperty(name="Cube minimal", default=32, min=1, max=4096)

    def invoke(self, context, event):
        col = _active_szs_collection(context)
        if col is not None:
            self.collection = col.name
            src = col["szs_source"]
            self.filepath = os.path.join(os.path.dirname(src),
                                         os.path.splitext(os.path.basename(src))[0] + "_mod.szs")
        return super().invoke(context, event)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "collection")
        layout.prop(self, "source_path")
        box = layout.box()
        box.prop(self, "export_model")
        row = box.row()
        row.enabled = self.export_model
        row.prop(self, "hide_missing")
        box.prop(self, "export_collision")
        col = box.column()
        col.enabled = self.export_collision
        col.prop(self, "kcl_thickness")
        col.prop(self, "kcl_sphere_radius")
        col.prop(self, "kcl_max_tris")
        col.prop(self, "kcl_min_cube")
        layout.prop(self, "compression")
        layout.label(text="Made by WibiDev")

    def execute(self, context):
        col = bpy.data.collections.get(self.collection)
        if col is None or "szs_source" not in col:
            self.report({"ERROR"}, "Choisissez une collection issue de l'import SZS")
            return {"CANCELLED"}
        src = bpy.path.abspath(self.source_path) if self.source_path else col["szs_source"]
        if not os.path.isfile(src):
            self.report({"ERROR"}, "SZS d'origine introuvable : %s (renseignez « SZS d'origine »)" % src)
            return {"CANCELLED"}
        out = self.filepath
        if os.path.abspath(out) == os.path.abspath(src):
            self.report({"ERROR"}, "Choisissez un autre nom que le .szs d'origine (il sert de base)")
            return {"CANCELLED"}
        if context.active_object and context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        st = blender_export.ExportSettings(
            scale=float(col.get("szs_scale", 0.01)),
            convert_axes=bool(col.get("szs_convert_axes", True)),
            export_model=self.export_model, export_collision=self.export_collision,
            hide_missing=self.hide_missing, yaz0_level=int(self.compression),
            kcl_settings=kcl.KclSettings(thickness=self.kcl_thickness,
                                         sphere_radius=self.kcl_sphere_radius,
                                         max_tris_in_cube=self.kcl_max_tris,
                                         min_cube_size=self.kcl_min_cube),
        )
        t0 = time.time()
        try:
            written = blender_export.export_szs(col, src, out, st, context.evaluated_depsgraph_get())
        except (blender_export.ExportError, bfres.BfresError, kcl.KclError,
                blender_export.bfres_writer.BfresWriteError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        for line in st.report[:10]:
            self.report({"WARNING"}, line)
        self.report({"INFO"}, "SZS écrit (%s) en %.1fs : %s" % (
            os.path.basename(out), time.time() - t0, ", ".join(written)))
        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_szs.bl_idname, text="Nintendo SZS (.szs) — BFRES / KCL")


def menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_szs.bl_idname, text="Nintendo SZS (.szs) — BFRES / KCL")


classes = (SMO_KclCodes, MATERIAL_PT_smo_kcl, IMPORT_OT_szs, EXPORT_OT_szs)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Material.smo_kcl = PointerProperty(type=SMO_KclCodes)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    del bpy.types.Material.smo_kcl
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
