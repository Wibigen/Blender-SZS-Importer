"""Test complet import -> modification -> export -> réimport dans un vrai Blender (pip install bpy).

python run_blender_test.py <dossier des fichiers de test>
"""

import math
import os
import sys

import bpy
import bmesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
DATA = sys.argv[1]

import io_import_szs  # noqa: E402
from io_import_szs import bfres, bfres_writer, byml, kcl, sarc  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
io_import_szs.register()
FAIL = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def run_import(path, **kw):
    try:
        return bpy.ops.import_scene.nintendo_szs(filepath=path, **kw)
    except RuntimeError as e:
        return "ERR: " + str(e).strip().splitlines()[-1]


def mesh_objects(col):
    return {o.name: o for o in col.all_objects if o.type == "MESH"}


# ---------------------------------------------------------------- import
print("== Import complet")
src = os.path.join(DATA, "TestModel.szs")
print(run_import(src, mode="BOTH"))
root = bpy.data.collections["TestModel"]
objs = mesh_objects(root)
check({"Body", "Hand", "TestModel"} <= set(objs), "objets : %s" % sorted(objs))
check("Body_alb" in bpy.data.images and bpy.data.images["Body_alb"].size[:] == (64, 64), "texture BC1 décodée")
check("Body_nrm" in bpy.data.images, "texture RGBA8 décodée")
mat = bpy.data.materials["Mat_Body"]
check(any(n.type == "TEX_IMAGE" for n in mat.node_tree.nodes), "texture reliée au matériau")
check("Wave" in bpy.data.actions, "animation importée")
arm = bpy.data.objects["TestModel_Skel"]
bpy.context.scene.frame_set(15)
pb = arm.pose.bones["Arm"]
q = pb.matrix_basis.to_quaternion()
check(abs(q.angle - 0.5) < 1e-3, "rotation animée à f15 : %.4f rad (attendu 0.5)" % q.angle)
check(abs(pb.matrix_basis.translation.length - 0.10) < 1e-4,
      "translation animée à f15 : %.4f (attendu 0.10)" % pb.matrix_basis.translation.length)
kmats = sorted(m.name for m in objs["TestModel"].data.materials)
check(kmats[0].startswith("KCL_00 Ground/Stone") and kmats[1].startswith("KCL_01 Metal/Wall/CameraThrough"),
      "matériaux KCL nommés via Attribute.byml : %s" % kmats)
check(bpy.data.materials[kmats[0]].smo_kcl.FloorCode == "Ground", "codes SMO dans le matériau")

# ---------------------------------------------------------------- export sans modification
print("== Export identique")
out1 = os.path.join(DATA, "Out_identity.szs")
bpy.context.view_layer.objects.active = objs["Body"]
r = bpy.ops.export_scene.nintendo_szs(filepath=out1, collection="TestModel")
check(r == {"FINISHED"}, "export identique")
f0 = {f.name: f.data for f in sarc.open_container(open(src, "rb").read())}
f1 = {f.name: f.data for f in sarc.open_container(open(out1, "rb").read())}
check(set(f0) == set(f1), "mêmes fichiers dans l'archive : %s" % sorted(f1))
import struct  # noqa: E402
ya0 = struct.unpack(">I", open(src, "rb").read()[8:12])[0]
ya1 = struct.unpack(">I", open(out1, "rb").read()[8:12])[0]
check(ya1 == (ya0 or 0x1000), "alignement Yaz0 conservé : 0x%X -> 0x%X" % (ya0, ya1))
check(f0["InitModel.byml"] == f1["InitModel.byml"], "fichiers non touchés conservés à l'octet près")
check(bfres_writer.validate(f1["TestModel.bfres"]) == [], "BFRES reconstruit valide")
import oead  # noqa: E402
s = oead.Sarc(oead.yaz0.decompress(open(out1, "rb").read()))
check(len(list(s.get_files())) == len(f0), "oead relit l'archive exportée")
r0, r1 = bfres.parse(f0["TestModel.bfres"]), bfres.parse(f1["TestModel.bfres"])
b0 = bfres.decode_attribute(r0.models[0].vertex_buffers[0], r0.models[0].vertex_buffers[0].attributes["_p0"])
b1 = bfres.decode_attribute(r1.models[0].vertex_buffers[0], r1.models[0].vertex_buffers[0].attributes["_p0"])
check(b0.shape == b1.shape and abs(b0 - b1).max() < 1e-3, "positions inchangées après aller-retour")
w1 = bfres.decode_attribute(r1.models[0].vertex_buffers[0], r1.models[0].vertex_buffers[0].attributes["_w0"])
check(abs(w1[0, 0] - 128 / 255) < 1e-6 and abs(w1[0, 1] - 127 / 255) < 1e-6, "poids conservés %s" % w1[0, :2])
k1 = kcl.parse(f1["TestModel.kcl"])
t1, a1 = k1.models[0].triangles()
check(len(t1) == 3 and sorted(a1.tolist()) == [0, 0, 1], "KCL régénéré : %d tri, attr %s" % (len(t1), a1.tolist()))

# ---------------------------------------------------------------- modifications
print("== Export modifié")
body = objs["Body"]
bm = bmesh.new()
bm.from_mesh(body.data)
bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=2, use_grid_fill=True)
bm.verts.ensure_lookup_table()
bm.verts[0].co.z += 0.2
bm.to_mesh(body.data)
bm.free()
bpy.data.objects.remove(objs["Hand"])
kcl_col = next(c for c in root.children_recursive if "szs_kcl" in c)
bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 2, 0.25))
cube = bpy.context.active_object
for c in cube.users_collection:
    c.objects.unlink(cube)
kcl_col.objects.link(cube)
new_mat = bpy.data.materials.new("Glace")
new_mat.smo_kcl.is_kcl = True
new_mat.smo_kcl.FloorCode = "Slide"
new_mat.smo_kcl.MaterialCode = "Ice"
cube.data.materials.append(new_mat)

out2 = os.path.join(DATA, "Out_modified.szs")
bpy.context.view_layer.objects.active = body
r = bpy.ops.export_scene.nintendo_szs(filepath=out2, collection="TestModel")
check(r == {"FINISHED"}, "export modifié")
f2 = {f.name: f.data for f in sarc.open_container(open(out2, "rb").read())}
check(bfres_writer.validate(f2["TestModel.bfres"]) == [], "BFRES modifié valide")
r2 = bfres.parse(f2["TestModel.bfres"])
vb = r2.models[0].vertex_buffers[0]
check(vb.vertex_count == 16, "Body : %d sommets (subdivisé, attendu 16)" % vb.vertex_count)
tris = bfres.mesh_triangles(r2.models[0].shapes[0].meshes[0])
check(len(tris) == 18, "Body : %d triangles (attendu 18)" % len(tris))
p = bfres.decode_attribute(vb, vb.attributes["_p0"])
check(abs(p[:, 2].max() - 20.0) < 1e-3 or abs(p[:, 1].max() - 100) < 1e-3, "sommet déplacé exporté")
hand = r2.models[0].shapes[1].meshes[0]
check(hand.index_count == 3 and set(bfres.mesh_indices(hand).tolist()) == {0}, "Hand supprimé -> masqué")
k2 = kcl.parse(f2["TestModel.kcl"])
t2, a2 = k2.models[0].triangles()
check(len(t2) == 3 + 12, "KCL : %d triangles (3 + cube 12)" % len(t2))
ent = byml.attribute_entries(byml.parse(f2["TestModelAttribute.byml"]))
check(len(ent) == 3 and ent[2]["FloorCode"] == "Slide" and ent[2]["MaterialCode"] == "Ice",
      "Attribute.byml : nouvelle entrée %s" % ent[-1])
check(sorted(set(a2.tolist())) == [0, 1, 2], "attributs KCL : %s" % sorted(set(a2.tolist())))
check(r2.external_files == r0.external_files, "BNTX embarqué conservé")
check(len(r2.skeletal_anims) == 1, "animation conservée")

# ---------------------------------------------------------------- réimport du résultat
print("== Réimport du .szs exporté")
bpy.ops.wm.read_factory_settings(use_empty=True)
print(run_import(out2, mode="BOTH"))
col = bpy.data.collections["Out_modified"]
objs = mesh_objects(col)
check(len(objs["Body"].data.vertices) == 16, "réimport Body : %d sommets" % len(objs["Body"].data.vertices))
check("Hand" not in objs, "Hand masqué n'est plus importé")
check(len(objs["TestModel"].data.polygons) == 15, "réimport KCL : %d faces" % len(objs["TestModel"].data.polygons))

# ---------------------------------------------------------------- erreurs
print("== Cas d'erreur")
check(str(run_import(os.path.join(DATA, "NoBfres.szs"), mode="COLLISION")).startswith("ERR: Error: Aucun .bfres"),
      "erreur si pas de .bfres")
check("Aucun .kcl" in str(run_import(os.path.join(DATA, "ModelOnly.szs"), mode="COLLISION")),
      "erreur si mode collision sans .kcl")

print("\nRESULTAT :", "OK" if not FAIL else "%d échec(s)" % len(FAIL))
sys.exit(1 if FAIL else 0)
