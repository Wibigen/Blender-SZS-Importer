# SZS Importer / Exporter (BFRES / KCL) — v2.1 — Made by WibiDev

## v2.1
- **Correctif crash en jeu** (`nn::gfx::TextureImpl::Initialize` au chargement) : l'alignement
  déclaré dans l'en-tête Yaz0 (0x1000 dans SMO) est désormais conservé. Il fixe l'alignement
  du buffer de décompression ; à 0, les textures BNTX étaient mal alignées en mémoire.
- Alignement d'origine de chaque fichier du SARC conservé.
- Sections de la table `_RLT` du BFRES calculées comme BfresLibrary, compteur total de sommets
  ajusté au lieu d'être recalculé, composante w des normales `_n0` conservée.

Add-on Blender pour les `.szs` Nintendo Switch (Super Mario Odyssey en priorité) :
**import** complet (modèles, textures, squelette, animations, collisions + attributs)
et **export** qui reconstruit un `.szs` complet et valide à partir du fichier d'origine.

## Installation
Blender 3.6 → 5.x : *Edit > Preferences > Add-ons > Install from Disk…* → choisir le zip
(désactiver/supprimer la v1.0 avant si elle est installée).
- *File > Import > Nintendo SZS (.szs) — BFRES / KCL*
- *File > Export > Nintendo SZS (.szs) — BFRES / KCL*

## Import
| Élément | Détail |
|---|---|
| Archive | Yaz0 → SARC (+ SARC imbriqués, fichiers sans nom) ; erreur si aucun `.bfres` |
| Modèles | tous les FMDL, shapes LOD 0, normales, UV0-3, couleurs, matériaux |
| Textures | BNTX embarqué dans le BFRES, BNTX de l'archive, et `<Nom>Texture.szs` voisin (SMO). Déswizzle Tegra X1 + BC1/BC2/BC3/BC4/BC5/BC7, RGBA8, BGRA8, RGB565, R8, RG8. Liées par sampler (`_a0` albédo, `_n0` normale, `_e0` émission…) ou suffixe (`_alb`, `_nrm`, `_emm`, `_rgh`, `_mtl`) |
| Squelette | armature + vertex groups, skinning 0/1/2+ (comme Switch Toolbox) |
| Animations | chaque FSKA devient une action (cubique / linéaire / step, quaternion ou Euler). Un `.szs` qui ne contient que des animations s'applique à l'armature active |
| Collisions | KCL v1/v2, un matériau par entrée de `<Nom>Attribute.byml` (ex. `KCL_00 Ground/Stone`) avec les codes SMO éditables dans *Material > Collision SMO (KCL)* |

## Export
L'export part du `.szs` d'origine (mémorisé à l'import) et ne remplace que ce qui a changé :
- **BFRES** : chaque objet est relié à son shape (`szs_model`/`szs_shape`). Géométrie libre
  (nombre de sommets/faces quelconque, UV, normales, tangentes, couleurs, poids),
  matériau changeable parmi ceux du modèle (nom identique). Le fichier est reconstruit
  dans l'ordre de BfresLibrary, tous les pointeurs sont relogés via la table `_RLT`.
  Textures, matériaux, shaders et animations sont conservés tels quels.
- **Shape supprimé** dans Blender → masqué (option).
- **KCL** : entièrement régénéré (prismes + octree, algorithme de KclLibrary) à partir de
  tous les objets de la collection de collision, y compris les nouveaux objets.
  Les codes SMO des matériaux réécrivent `<Nom>Attribute.byml` si besoin.
- **SARC** reconstruit (hash, alignements SARCExt : BFRES 0x2000, BNTX 0x1000) puis Yaz0.
- Le `.szs` produit est relu et vérifié avant d'être écrit.

## Limites
- Un objet **nouveau** dans un modèle n'est pas ajouté comme nouveau shape : joignez-le
  (Ctrl+J) à un objet existant. Les nouveaux objets de collision, eux, sont exportés.
- Le jeu de vertex attributes d'un shape est celui d'origine (format conservé).
- Textures et animations : import seulement (l'export garde celles d'origine).
- Non gérés : BFRES Wii U, BFRES v10 avec buffers/strings externes (TotK), ASTC, BC6H.

## Structure
```
__init__.py        opérateurs import/export, panneau matériaux KCL
yaz0.py            décompression + compression Yaz0
sarc.py            lecture / écriture SARC, reconstruction d'archives imbriquées
byml.py            BYML v2-v7 lecture / écriture, codes de collision SMO
bntx.py            BNTX + déswizzle block-linear Tegra X1
texdecode.py       décodeurs BCn / RGBA (numpy)
bfres.py           parseur BFRES Switch (modèles, matériaux, squelette, FSKA)
bfres_writer.py    reconstruction BFRES (patch + relocation _RLT), encodage des attributs
kcl.py             lecture / écriture KCL (octree de prismes)
blender_build.py   création des objets Blender
blender_export.py  lecture des objets Blender -> géométries jeu
tests/             générateur de fichiers de test + test complet dans Blender
```
Tests : `pip install bpy oead numpy`, cloner BNTX-Extractor, puis
`python tests/make_test_files.py <dossier> <BNTX-Extractor>` et
`python tests/run_blender_test.py <dossier>`.

## Références
Switch Toolbox, BfresLibrary, KclLibrary (KillzXGaming) ; Yaz0/SARC/BNTX d'AboodXD et exelix ;
tables BC7 de texture2ddecoder (K0lb3).
