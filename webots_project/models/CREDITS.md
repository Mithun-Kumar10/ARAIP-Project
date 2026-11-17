# 3D model credits

These models are used in `worlds/cnn_object_test_v2.wbt` to test my CIFAR-10 CNN on the e-puck camera with
the same classes as my own photos (bird, cat, automobile). All are licensed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) and were downloaded from Sketchfab.

| Folder | Model | Creator | Source |
|---|---|---|---|
| `blue_jay/` | Realistic Blue Jay Bird | ItsKrish7 | https://sketchfab.com/3d-models/realistic-blue-jay-bird-04bf7be89b294f32a2d100f9b4d75f7f |
| `cat/` | Curious Cat 3d model free | rt699448 (iRahulRajput) | https://sketchfab.com/3d-models/curious-cat-3d-model-free-c1465490a7c74d48b599e1d68ff990ef |
| `car/` | Low Poly Car - BMW E30 1985 White | roh3d | https://sketchfab.com/3d-models/low-poly-car-bmw-e30-1985-white-9dea494b447e442fafbddfc7eccbf158 |

Changes made: converted from glTF (.glb, 1k textures) to Wavefront OBJ with trimesh, rotated from Y-up to
Webots' Z-up, scaled (bird 16 cm, cat 40 cm, car 55 cm long) and moved so each model stands on the floor.
