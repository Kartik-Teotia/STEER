"""Minimal FLAME configuration for STEER inference.

`generic_model.pkl` (FLAME 2020) must be downloaded by the user from
https://flame.is.tue.mpg.de (registration required; the FLAME license does not
permit redistribution) and placed at assets/FLAME2020/generic_model.pkl.
"""
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class FlameConfig:
    flame_geom_path = os.path.join(_ROOT, "assets", "FLAME2020", "generic_model.pkl")
    flame_lmk_path = os.path.join(_ROOT, "assets", "landmark_embedding.npy")
    tex_space_path = ""            # textures unused (geometry-only rendering)
    tex_params = 50
    image_size = 256
    actor = ""
    num_shape_params = 300
    num_exp_params = 100
