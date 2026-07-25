# Third-party code and assets

The CC BY-NC 4.0 license in `LICENSE` covers the STEER code and checkpoints
authored by us. The following components are third-party and remain under
their own terms:

## FLAME model (not included)

The FLAME 2020 model file (`generic_model.pkl`) is **not** distributed with
this repository. Obtain it from <https://flame.is.tue.mpg.de> after
registering and accepting the FLAME license (Max Planck Institute for
Intelligent Systems). Place it at `assets/FLAME2020/generic_model.pkl`.

## FLAME/smplx-derived code

`steer/flame/flame_model.py` and `steer/flame/lbs.py` are adapted from the
smplx/FLAME PyTorch code lineage (via the MICA / DualTalk / SMIRK codebases).
These files are © Max-Planck-Gesellschaft zur Förderung der Wissenschaften
e.V. (MPG); the original copyright headers are retained in the files, and
their non-commercial scientific-research terms apply. Our modifications are
limited to import/path adjustments and removal of unused components.

## Bundled small assets

- `steer/flame/blendshapes/{l,r}_eyelid.npy` — SMIRK eyelid blendshapes
  (Retsinas et al., CVPR 2024; <https://github.com/georgeretsi/smirk>).
- `steer/flame/mediapipe/mediapipe_landmark_embedding.npz` — MediaPipe
  landmark embedding (Google, Apache 2.0), as redistributed by the SMIRK/MICA
  code lineage.
- `assets/landmark_embedding.npy` — FLAME landmark embedding from the FLAME
  distribution (MPI-IS); subject to the FLAME license.

## Downloaded at runtime

- `facebook/wav2vec2-large-960h` (Hugging Face hub) — Apache 2.0 (Meta AI).

## Data

The packaged examples contain features derived from the RealTalk corpus
(tracked 61-D FLAME motion, per-speaker audio, gaze cues). No RealTalk video
frames or images are redistributed. Use of the derived features is for
non-commercial research only.
