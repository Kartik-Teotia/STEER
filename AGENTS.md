# STEER inference: instructions for coding agents

Generates a target person's 61-D FLAME facial motion in a two-person conversation, then renders it next to the partner. Inference only; there is no training code here.

## Setup
Requires a CUDA GPU, Python 3.10 and `ffmpeg` on PATH.

```bash
conda create -n steer python=3.10 -y && conda activate steer
pip install -r requirements.txt
gdown 1VIzdZPN4PuF4mJGSgXqCw_elw29ehRtB -O checkpoints/steer_prior.pt
sha256sum checkpoints/steer_prior.pt   # e5ec8ed4a57bce0711d7b2c66bd866eef1c5d596ba02b646e73adb9d843a6525
```

`checkpoints/motion_ae.pt` and `checkpoints/stats_61d.npz` are already in the repo. Nothing is downloaded at runtime.

**FLAME 2020 is needed only for rendering, and only the user can get it.** Its license forbids redistribution, so never commit it or fetch it from a third party. Ask the user to register at <https://flame.is.tue.mpg.de> and place `generic_model.pkl` at `assets/FLAME2020/generic_model.pkl`. Without it, use `--no-render`.

## Run
```bash
python run_inference.py --example examples/listen_laugh_y8lxhe7iBcE              # natural response
python run_inference.py --example examples/speak_Awu5Pgd2Oe4 --gaze avert --head still --emotion disgust
python run_inference.py --example examples/listen_laugh_y8lxhe7iBcE --no-render  # motion only
```

| Option | Values |
|---|---|
| `--gaze` | `contact`, `avert` |
| `--head` | `still`, `nod`, `shake`, `tilt` (`shake` is for speaking clips) |
| `--emotion` | `neutral`, `happy`, `sad`, `surprise`, `angry`, `disgust`, `fear` |
| `--seed`, `--size`, `--device` | sampling seed, render size (default 400), torch device (default `cuda`) |

Controls combine freely. Keep the recipe in `steer/controls.py: RECIPE` unless the user asks otherwise.

## Output
In `<example>/output/`, where `<controls>` is `base` or the options joined by `_` (e.g. `avert_still_disgust`):
- `steer_<controls>.npz`: `motion61` with shape `(T, 61)`. Channels: expression 0:50, jaw 50:53, neck 53:56, eye 56:59, eyelid 59:61.
- `steer_<controls>.mp4`: 800x400 at 25 fps with audio, partner on the left, STEER on the right.

## Verify a run
`examples/listen_laugh_y8lxhe7iBcE` should give `motion61` of shape `(435, 61)`, no NaNs, and a 17.4-s video. With the same seed the motion is reproducible; across library versions it differed by at most 0.003 (data std about 0.97).

## Custom input
Create `<folder>/inputs.npz` with `partner_motion61 (T,61)`, `prev_motion61 (12,61)`, `target_audio (T*640,)`, `partner_audio (T*640,)`, `partner_gaze (T,3)`, `n_frames`, `prefix`, `category`. Motion is SMIRK/FLAME at 25 fps; audio is 16 kHz per speaker. See "Example folder format" in `README.md`.

## Rules
- License is CC BY-NC 4.0 (non-commercial); third-party terms are in `NOTICE.md`.
- If rendering fails with an EGL error, check the GPU driver; `PYOPENGL_PLATFORM=egl` is set automatically.
