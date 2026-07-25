#!/usr/bin/env python3
"""STEER inference on a packaged example.

Base mode (no controls):
    python run_inference.py --example examples/listen_laugh_sdnaCBejMmw

Control mode (semantic edits; combos of gaze x head x emotion):
    python run_inference.py --example examples/speak_WfYmsNBQwg8 \
        --gaze avert --head nod --emotion happy

Output: <example>/output/steer_<tag>.mp4 — the input partner video side by
side with the generated FLAME render (eyes included), plus the motion .npz.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from steer.controls import (EMO_NAMES, RECIPE, build_combo_timeline, build_gaze_only,
                            build_head_only, emo_one_hot, posthoc_avert_eye_yaw,
                            posthoc_tilt_hold, savgol_neck)
from steer.generate import SPF, _chunked_generate, _denormalize, _normalize
from steer.loader import load_steer
from steer.render import _sym_eyelid, init_flame, motion61_to_verts, render_context_frames
from steer.video import write_side_by_side

ROOT = Path(__file__).parent
PRIOR_CKPT = ROOT / "checkpoints" / "steer_prior.pt"
AE_CKPT = ROOT / "checkpoints" / "motion_ae.pt"
STATS = ROOT / "checkpoints" / "stats_61d.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", required=True, help="example folder (see examples/)")
    ap.add_argument("--gaze", choices=["contact", "avert"], default=None)
    ap.add_argument("--head", choices=["still", "nod", "shake", "tilt"], default=None)
    ap.add_argument("--emotion", choices=EMO_NAMES, default=None)
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--seed", type=int, default=RECIPE["seed"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ex = Path(args.example)
    data = np.load(ex / "inputs.npz")
    meta = {k: data[k] for k in data.files}
    n_frames = int(meta["n_frames"])
    device = torch.device(args.device)

    print(f"[load] STEER prior + AE")
    model, ae, margs = load_steer(str(PRIOR_CKPT), str(AE_CKPT), device=args.device)
    stats = np.load(STATS)
    mean_s = stats["motion_mean"].astype(np.float32)
    std_s = stats["motion_std"].astype(np.float32)

    partner_norm = _normalize(meta["partner_motion61"], mean_s, std_s)
    prev_norm = _normalize(meta["prev_motion61"], mean_s, std_s)
    target_request = max(n_frames, 100 + ((n_frames - 100 + 79) // 80) * 80 + 20)
    n_total_lat = target_request // 4
    if partner_norm.shape[0] < target_request:
        partner_norm = np.pad(partner_norm, ((0, target_request - partner_norm.shape[0]), (0, 0)),
                              mode="edge")
    na = target_request * SPF
    audio_t = np.pad(meta["target_audio"], (0, max(0, na - len(meta["target_audio"]))))
    audio_p = np.pad(meta["partner_audio"], (0, max(0, na - len(meta["partner_audio"]))))
    partner_gaze = np.zeros((target_request, 3), np.float32)
    pg = meta["partner_gaze"]
    partner_gaze[: pg.shape[0]] = pg[:target_request]

    # semantic controls (optional)
    timeline = None
    emotion = None
    tag = "base"
    controlled = args.gaze or args.head or args.emotion
    if controlled:
        g = args.gaze or "contact"
        h = args.head or "still"
        if args.gaze and args.head:
            ctrl, mask = build_combo_timeline(
                n_total_lat, gaze=g, head=h, ramp=RECIPE["ramp_latents"],
                rhythm_str_dyn=RECIPE["rhythm_str_dyn"],
                gaze_str_contact=RECIPE["gaze_str_contact"],
                gaze_str_avert=RECIPE["gaze_str_avert"], device=args.device)
        elif args.gaze:
            ctrl, mask = build_gaze_only(g, n_total_lat, RECIPE["ramp_latents"],
                                         RECIPE["gaze_str_contact"],
                                         RECIPE["gaze_str_avert"], args.device)
        else:
            ctrl, mask = build_head_only(h, n_total_lat, RECIPE["ramp_latents"],
                                         RECIPE["rhythm_str_dyn"], args.device)
        timeline = (ctrl, mask)
        if args.emotion:
            emotion = emo_one_hot(args.emotion, RECIPE["emotion_strength"], args.device)
        tag = "_".join(x for x in [args.gaze, args.head, args.emotion] if x)

    print(f"[generate] {ex.name}: {n_frames} frames, controls={tag}")
    pred_norm = _chunked_generate(
        model, ae,
        partner_motion_norm=partner_norm, prev_motion_norm=prev_norm,
        self_audio=audio_t, other_audio=audio_p,
        partner_gaze_feat_full=partner_gaze,
        n_frames_target=target_request,
        behavior_control=None, behavior_mask=None,
        behavior_timeline=timeline,
        cfg_scale=RECIPE["cfg_scale"],
        behavior_cfg_scale=RECIPE["behavior_cfg_scale"],
        sample_steps=RECIPE["sample_steps"],
        seed=args.seed, device=device,
        emotion_control=emotion,
    )
    pred = _denormalize(pred_norm[:n_frames], mean_s, std_s)

    if controlled:
        if args.gaze == "avert":
            pred = posthoc_avert_eye_yaw(pred, RECIPE["eye_yaw_boost"])
        if args.head == "tilt":
            pred = posthoc_tilt_hold(pred, RECIPE["tilt_hold_rad"], RECIPE["tilt_ramp_frames"])
        pred = savgol_neck(pred, RECIPE["savgol_neck"])

    out_dir = ex / "output"
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(out_dir / f"steer_{tag}.npz", motion61=pred)

    print("[render] FLAME (with eyes): partner input + STEER output")
    flame, faces = init_flame()
    verts = motion61_to_verts(flame, _sym_eyelid(pred), np.zeros(100, np.float32))
    frames = render_context_frames(verts, faces, "", args.size, "smirk")
    pverts = motion61_to_verts(flame, _sym_eyelid(meta["partner_motion61"][:n_frames]),
                               np.zeros(100, np.float32))
    partner_frames = render_context_frames(pverts, faces, "", args.size, "smirk")

    out_mp4 = out_dir / f"steer_{tag}.mp4"
    label = "STEER (generated)" if tag == "base" else f"STEER  [{tag.replace('_', ' + ')}]"
    write_side_by_side(partner_frames, frames,
                       meta["target_audio"], meta["partner_audio"],
                       out_mp4, labels=("Partner (tracked input)", label))
    print(f"[done] {out_mp4}")


if __name__ == "__main__":
    main()
