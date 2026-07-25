"""Semantic-control timelines and the default control recipe.

RECIPE holds the default settings for the control examples:
audio CFG 1.5, behavior CFG 2.0, emotion strength 1.2, 10-step sampling,
raised-cosine ramp over 4 latent tokens. Post-hoc refinements applied only to
control renders: a signed eye-yaw offset during
gaze-avert (the 10-D control vector has no signed eye-direction channel), a
sustained roll offset for head-tilt, and a light Savitzky-Golay smoothing of
the neck channels. Head-shake is a speaking-behavior class; request it on
speaking clips.
"""
import numpy as np
import torch
from scipy.signal import savgol_filter

EMO_NAMES = ["neutral", "happy", "sad", "surprise", "angry", "disgust", "fear"]
EMOS = EMO_NAMES  # alias used by the extracted builders

RECIPE = dict(
    cfg_scale=1.5,
    behavior_cfg_scale=2.0,
    emotion_strength=1.2,
    sample_steps=10,
    seed=2026,
    ramp_latents=4,
    rhythm_str_dyn=0.95,
    gaze_str_contact=0.0,
    gaze_str_avert=0.8,
    eye_yaw_boost=0.25,     # rad, applied as -abs() -> look-left during avert
    tilt_hold_rad=0.20,
    tilt_ramp_frames=12,
    savgol_neck=9,
)


def posthoc_avert_eye_yaw(motion61, boost=0.25, ramp_frames=8):
    """Signed eye-yaw offset during gaze-avert (raised-cosine in/out ramp)."""
    out = motion61.copy()
    T = out.shape[0]
    n = min(ramp_frames, T // 4)
    env = np.ones(T, np.float32)
    if n > 0:
        env[:n] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        env[-n:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n))
    out[:, 57] += -abs(boost) * env
    return out


def posthoc_tilt_hold(motion61, hold_rad=0.20, ramp_frames=12):
    """Sustained roll offset for the tilt class (ramped in)."""
    out = motion61.copy()
    T = out.shape[0]
    n = max(1, ramp_frames)
    rmp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n).astype(np.float32))
    env = np.ones(T, np.float32) * hold_rad
    env[:n] = hold_rad * rmp
    out[:, 55] = env
    return out


def savgol_neck(motion61, win=9, order=2):
    """Light neck smoothing used for the control showcase renders."""
    out = motion61.copy()
    if win >= 3 and out.shape[0] > win:
        if win % 2 == 0:
            win += 1
        out[:, 53:56] = savgol_filter(out[:, 53:56], win, min(order, win - 1),
                                      axis=0, mode="nearest")
    return out


def _ramp(n, ramp=4):
    r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, max(1, ramp)).astype(np.float32))
    return np.concatenate([r, np.ones(max(0, n - ramp), np.float32)])[:n]


def build_combo_timeline(n_lat, *, gaze, head, ramp=4,
                          rhythm_str_dyn=0.95, rhythm_str_still=0.7,
                          gaze_str_contact=0.0, gaze_str_avert=0.8, device="cuda"):
    """Per-latent (n_lat, 10) ctrl + mask for a (gaze, head) combo held constant.

    Training-aligned strengths:
      contact-clip GT → gaze_strength ≈ 0    (eyes near-still)
      avert-clip GT   → gaze_strength ≈ 0.8  (eyes deviate)
      still-clip GT   → rhythm_strength ≈ 0
      dyn-clip GT     → rhythm_strength ≈ 0.95
    """
    ctrl = np.zeros((n_lat, 10), np.float32)
    mask = np.zeros((n_lat, 10), np.float32)
    rmp = _ramp(n_lat, ramp)[:, None]

    # gaze on dims [0:3]
    ctrl[:, 0 if gaze == "contact" else 1] = 1.0
    ctrl[:, 2] = gaze_str_contact if gaze == "contact" else gaze_str_avert
    mask[:, 0:3] = rmp

    # head_rhythm on dims [3:8]
    # still: strength=0 (truly still — observer's neck range ≈ 0 → strength ≈ 0 in training)
    # nod/shake/tilt: strength=rhythm_str_dyn (large to actually fire the axis)
    rhythm_idx = {"still": 3, "nod": 4, "shake": 5, "tilt": 6}[head]
    ctrl[:, rhythm_idx] = 1.0
    ctrl[:, 7] = 0.0 if head == "still" else rhythm_str_dyn
    mask[:, 3:8] = rmp

    return (torch.from_numpy(ctrl).to(device), torch.from_numpy(mask).to(device))


def build_head_only(head, n_lat, ramp, rhythm_str_dyn, device):
    """Condition only the head_rhythm dims [3:8] (gaze + emotion left free)."""
    ctrl = np.zeros((n_lat, 10), np.float32)
    mask = np.zeros((n_lat, 10), np.float32)
    rmp = _ramp(n_lat, ramp)[:, None]
    idx = {"still": 3, "nod": 4, "shake": 5, "tilt": 6}[head]
    ctrl[:, idx] = 1.0
    ctrl[:, 7] = 0.0 if head == "still" else rhythm_str_dyn
    mask[:, 3:8] = rmp
    return torch.from_numpy(ctrl).to(device), torch.from_numpy(mask).to(device)


def build_gaze_only(gaze, n_lat, ramp, gaze_str_contact, gaze_str_avert, device):
    """Condition only the gaze dims [0:3] (head + emotion left free)."""
    ctrl = np.zeros((n_lat, 10), np.float32)
    mask = np.zeros((n_lat, 10), np.float32)
    rmp = _ramp(n_lat, ramp)[:, None]
    ctrl[:, 0 if gaze == "contact" else 1] = 1.0
    ctrl[:, 2] = gaze_str_contact if gaze == "contact" else gaze_str_avert
    mask[:, 0:3] = rmp
    return torch.from_numpy(ctrl).to(device), torch.from_numpy(mask).to(device)


def emo_one_hot(name, strength, device):
    e = torch.zeros(1, len(EMOS), device=device)
    e[0, EMOS.index(name)] = float(strength)
    return e
