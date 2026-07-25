"""FLAME rendering for STEER: 61-D motion -> vertices -> shaded frames.

FLAME renderer:
the head is rendered with a skin material while the eyeball region is split
out and vertex-colored (iris + pupil derived from the neutral-pose eyeball
geometry), and SMIRK's eyelid blendshapes are applied as additive vertex
offsets. Do not swap this for a generic FLAME renderer: the eyes are part of
the model output (motion dims 56:59 eye rotation, 59:61 eyelid opening).
"""
import math
import os

import cv2
import numpy as np
import torch

from steer.flame.config import FlameConfig
from steer.flame.flame_model import FLAME

LEFT_EYE_RANGE = (3931, 4477)   # 546 verts
RIGHT_EYE_RANGE = (4477, 5023)  # 546 verts
CONTEXT_SZ = 300
FPS = 25

_iris_set = None
_pupil_set = None
_eyeball_set = None
_SMIRK_EYELID_BS = None


def init_flame():
    """Load the FLAME 2020 model (requires assets/FLAME2020/generic_model.pkl)."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    flame = FLAME(FlameConfig()).cuda().eval()
    faces = flame.faces.cpu().numpy() if torch.is_tensor(flame.faces) else np.asarray(flame.faces)
    return flame, faces


def _sym_eyelid(m):
    """Average the two eyelid channels (SMIRK left/right ambiguity)."""
    out = m.copy()
    if out.shape[1] >= 61:
        lid = 0.5 * (out[:, 59] + out[:, 60])
        out[:, 59] = lid
        out[:, 60] = lid
    return out


def _compute_iris_pupil(neutral_verts):
    """Find iris/pupil vertex indices from neutral-pose eyeball geometry."""
    iris_set = set()
    pupil_set = set()
    for start, end in [LEFT_EYE_RANGE, RIGHT_EYE_RANGE]:
        ev = neutral_verts[start:end]
        center = ev.mean(axis=0)
        dirs = ev - center
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
        cos_fwd = dirs @ np.array([0.0, 0.0, 1.0])
        for i in range(len(ev)):
            gi = start + i
            if cos_fwd[i] > 0.85:
                pupil_set.add(gi)
                iris_set.add(gi)
            elif cos_fwd[i] > 0.55:
                iris_set.add(gi)
    return iris_set, pupil_set


# Module-level cache for iris/pupil sets


def _get_eye_sets(neutral_verts):
    """Lazy-init and cache iris/pupil/eyeball sets."""
    global _iris_set, _pupil_set, _eyeball_set
    if _iris_set is None:
        _iris_set, _pupil_set = _compute_iris_pupil(neutral_verts)
        _eyeball_set = set()
        for start, end in [LEFT_EYE_RANGE, RIGHT_EYE_RANGE]:
            _eyeball_set.update(range(start, end))
    return _iris_set, _pupil_set, _eyeball_set


def generate_flame_verts(flame, neck_aa, exp50, jaw_aa, shape100, eye_aa=None):
    """Run FLAME forward pass.

    Args:
        neck_aa: (N, 3) neck axis-angle
        exp50: (N, 50) expression codes
        jaw_aa: (N, 3) jaw axis-angle
        shape100: (100,) shape_mica
        eye_aa: (N, 3) eye axis-angle (left eye, right=copy). None = identity.
    Returns:
        verts: (N, 5023, 3) numpy
    """
    from steer.rotations import axis_angle_to_matrix, matrix_to_rotation_6d

    N = len(neck_aa)
    dev = torch.device('cuda')

    # Pad to FLAME dimensions
    shape_pad = torch.cat([
        torch.FloatTensor(shape100).unsqueeze(0).expand(N, -1),
        torch.zeros(N, 200)
    ], dim=-1).to(dev)
    exp_pad = torch.cat([
        torch.FloatTensor(exp50),
        torch.zeros(N, 50)
    ], dim=-1).to(dev)

    def aa_to_6d(aa_np):
        aa = torch.FloatTensor(aa_np).to(dev)
        return matrix_to_rotation_6d(axis_angle_to_matrix(aa))

    identity_6d = matrix_to_rotation_6d(axis_angle_to_matrix(
        torch.zeros(N, 3, device=dev)))
    neck_6d = aa_to_6d(neck_aa)
    jaw_6d = aa_to_6d(jaw_aa)

    if eye_aa is not None:
        eye_one_6d = aa_to_6d(eye_aa)
        eye_6d = torch.cat([eye_one_6d, eye_one_6d], dim=1)  # left + right (symmetric)
    else:
        eye_6d = torch.cat([identity_6d, identity_6d], dim=1)

    with torch.no_grad():
        out = flame(
            shape_params=shape_pad,
            expression_params=exp_pad,
            rot_params=identity_6d,
            neck_pose_params=neck_6d,
            jaw_pose_params=jaw_6d,
            eye_pose_params=eye_6d,
            trans_params=torch.zeros(N, 3, device=dev),
        )
    return out[0].cpu().numpy()


def smooth_non_eye(motion_59, sigma=1.5):
    """Smooth neck/exp/jaw but NOT eye dims."""
    out = motion_59.copy()
    out[:, :56] = gaussian_filter1d(motion_59[:, :56], sigma=sigma, axis=0)
    # eye dims 56:59 left unsmoothed
    return out.astype(np.float32)


def _load_smirk_eyelid():
    """Lazy-load SMIRK's l_eyelid / r_eyelid per-vertex blendshapes.

    Each is (5023, 3) float; final vertices = flame_verts + l * lid[:, 0:1] + r * lid[:, 1:2].
    Asset location: /CT/head_recordings3/work/smirk/assets/{l,r}_eyelid.npy
    """
    global _SMIRK_EYELID_BS
    if _SMIRK_EYELID_BS is None:
        import os
        base = os.path.join(os.path.dirname(__file__), 'flame', 'blendshapes')
        if not os.path.exists(os.path.join(base, 'l_eyelid.npy')):
            _SMIRK_EYELID_BS = (None, None)
        else:
            l = np.load(os.path.join(base, 'l_eyelid.npy')).astype(np.float32)
            r = np.load(os.path.join(base, 'r_eyelid.npy')).astype(np.float32)
            _SMIRK_EYELID_BS = (l, r)
    return _SMIRK_EYELID_BS


def motion59_to_verts(flame, motion_59, shape_mica, neck_override=None, eye_gain=1.0,
                      smooth=False):
    """Convert 59D motion to FLAME vertices.

    Args:
        neck_override: (N, 3) replacement neck AA. If set, use this instead of motion[:,53:56].
        eye_gain: multiply eye_aa by this factor (for amplified visualization).
        smooth: if True, apply gaussian_filter1d (sigma=1.5) on dims [0:56] before
                FLAME forward (legacy behavior — smooths exp+jaw+neck, leaves eyes
                untouched). Off by default — most callers feed already-smooth model
                output and applying this on top destroys lip dynamics.
    """
    if smooth:
        motion_59 = smooth_non_eye(motion_59)
    neck = neck_override if neck_override is not None else motion_59[:, 53:56]
    eye = motion_59[:, 56:59] * eye_gain
    verts = generate_flame_verts(
        flame,
        neck_aa=neck,
        exp50=motion_59[:, :50],
        jaw_aa=motion_59[:, 50:53],
        shape100=shape_mica,
        eye_aa=eye,
    )
    return verts


def motion61_to_verts(flame, motion_61, shape_mica, neck_override=None, eye_gain=1.0,
                      smooth=False):
    """Convert 61D motion (59D + eyelid2) to FLAME vertices with SMIRK eyelid
    blendshapes applied as additive vertex offsets (matches SMIRK's FLAME forward:
    verts = flame_verts + l_eyelid * lid[:, 0:1] + r_eyelid * lid[:, 1:2]).

    motion_61 layout: [0:50]=exp, [50:53]=jaw, [53:56]=neck, [56:59]=eye,
                      [59:61]=eyelid (SMIRK convention: [l_amount, r_amount]).

    smooth: forwarded to motion59_to_verts (off by default — see that docstring).
    """
    verts = motion59_to_verts(flame, motion_61[:, :59], shape_mica,
                              neck_override=neck_override, eye_gain=eye_gain,
                              smooth=smooth)
    l_bs, r_bs = _load_smirk_eyelid()
    if l_bs is None:
        return verts
    lid = motion_61[:, 59:61].astype(np.float32)                    # (N, 2)
    # verts shape: (N, 5023, 3). Broadcast blendshapes over frames.
    verts = verts + l_bs[None] * lid[:, 0:1, None] + r_bs[None] * lid[:, 1:2, None]
    return verts


def render_context_frames(verts_batch, faces, label_text="", size=CONTEXT_SZ,
                          pupil_indices=None, style='default'):
    """Render full-head context view with pupil centroid overlay.

    Args:
        style: 'default'  — current skin-tone on near-black background.
               'dualtalk' — paper-style grey head on white background,
                            matching DualTalk's MetallicRoughness material
                            and 5-light rig. Eyes still render (DualTalk
                            renders have no eyes; we keep ours for clarity).
    """
    import pyrender
    import trimesh

    renderer = pyrender.OffscreenRenderer(size, size)
    yfov = math.radians(18)
    camera = pyrender.PerspectiveCamera(yfov=yfov, znear=0.01, zfar=3.0)
    cam_pos = np.array([0.0, 0.025, 1.0])
    cam_z = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 1.0, 0.0])
    cam_x = np.cross(up, cam_z); cam_x /= np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)
    cp = np.eye(4)
    cp[:3, 0] = cam_x; cp[:3, 1] = cam_y; cp[:3, 2] = cam_z; cp[:3, 3] = cam_pos

    if style == 'dualtalk':
        material = pyrender.material.MetallicRoughnessMaterial(
            alphaMode='BLEND', baseColorFactor=[0.3, 0.3, 0.3, 1.0],
            metallicFactor=0.8, roughnessFactor=0.8)
    else:
        material = pyrender.material.MetallicRoughnessMaterial(
            alphaMode='BLEND', baseColorFactor=[0.7, 0.58, 0.5, 1.0],
            metallicFactor=0.15, roughnessFactor=0.55)

    iris_set, pupil_set, eyeball_set = _get_eye_sets(verts_batch[0])
    eye_face_mask = np.zeros(len(faces), dtype=bool)
    for fi, face in enumerate(faces):
        if any(int(v) in eyeball_set for v in face):
            eye_face_mask[fi] = True
    skin_faces = faces[~eye_face_mask]
    eye_faces = faces[eye_face_mask]

    # Precompute off-axis fill-light pose (rotate camera pose around world X-axis)
    fill_pose = cp.copy()
    fill_pose[:3, :3] = cv2.Rodrigues(np.array([0.25, -0.4, 0.0]))[0] @ cp[:3, :3]

    frames = []
    try:
        for vertices in verts_batch:
            if style == 'dualtalk':
                scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2],
                                       bg_color=[255, 255, 255, 255])
            else:
                scene = pyrender.Scene(ambient_light=[0.08, 0.08, 0.08],
                                       bg_color=[18, 18, 18, 255])
            skin_tri = trimesh.Trimesh(vertices=vertices, faces=skin_faces, process=False)
            scene.add(pyrender.Mesh.from_trimesh(skin_tri, material=material, smooth=True))
            n_v = len(vertices)
            vc = np.full((n_v, 4), 235, dtype=np.uint8)
            for idx in iris_set: vc[idx] = [60, 90, 140, 255]
            for idx in pupil_set: vc[idx] = [15, 15, 15, 255]
            eye_tri = trimesh.Trimesh(vertices=vertices, faces=eye_faces, vertex_colors=vc, process=False)
            scene.add(pyrender.Mesh.from_trimesh(eye_tri, smooth=False))
            scene.add(camera, pose=cp)
            if style == 'dualtalk':
                # DualTalk's light rig: its source code only rotates the light
                # TRANSLATION vector, not the rotation matrix, so all 5 lights
                # share the same world direction (camera +Z). Net effect is a
                # single directional light at 5× intensity — rock-stable, no
                # per-frame lighting recomputation, no specular flicker.
                scene.add(pyrender.DirectionalLight(color=[1., 1., 1.], intensity=10.0), pose=cp)
            else:
                # Default: key + dim off-axis fill.
                scene.add(pyrender.DirectionalLight(color=[1., 0.97, 0.92], intensity=5.0), pose=cp)
                scene.add(pyrender.DirectionalLight(color=[0.85, 0.9, 1.0], intensity=1.3), pose=fill_pose)
            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
            frame = color[..., ::-1].copy()
            if label_text:
                lbl_rgb = (40, 40, 40) if style == 'dualtalk' else (255, 255, 255)
                cv2.putText(frame, label_text, (8, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, lbl_rgb, 1, cv2.LINE_AA)
            frames.append(frame)
    finally:
        renderer.delete()
    return frames
