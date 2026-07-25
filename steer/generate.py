"""Chunked streaming generation for the STEER motion prior.

Defaults: cfg_scale 1.5, 10-step rectified-flow sampling, 3-latent clean prefix,
raised-cosine chunk stitching.
"""
import math
import numpy as np
import torch

SPF = 640           # audio samples per motion frame (16 kHz / 25 fps)


def _normalize(m, mean, std):
    return ((m - mean) / (std + 1e-8)).astype(np.float32)


def _denormalize(m, mean, std):
    return (m * (std + 1e-8) + mean).astype(np.float32)


def _chunked_generate(model, ae, *, partner_motion_norm, prev_motion_norm,
                      self_audio, other_audio, partner_gaze_feat_full,
                      n_frames_target, behavior_control, behavior_mask,
                      cfg_scale, behavior_cfg_scale, sample_steps,
                      seed, device,
                      n_motions=100, n_prev_motions=12, n_prev_latents=3,
                      chunk_overlap=20,
                      behavior_timeline=None,
                      emotion_control=None):
    """If `behavior_timeline=(ctrl_full, mask_full)` is given (each shape
    [total_latents, 10]), the per-chunk behavior_control is sliced from the
    timeline rather than reused as a single block. Lets us schedule e.g.
    'still → gaze_avert' across an arbitrary clip length without per-chunk
    envelope resets."""
    Tp = (partner_motion_norm.shape[0] // 4) * 4
    with torch.no_grad():
        z_partner_full = ae.encode_mean(
            torch.from_numpy(partner_motion_norm[:Tp]).unsqueeze(0).to(device))[0]
        z_prev = ae.encode_mean(
            torch.from_numpy(prev_motion_norm[:n_prev_motions]).unsqueeze(0).to(device))[0]
    if z_prev.shape[0] > n_prev_latents:
        z_prev = z_prev[-n_prev_latents:]

    stride = n_motions - chunk_overlap
    n_chunks = max(1, (n_frames_target - chunk_overlap) // stride)
    out = []
    for ci in range(n_chunks):
        fs = ci * stride; fe = fs + n_motions
        lat_s, lat_e = fs // 4, fs // 4 + n_motions // 4
        z_part = z_partner_full[lat_s:lat_e]
        if z_part.shape[0] < n_motions // 4:
            pad = n_motions // 4 - z_part.shape[0]
            z_part = torch.cat([z_part, z_part[-1:].expand(pad, -1)], dim=0)
        sa = self_audio[fs * SPF:fe * SPF]
        oa = other_audio[fs * SPF:fe * SPF]
        if len(sa) < n_motions * SPF: sa = np.pad(sa, (0, n_motions * SPF - len(sa)))
        if len(oa) < n_motions * SPF: oa = np.pad(oa, (0, n_motions * SPF - len(oa)))
        sa_t = torch.from_numpy(sa.astype(np.float32)).unsqueeze(0).to(device)
        oa_t = torch.from_numpy(oa.astype(np.float32)).unsqueeze(0).to(device)
        pg_chunk = partner_gaze_feat_full[fs:fe]
        if pg_chunk.shape[0] < n_motions:
            pg_chunk = np.pad(pg_chunk, ((0, n_motions - pg_chunk.shape[0]),(0,0)))
        pg_t = torch.from_numpy(pg_chunk.astype(np.float32)).unsqueeze(0).to(device)
        g = torch.Generator(device="cpu").manual_seed(seed + ci * 9973)
        noise = torch.randn(1, model.n_latents, model.latent_dim, generator=g).to(device)
        if getattr(model, "use_emotion_control", False):
            # If caller passed an emotion_control vector, set it for this chunk;
            # otherwise clear (preserves the previous-default of "no emotion").
            model.set_emotion_control(emotion_control)
        # Pick the behavior conditioning for this chunk: either reuse the same
        # block (default) or slice from the per-clip timeline.
        if behavior_timeline is not None:
            ctrl_full, mask_full = behavior_timeline
            n_lat_chunk = n_motions // 4
            le = lat_s + n_lat_chunk
            ctrl_slice = ctrl_full[lat_s:le]
            mask_slice = mask_full[lat_s:le]
            if ctrl_slice.shape[0] < n_lat_chunk:
                pad = n_lat_chunk - ctrl_slice.shape[0]
                ctrl_slice = torch.cat([ctrl_slice, ctrl_slice[-1:].expand(pad, -1)], dim=0)
                mask_slice = torch.cat([mask_slice, mask_slice[-1:].expand(pad, -1)], dim=0)
            bc_chunk = ctrl_slice.unsqueeze(0)
            bm_chunk = mask_slice.unsqueeze(0)
        else:
            bc_chunk = behavior_control
            bm_chunk = behavior_mask
        with torch.no_grad():
            z_pred, _ = model.sample(
                z_prev=z_prev.unsqueeze(0), z_partner=z_part.unsqueeze(0),
                self_audio_raw=sa_t, other_audio_raw=oa_t,
                partner_gaze_feat=pg_t,
                behavior_control=bc_chunk, behavior_mask=bm_chunk,
                cfg_scale=cfg_scale, behavior_cfg_scale=behavior_cfg_scale,
                sample_steps=sample_steps, z_at_T=noise,
            )
        z_pred = z_pred[0]
        decode_ctx = torch.cat([z_prev, z_pred], dim=0).unsqueeze(0)
        with torch.no_grad():
            motion_chunk = ae.decode(decode_ctx)[0]
        motion_chunk = motion_chunk[n_prev_latents * 4:]
        if getattr(model, "use_expression_contagion", False):
            pc = torch.from_numpy(partner_motion_norm[fs:fe]).to(device).unsqueeze(0)
            if pc.shape[1] < n_motions:
                pc = torch.cat([pc, pc[:, -1:].expand(-1, n_motions - pc.shape[1], -1)], dim=1)
            motion_chunk = model.apply_expression_contagion(
                motion_chunk.unsqueeze(0), other_motion=pc)[0]
        out.append(motion_chunk.detach().cpu().numpy())
        z_prev = z_pred[-n_prev_latents:]

    if chunk_overlap > 0 and n_chunks > 1:
        stitched = out[0].copy()
        for nxt in out[1:]:
            for f in range(chunk_overlap):
                alpha = 0.5 * (1.0 - math.cos(math.pi * f / max(1, chunk_overlap)))
                idx = stitched.shape[0] - chunk_overlap + f
                stitched[idx] = (1 - alpha) * stitched[idx] + alpha * nxt[f]
            stitched = np.concatenate([stitched, nxt[chunk_overlap:]], axis=0)
        return stitched
    return np.concatenate(out, axis=0)
