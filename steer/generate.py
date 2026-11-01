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


@torch.inference_mode()
def _stride1_generate(model, ae, *, partner_motion_norm, self_audio, other_audio,
                      total_frames, cfg_scale, sample_steps, seed, device, overlap=2):
    """Stride-1 streaming generation: one latent token (4 frames) per forward pass.

    Starts from the model's learned start latent, with a noise tape shared across
    windows and a raised-cosine blend over `overlap` tokens. Partner gaze is zero.
    """
    n_per_token = 4
    total_tokens = total_frames // n_per_token
    n_latents, n_prev = model.n_latents, model.n_prev_latents
    n_window = n_latents * n_per_token
    torch.manual_seed(seed)  # padding noise at the end of the clip

    z_partner = ae.encode_mean(
        torch.from_numpy(partner_motion_norm[:total_tokens * n_per_token]).unsqueeze(0).to(device))
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise_tape = torch.randn(1, total_tokens, model.latent_dim, generator=g).to(device)
    z_out = torch.zeros(1, total_tokens, model.latent_dim, device=device)
    z_weight = torch.zeros(1, total_tokens, 1, device=device)
    gaze = torch.zeros(1, n_window, 3, device=device)
    taper = 0.5 - 0.5 * torch.cos(torch.linspace(0, float(np.pi), overlap, device=device))

    def audio_window(a, t):
        s = t * n_per_token * SPF
        chunk = a[s:s + n_window * SPF]
        chunk = np.pad(chunk, (0, n_window * SPF - len(chunk)))
        return torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)

    for t in range(total_tokens):
        end = min(t + n_latents, total_tokens)
        zp = z_partner[:, t:end]
        noise = noise_tape[:, t:end]
        if zp.shape[1] < n_latents:
            pad = n_latents - zp.shape[1]
            zp = torch.cat([zp, torch.zeros(1, pad, zp.shape[2], device=device)], dim=1)
            noise = torch.cat([noise, torch.randn(1, pad, model.latent_dim, device=device)], dim=1)
        if t >= n_prev:
            z_prev = z_out[:, t - n_prev:t]
        elif t > 0:
            z_prev = torch.cat([model.start_latent[:, -(n_prev - t):], z_out[:, :t]], dim=1)
        else:
            z_prev = model.start_latent
        z, _ = model.sample(
            z_prev=z_prev, z_partner=zp,
            self_audio_raw=audio_window(self_audio, t),
            other_audio_raw=audio_window(other_audio, t),
            partner_gaze_feat=gaze,
            behavior_control=None, behavior_mask=None,
            cfg_scale=cfg_scale, sample_steps=sample_steps, z_at_T=noise,
        )
        w = torch.ones(n_latents, device=device)
        if t > 0:
            w[:overlap] = taper
        for i in range(end - t):
            denom = z_weight[0, t + i] + w[i] + 1e-8
            z_out[0, t + i] = z_out[0, t + i] * (z_weight[0, t + i] / denom) + z[0, i] * (w[i] / denom)
            z_weight[0, t + i] += w[i]

    decoded = ae.decode(z_out)
    if getattr(model, "use_expression_contagion", False):
        pc = torch.from_numpy(partner_motion_norm[:total_tokens * n_per_token]).to(device).unsqueeze(0)
        decoded = model.apply_expression_contagion(decoded, other_motion=pc)
    return decoded[0].cpu().numpy()
