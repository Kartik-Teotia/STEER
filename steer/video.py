"""Side-by-side video composition: [input partner video | STEER FLAME render]."""
import subprocess

import cv2
import numpy as np
from scipy.io import wavfile


def mix_audio(a, b):
    n = max(len(a), len(b))
    out = np.zeros(n, np.float32)
    out[: len(a)] += a
    out[: len(b)] += b
    m = np.abs(out).max()
    return out / m * 0.95 if m > 1.0 else out


def write_side_by_side(partner_frames, render_frames, target_audio, partner_audio,
                       out_mp4, fps=25, sr=16000, labels=("Input (partner)", "STEER")):
    n = min(len(partner_frames), len(render_frames))
    rows = []
    for i in range(n):
        left, right = partner_frames[i], render_frames[i]
        if left.shape[0] != right.shape[0]:
            left = cv2.resize(left, (right.shape[1], right.shape[0]))
        row = np.hstack([left, right])
        for ci, txt in enumerate(labels):
            x = ci * right.shape[1] + 10
            cv2.rectangle(row, (x - 6, 6), (x + 8 * len(txt) + 14, 30), (255, 255, 255), -1)
            cv2.putText(row, txt, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(row)

    h, w = rows[0].shape[:2]
    h += h % 2
    w += w % 2
    tmp_wav = str(out_mp4) + ".tmp.wav"
    mixed = mix_audio(target_audio.astype(np.float32), partner_audio.astype(np.float32))
    wavfile.write(tmp_wav, sr, (np.clip(mixed, -1, 1) * 32767).astype(np.int16))
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "pipe:0", "-i", tmp_wav,
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-c:a", "aac", "-shortest", str(out_mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for r in rows:
        if r.shape[0] != h or r.shape[1] != w:
            r = cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, w - r.shape[1],
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        proc.stdin.write(r.tobytes())
    proc.stdin.close()
    proc.wait()
    import os
    os.remove(tmp_wav)
