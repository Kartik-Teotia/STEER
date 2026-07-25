"""Checkpoint loading for the STEER prior + frozen causal motion AE."""
import torch

from models.causal_temporal_motion_ae import CausalTemporalMotionAE
from models.motion_prior_v6_dyadic_latent_arflow import MotionPriorV6LatentARFlow
from utils.common import NullableArgs


def load_steer(prior_ckpt, ae_ckpt, device="cuda"):
    """Load the STEER motion prior and its frozen motion AE.

    Returns (model, ae, args). Both are set to eval mode; AE gradients off.
    """
    ckpt = torch.load(prior_ckpt, map_location=device, weights_only=False)
    args = NullableArgs(ckpt["args"])

    ae_state = torch.load(ae_ckpt, map_location=device, weights_only=False)
    ae_args = ae_state.get("args", {}) or {}
    ae = CausalTemporalMotionAE(
        motion_dim=int(ae_args.get("motion_dim", 61)),
        hidden_dim=int(ae_args.get("encoder_hidden", 128)),
        latent_dim=int(ae_args.get("latent_dim", 256)),
        bottleneck_layers=int(ae_args.get("bottleneck_layers", 2)),
        bottleneck_heads=int(ae_args.get("bottleneck_heads", 4)),
        bottleneck_ff=int(ae_args.get("bottleneck_ff", 512)),
        use_kl=bool(ae_args.get("use_kl", True)),
        right_context_frames=int(getattr(args, "ae_decoder_right_context_frames", 0) or 0),
        encoder_pe_type=ae_args.get("encoder_pe_type", "rope"),
    ).to(device)
    ae.load_state_dict(ae_state["model"], strict=True)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    model = MotionPriorV6LatentARFlow(args, device=device)
    state = ckpt["model"]
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if not (k in model_state and v.shape != model_state[k].shape)}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    if getattr(model, "use_style_conditioning", False):
        model.set_style_reference(None)
    if getattr(model, "use_emotion_control", False):
        model.set_emotion_control(None)
    return model, ae, args
