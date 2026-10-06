import torch
import torch.nn.functional as F

import comfy.nested_tensor


class MiniMaxH3V2VInit:
    """Initialize an H3 AV latent from real video frames (+ template audio) for a low-denoise refine pass."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "vae": ("VAE",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "init_video"
    CATEGORY = "MiniMax H3/V2V"
    DESCRIPTION = (
        "Encodes pixel frames with the H3 video VAE and writes them into the video half of a joint "
        "AV latent (audio half untouched). Feed the result to a sampler with truncated sigmas "
        "(denoise < 1) for a hires-fix style video-to-video refine."
    )

    def init_video(self, av_latent, vae, images):
        samples = av_latent.get("samples")
        if samples is None or not getattr(samples, "is_nested", False):
            raise ValueError("MiniMaxH3V2VInit requires a joint MiniMax H3 AV latent.")

        video_template, audio_latent = samples.unbind()[:2]
        encoded = vae.encode(images)
        if encoded.ndim == 4:
            encoded = encoded.unsqueeze(0)
        encoded = encoded.to(device=video_template.device, dtype=video_template.dtype)

        t_target = video_template.shape[2]
        if encoded.shape[2] > t_target:
            encoded = encoded[:, :, :t_target]
        elif encoded.shape[2] < t_target:
            encoded = F.pad(encoded, (0, 0, 0, 0, 0, t_target - encoded.shape[2]))
        if encoded.shape[1:] != video_template.shape[1:]:
            raise ValueError(
                f"Encoded video latent {tuple(encoded.shape)} does not match AV template "
                f"{tuple(video_template.shape)}; check resolution/length."
            )

        out = dict(av_latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((encoded, audio_latent))
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3V2VInit": MiniMaxH3V2VInit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3V2VInit": "MiniMax H3 V2V Init (video latent from frames)",
}
