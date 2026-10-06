import torch
import torch.nn.functional as F
import comfy.nested_tensor


class H3SpatialNoiseMask:
    """Set a spatial noise_mask on an H3 joint AV latent.

    1.0 = fully resample, 0.0 = keep the injected latent. Soft masks give a spatial
    gradient of edit strength. Audio stream is held (mask 0).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"av_latent": ("LATENT",), "mask": ("MASK",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Face Refine"

    def run(self, av_latent, mask):
        members = list(av_latent["samples"].unbind())
        video = members[0]
        h, w = video.shape[-2], video.shape[-1]
        m = mask[:1].reshape(1, 1, mask.shape[-2], mask.shape[-1]).float()
        m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
        vmask = m.view(1, 1, 1, h, w).expand(video.shape[0], video.shape[1], video.shape[2], h, w).contiguous()
        streams = [vmask.to(video.device, video.dtype)] + [torch.zeros_like(x) for x in members[1:]]
        out = dict(av_latent)
        out["noise_mask"] = comfy.nested_tensor.NestedTensor(tuple(streams))
        return (out,)


NODE_CLASS_MAPPINGS = {"H3SpatialNoiseMask": H3SpatialNoiseMask}
NODE_DISPLAY_NAME_MAPPINGS = {"H3SpatialNoiseMask": "H3 Spatial Noise Mask"}
