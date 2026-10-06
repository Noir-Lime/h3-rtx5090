"""Small MiniMax H3 helper nodes."""
from .spatial_noise_mask import NODE_CLASS_MAPPINGS as _mask, NODE_DISPLAY_NAME_MAPPINGS as _mask_names
from .v2v_init import NODE_CLASS_MAPPINGS as _v2v, NODE_DISPLAY_NAME_MAPPINGS as _v2v_names

NODE_CLASS_MAPPINGS = {**_mask, **_v2v}
NODE_DISPLAY_NAME_MAPPINGS = {**_mask_names, **_v2v_names}
