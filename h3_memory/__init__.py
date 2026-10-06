"""MiniMax H3 memory and speed patches for ComfyUI (RTX 5090-class, 32 GB).

Installed at import, each read per call so the H3MemoryPatchSettings node can
change them per graph without a restart. Settings are process-global and sticky:
a graph without the node inherits whatever the last graph set.

1. Row-chunked TensorWiseINT8 linear (Comfy Kitchen INT8 / ConvRot weights).
   Reduces GEMM scratch. It still allocates the complete output tensor.
2. Row-chunked DiT MLP: never materializes the full fc1 expansion. Same math per
   row; not a promise of bit-identical kernels.
3. Rotated INT8 attention, OFF by default. One projection pass with compact Q
   caching, signed-H128 INT8 Q/K and INT8 V/P through Kitchen, 28-head groups,
   full context. Lossy. For long sequences (e.g. latent 2x refine) where dense
   Kitchen INT8 runs out of memory. Bypasses Sol for those calls.
4. Batched tiled VAE decode: decodes equal-size tiles in batches instead of one
   at a time. 0 restores the stock sequential decode.

Environment variables set the initial values:
  H3_INT8_CHUNK_ROWS          default 32768; 0 disables 1 and 2
  H3_ROTATED_INT8_CHUNK_ROWS  default 0 (off); otherwise a multiple of 128
  H3_VAE_TILE_BATCH           default 56 tiles/call (~9 GiB peak at 2688x1536); 0 = stock
"""
import math
import os

import torch

CHUNK_ROWS = int(os.environ.get("H3_INT8_CHUNK_ROWS", "32768"))
TAG = "[h3-rtx5090 memory]"

# ---- 1. row-chunked TensorWiseINT8 linear ------------------------------------------
try:
    from comfy_kitchen.tensor import base as _base
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout, _handle_int8_linear_tensorwise
except Exception as exc:  # pragma: no cover
    print(f"{TAG} comfy_kitchen not importable, linear patch skipped: {exc}")
    _base = None


def _chunked_handler(qt, args, kwargs):
    input_tensor = args[0]
    weight = args[1]
    if (CHUNK_ROWS <= 0 or not isinstance(input_tensor, torch.Tensor) or input_tensor.ndim < 2
            or not isinstance(weight, _base.QuantizedTensor)):
        return _handle_int8_linear_tensorwise(qt, args, kwargs)
    lead = input_tensor.shape[:-1]
    rows = math.prod(lead)
    if rows <= CHUNK_ROWS:
        return _handle_int8_linear_tensorwise(qt, args, kwargs)
    x2 = input_tensor.reshape(rows, input_tensor.shape[-1])
    out = None
    for start in range(0, rows, CHUNK_ROWS):
        piece = _handle_int8_linear_tensorwise(qt, (x2[start:start + CHUNK_ROWS],) + tuple(args[1:]), kwargs)
        if out is None:
            out = torch.empty((rows, piece.shape[-1]), dtype=piece.dtype, device=piece.device)
        out[start:start + CHUNK_ROWS] = piece
        del piece
    return out.reshape(tuple(lead) + (out.shape[-1],))


if _base is not None:  # always installed; CHUNK_ROWS is read per call (<=0 -> original handler)
    _base._LAYOUT_DISPATCH_TABLE.setdefault(torch.ops.aten.linear.default, {})[TensorWiseINT8Layout] = _chunked_handler
    print(f"{TAG} TensorWiseINT8 linear row-chunked at {CHUNK_ROWS} rows")

# ---- 2. row-chunked DiT MLP ----------------------------------------------------------
try:
    import comfy.ops
    import comfy.ldm.minimax.model as _h3

    _orig_mlp_forward = _h3.MLP.forward

    def _mlp_forward_chunked(self, x):
        rows = x.shape[0] if x.ndim == 2 else math.prod(x.shape[:-1])
        if CHUNK_ROWS <= 0 or rows <= CHUNK_ROWS:
            return _orig_mlp_forward(self, x)
        x2 = x.reshape(rows, x.shape[-1])
        out = torch.empty((rows, self.fc2.weight.shape[0]), dtype=x.dtype, device=x.device)
        for start in range(0, rows, CHUNK_ROWS):
            out[start:start + CHUNK_ROWS] = comfy.ops.linear_input_act(self.fc2, self.fc1(x2[start:start + CHUNK_ROWS]), "swiglu")
        return out.reshape(x.shape[:-1] + (out.shape[-1],))

    _h3.MLP.forward = _mlp_forward_chunked  # always installed; CHUNK_ROWS read per call
    print(f"{TAG} MiniMax H3 MLP row-chunked at {CHUNK_ROWS} rows")
except Exception as exc:  # pragma: no cover
    print(f"{TAG} MLP patch skipped: {exc}")

# ---- 3. EXPERIMENTAL rotated INT8 attention; disabled unless explicitly enabled -----
# This preserves the original ConvRot weight modules. Kitchen applies a separate
# signed H128 Q/K rotation; V/P use its INT8 representation. NOT lossless BF16.
# Compact cached Q avoids the second projection pass and full BF16 QKV storage.
# Full BF16 K/V still exists during prequantization: caller hidden states and weights
# must be included in any peak-memory claim. No live default is changed.
ROTATED_INT8_ROWS = int(os.environ.get("H3_ROTATED_INT8_CHUNK_ROWS", "0"))


def _profile_call(profile, name, function, *args, **kwargs):
    if profile is None:
        return function(*args, **kwargs)
    return profile(name, function, *args, **kwargs)


def _attend_head_groups(packed, group_heads=28, profile=None):
    """Keep each kernel's packed offsets/strides below INT32 at long sequences.

    Heads are independent. Rebuild the singleton batch stride after slicing,
    and slice packed V by its flattened [head * dim, padded_sequence] rows.
    No sequence is split and all queries retain every key/value token.
    """
    from dataclasses import replace
    from comfy_kitchen.sage_attention import int8_attention_from_prequantized
    batch, heads, length, dim = packed.q.shape
    if batch != 1 or packed.k.shape[1] != heads or packed.attn_mask is not None:
        raise ValueError("Head-group prototype requires H3 batch=1, equal heads and no mask")
    if group_heads < 1 or group_heads * dim * max(packed.k.shape[2], packed.v.shape[1], length) >= 2**31:
        raise ValueError("Head-group packed offsets would exceed signed INT32")
    out = torch.empty((batch, heads, length, dim), device=packed.q.device, dtype=packed.input_dtype)
    for start in range(0, heads, group_heads):
        stop = min(start + group_heads, heads)
        group = replace(packed,
            q=packed.q[0, start:stop].unsqueeze(0),
            k=packed.k[0, start:stop].unsqueeze(0),
            q_scale=packed.q_scale[0, start:stop].unsqueeze(0),
            k_scale=packed.k_scale[0, start:stop].unsqueeze(0),
            v=packed.v[start * dim:stop * dim],
            v_scale=packed.v_scale[start * dim:stop * dim])
        result = _profile_call(profile, "attention.kernel", int8_attention_from_prequantized, group)
        _profile_call(profile, "attention.copy", out[:, start:stop].copy_, result)
        del result
    return out


def rotated_int8_attention_forward(self, x, rope_freqs, rows=8192, profile=None):
    from dataclasses import replace
    from comfy_kitchen.sage_attention import prequantize_int8_attention

    if (torch.is_grad_enabled() or x.device.type != "cuda" or x.dtype != torch.bfloat16
            or self.head_dim != 128 or x.ndim != 2 or x.shape[0] <= 2048
            or rope_freqs is None or rows < 128 or rows % 128):
        raise ValueError("Rotated INT8 prototype requires inference/CUDA/BF16/D128, >2048 rows, RoPE and 128-aligned chunks")
    count, inner = x.shape[0], self.heads * self.head_dim
    qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
    kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)

    def project(start, stop, phase):
        # Keep weight casting/offload and ConvRot metadata on the original module.
        qkv = _profile_call(profile, phase + ".qkv_proj", self.qkv_proj, x[start:stop])
        q, k, _ = qkv.split(inner, dim=-1)
        q = q.view(1, stop - start, self.heads, self.head_dim)
        k = k.view(1, stop - start, self.heads, self.head_dim)
        _profile_call(profile, phase + ".rms_rope", comfy.quant_ops.ck.rms_rope_split_half_,
            q, k, rope_freqs[:, start:stop], qw, kw,
            epsilon=self.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
        return qkv

    # Global BF16 K/V allows the existing kernel to choose sequence-wide V
    # scales and key anchoring correctly, rather than inventing chunk scales.
    key = torch.empty((1, self.heads, count, self.head_dim), device=x.device, dtype=x.dtype)
    value = torch.empty_like(key)
    first_query = None
    # >1024 dummy keys select the same signed-H128/CTA path. Only the
    # compact Q bytes/scales survive; global K/V are still quantized together.
    dummy = torch.zeros((1, self.heads, 2048, self.head_dim), device=x.device, dtype=x.dtype)
    queries = []
    for start in range(0, count, rows):
        stop = min(start + rows, count)
        qkv = project(start, stop, "pass1")
        q, k, v = [part.view(stop - start, self.heads, self.head_dim).transpose(0, 1).unsqueeze(0)
                   for part in qkv.split(inner, dim=-1)]
        _profile_call(profile, "pass1.kv_copy", key[:, :, start:stop].copy_, k)
        _profile_call(profile, "pass1.kv_copy", value[:, :, start:stop].copy_, v)
        if start == 0:
            first_query = _profile_call(profile, "pass1.first_q_copy", q.contiguous)
            queries.append(None)
        else:
            query_pack = _profile_call(profile, "query_prequantize", prequantize_int8_attention, q, dummy, dummy)
            queries.append((query_pack.q, query_pack.q_scale))
            del query_pack
        del q, k, v, qkv
    packed = _profile_call(profile, "global_prequantize", prequantize_int8_attention, first_query, key, value)
    del key, value, first_query

    del dummy
    output = torch.empty_like(x)
    for start in range(0, count, rows):
        stop = min(start + rows, count)
        if start == 0:
            active = packed
        else:
            query, query_scale = queries[start // rows]
            active = replace(packed, q=query, q_scale=query_scale)
            queries[start // rows] = None
            del query, query_scale
        attended = _attend_head_groups(active, profile=profile)
        projected = _profile_call(profile, "out_proj", self.out_proj,
                                  attended.transpose(1, 2).reshape(stop - start, inner))
        _profile_call(profile, "output_copy", output[start:stop].copy_, projected)
        del projected
        del attended, active
    return output


def _check_rotated_rows(rows):
    if rows and (rows < 128 or rows % 128):
        raise ValueError("H3_ROTATED_INT8_CHUNK_ROWS must be 0 or a positive multiple of 128")
    return int(rows)


_check_rotated_rows(ROTATED_INT8_ROWS)
try:
    _orig_attention_forward = _h3.Attention.forward

    def _attention_forward_rotated(self, x, rope_freqs=None, transformer_options={}):
        # Leave token-refiner/small attention untouched. The opt-in long pass
        # explicitly uses Kitchen instead of any Sol attention override.
        # ROTATED_INT8_ROWS is read per call: 0 -> the original dense forward.
        if not ROTATED_INT8_ROWS or rope_freqs is None or x.shape[0] <= 2048:
            return _orig_attention_forward(self, x, rope_freqs, transformer_options)
        return rotated_int8_attention_forward(self, x, rope_freqs, ROTATED_INT8_ROWS)

    _h3.Attention.forward = _attention_forward_rotated
    if ROTATED_INT8_ROWS:
        print(f"{TAG} EXPERIMENTAL Kitchen signed-H128 INT8 attention, chunk={ROTATED_INT8_ROWS}; bypasses Sol for long RoPE attention")
    else:
        print(f"{TAG} rotated INT8 attention wrapper installed, OFF (settable per graph)")
except Exception as exc:  # pragma: no cover
    print(f"{TAG} attention wrapper skipped: {exc}")

# ---- 4. batched tiled VAE decode ---------------------------------------------------
VAE_TILE_BATCH = int(os.environ.get("H3_VAE_TILE_BATCH", "56"))


def _batched_tiled_decode(self, z):
    """Stock MiniMaxH3VideoVAE.tiled_decode with the per-chunk tiles decoded in batches."""
    if VAE_TILE_BATCH <= 0:  # read per call: 0 -> the stock sequential decode (bit-identical to unpatched)
        return _orig_tiled_decode(self, z)
    height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
    y_idx, y_len, y_overlap = self.split_tiles(height)
    x_idx, x_len, x_overlap = self.split_tiles(width)
    tiles = []
    for i_pos, i_len in zip(y_idx, y_len):
        zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
        for j_pos, j_len in zip(x_idx, x_len):
            zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio
            tiles.append(z[..., zi:zi + zl, zj:zj + zw])
    decoded = []
    for start in range(0, len(tiles), VAE_TILE_BATCH):
        batch = tiles[start:start + VAE_TILE_BATCH]
        if any(t.shape != batch[0].shape for t in batch):  # split_tiles gives equal tiles; never mix if not
            decoded.extend(self._decode_pixels(t) for t in batch)
        else:
            out = self._decode_pixels(torch.cat(batch, dim=0))
            decoded.extend(out[k:k + 1] for k in range(out.shape[0]))
    canvas = None; row_tails = []; out_y = 0; k = 0
    for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
        new_tails = []; left_tail = None; out_x = 0
        for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
            tile = decoded[k]; k += 1
            if i < len(y_idx) - 1:
                new_tails.append(tile[..., -y_overlap[i]:, :].clone())
            next_left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None
            if i > 0:
                tile = self.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
            left_tail = next_left_tail
            if i < len(y_idx) - 1:
                tile = tile[..., :-y_overlap[i], :]
            if j < len(x_idx) - 1:
                tile = tile[..., :, :-x_overlap[j]]
            if canvas is None:
                canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
            canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
            out_x += tile.shape[-1]
        row_tails = new_tails; out_y += tile.shape[-2]
    return canvas


try:
    import comfy.ldm.minimax.vae as _h3vae
    _orig_tiled_decode = _h3vae.MiniMaxH3VideoVAE.tiled_decode
    _h3vae.MiniMaxH3VideoVAE.tiled_decode = _batched_tiled_decode
    print(f"{TAG} H3 video VAE tiled decode batched, {VAE_TILE_BATCH} tiles per call (0 = stock; settable per graph)")
except Exception as exc:  # pragma: no cover
    print(f"{TAG} VAE batch patch skipped: {exc}")


# ---- 5. nodes: per-graph settings and the experiment hook ---------------------------
class H3MemoryPatchSettings:
    """Set this module's patch state for the current process. Model passthrough so it sits in
    the model chain and executes before the sampler; never cached (IS_CHANGED nan)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "rotated_attention": ("BOOLEAN", {"default": False, "tooltip": "Streamed rotated INT8 attention (H3_ROTATED_INT8_CHUNK_ROWS=rotated_rows). Lossy; use when dense CK INT8 OOMs."}),
            "vae_tile_batch": ("INT", {"default": 56, "min": 0, "max": 512, "tooltip": "Video VAE tiles per decoder call; 0 = stock sequential decode (bit-identical to unpatched)."}),
            "int8_chunk_rows": ("INT", {"default": 32768, "min": 0, "max": 1 << 22, "tooltip": "Row chunk for TensorWiseINT8 linear and the DiT MLP; 0 = unchunked."}),
        }, "optional": {
            "rotated_rows": ("INT", {"default": 8192, "min": 128, "max": 1 << 20, "step": 128}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/memory"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def apply(self, model, rotated_attention, vae_tile_batch, int8_chunk_rows, rotated_rows=8192):
        global ROTATED_INT8_ROWS, VAE_TILE_BATCH, CHUNK_ROWS
        ROTATED_INT8_ROWS = _check_rotated_rows(int(rotated_rows) if rotated_attention else 0)
        VAE_TILE_BATCH = int(vae_tile_batch)
        CHUNK_ROWS = int(int8_chunk_rows)
        info = f"rotated_rows={ROTATED_INT8_ROWS} vae_tile_batch={VAE_TILE_BATCH} int8_chunk_rows={CHUNK_ROWS}"
        print(f"{TAG} settings: {info}", flush=True)
        return (model, info)


NODE_CLASS_MAPPINGS = {
    "H3MemoryPatchSettings": H3MemoryPatchSettings,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MemoryPatchSettings": "H3 Memory Patch Settings",
}
