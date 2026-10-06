# ComfyUI core patches

Changes to ComfyUI core that can't be done from a custom node. Generated against Comfy-Org/ComfyUI `b33e2b55`.

```bash
cd ComfyUI
git apply /path/to/h3-rtx5090/patches/0001-release-copied-h3-embeddings.patch
```

| Patch | File | Change | Evidence |
|---|---|---|---|
| `0001-release-copied-h3-embeddings` | `comfy/ldm/minimax/model.py` | Frees the video/audio embeddings once copied into the hidden state, before the transformer blocks | Saves one full hidden-state buffer on long clips; needed for long full-model runs with rotated attention |
| `0002-sparse-attention-16k-producer-chunk` | `comfy_extras/nodes_sparse_attention.py` | Sol producer chunk 4096 → 16384 rows | Used in production on 32 GB without `--lowvram`; not measured in isolation |

Both are intended as upstream pull requests. 0002 would be better upstream as a configurable option than as a new constant.
