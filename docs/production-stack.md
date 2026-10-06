# The production stack, and why each setting

Reference graph: [`workflows/hyperflow_sol_speed.py`](../workflows/hyperflow_sol_speed.py). Machine: RTX 5090 32 GB, Ryzen 5800X3D, 64 GB RAM, PCIe 3.0 x16. Times are single runs unless stated.

## 1. Checkpoint: FL2VA, INT8 ConvRot, HyperFlow baked in

- Use the FL2VA checkpoint, not Ref2VA. Reference images and audio still work through `MiniMaxH3ReferenceToVideo`.
- **Bake HyperFlow instead of loading it as a LoRA.** Under memory pressure ComfyUI offloads layers, and for every offloaded layer it re-applies the LoRA on *every step*: dequantize INT8, add the rank-256 product, requantize. At ~190k tokens that repeats across most of the 50 blocks.
  - Same two Sol steps at 1088p: LoRA + `--lowvram` 94.0 s/step → baked + `--lowvram` 75.8–77.0 s → baked, no `--lowvram` 66.6–72.4 s.
  - 12 s 1088p walk, whole job: 24:37 → 13:35. Output visually near-identical to the LoRA run.
- Baking needs stochastic rounding. See [findings](findings.md#round-to-nearest-erases-a-lora-baked-into-int8).
- Run HyperFlow only at its official 8 Euler sigmas: `1.0, 0.9939097854, 0.9842874953, 0.9660637197, 0.9230769231, 0.8349423898, 0.6968520491, 0.4687533149, 0.0`. Don't stack other distillation or realism LoRAs on it.

## 2. Attention: Kitchen INT8 Sol through the core node

```
BlockSparseAttention(selection='sol-attn', tau=1.0, sink_conditioning='exact_kv_and_rows',
                     start_percent=0.2, min_tokens=12288, extra_tokens=256)
```

- `exact_kv_and_rows` keeps the reference/conditioning prefix as an exact sink inside the kernel. On a reference-heavy 5 s shot (38% of the sequence is conditioning) this gave −16% sampling time (−21% per sparse step) versus dense INT8.
- `start_percent=0.2`: the first steps stay dense.
- [`patches/0002`](../patches) raises the producer chunk from 4096 to 16384 rows. Used in production with ComfyUI running without `--lowvram`; its gain was not measured in isolation.

## 3. SPEED: fewer full-resolution steps

- **1088p:** automatic two-stage SPEED (0.5× first stage).
- **768p:** manual SPEED, 3 steps at 0.71× then 5 at 1.0×. 56.5 s → 43.4 s for the whole job.
- **Don't use 0.5× at 768p.** It loses the prompt's framing. See [findings](findings.md#speed-at-05-loses-framing-at-768p-but-low-resolution-alone-doesnt).
- Minimum useful resolution with this stack is about 700p.

## 4. Memory: `h3_memory`

| Setting | Default | What it does | Evidence |
|---|---|---|---|
| `int8_chunk_rows` | 32768 | Row-chunks INT8 linear and the DiT MLP so the full fc1 expansion is never materialized | Required for long clips at 1088p+ |
| `vae_tile_batch` | 56 | Decodes 256×256 video VAE tiles in batches of 56 | 1.5× decode on the INT8 VAE, sub-LSB differences; ~9 GiB peak at 2688×1536. Don't change tile size/overlap: larger tiles artifact |
| `rotated_attention` | off | Streamed signed-H128 INT8 attention with compact cached Q, for sequences where dense Kitchen INT8 runs out of memory | Latent 2× refine at 2688×1536: 240 frames in 488 s, 294 frames in 614 s. Synthetic 370k tokens: 28.84 GiB, 1.20× per forward vs the original rotated path. Lossy |

[`patches/0001`](../patches) frees the copied input embeddings before the transformer blocks. It saves one full hidden-state buffer, which long clips need.

## 5. VDN-H3 (alternative 8-step stage)

VDN-H3 is a different distillation with hybrid local/global attention. On this machine it's slower than HyperFlow/PDD + Sol, but if you want VDN, use VDN-H3 Fast. See [vdn_h3_fast.md](vdn_h3_fast.md).

- Kitchen INT8 window attention: 1.64× per step versus BF16 windows, with no measured or visible softening. Still 1.85× slower per step than PDD-8 + Sol on a 5 s clip.
- Fused CuTe temporal op + helpers: sampler 88.3 s → 82.6 s at 768×448, 362 frames. PDD + Sol on the same scene: ~58 s.
