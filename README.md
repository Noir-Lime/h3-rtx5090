# h3-rtx5090

Making MiniMax H3 video generation fast and memory-safe on **one 32 GB consumer GPU** (RTX 5090). This is a ComfyUI custom-node pack plus the patches, workflow, tools and measurements behind it.

Much of the fast path is other people's work that this stack combines and tunes. Their projects are credited and linked below, not copied. The code in this repo is the part we wrote.

## What's in here

| Path | What it is |
|---|---|
| `h3_memory/` | Memory/speed patches with one settings node (`H3 Memory Patch Settings`): row-chunked INT8 linear and MLP, batched VAE tile decode, opt-in rotated INT8 attention for very long sequences |
| `vdn_h3_fast/` | **VDN-H3 Fast**: an optimized VDN-H3 8-step stage as one `MODEL → MODEL` node (fused CuTe temporal op, Kitchen INT8 window attention, pruned-AdaLN support). Apache-2.0, see its `NOTICE` |
| `h3_helpers/` | Small nodes: V2V init (encode real frames into the AV latent for low-denoise refines) and spatial noise mask (per-pixel edit strength) |
| `patches/` | Changes to ComfyUI core that can't live in a node |
| `tools/bake_lora_int8.py` | Bakes a LoRA into an INT8 ConvRot checkpoint without erasing it |
| `workflows/` | The reference production graph as a ComfyScript file |
| `benchmarks/` | Attention benchmark harness and result summary |
| `docs/` | How the stack fits together, VDN-H3 Fast details, and findings (including what didn't work) |

## Getting started

1. **Have a working ComfyUI** with native MiniMax H3 support and Comfy Kitchen installed (recent ComfyUI releases include both). A 32 GB NVIDIA GPU is the target; smaller cards may work at lower resolutions.
2. **Install this pack** (below) and restart ComfyUI.
3. **Get the models.** Put these in the usual ComfyUI model folders. Weights are not distributed here; get them from their publishers.
   - `diffusion_models/`: an FL2VA INT8 ConvRot H3 checkpoint, ideally with HyperFlow baked in (step 4)
   - `text_encoders/`: `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
   - `vae/`: `minimax_h3_video_vae_int8_convrot.safetensors`, `minimax_h3_audio_vae_fp32.safetensors`
4. **Optional, recommended: bake HyperFlow** into the checkpoint so it isn't re-applied every step:
   ```bash
   <ComfyUI python> tools/bake_lora_int8.py BASE.safetensors HYPERFLOW_LORA.safetensors OUT.safetensors
   ```
   It prints how much of the LoRA survived (should be ~1.000).
5. **Install SPEED** ([ComfyUI-MiniMax-H3-SPEED](https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED)) if you want the full production graph. Noncommercial license.
6. **Run the reference workflow:** edit the prompt, image names and checkpoint name at the top of `workflows/hyperflow_sol_speed.py`, then:
   ```bash
   pip install "comfy-script[runtime]"
   python workflows/hyperflow_sol_speed.py --dry-run   # writes graph.json
   python workflows/hyperflow_sol_speed.py --queue     # runs it on http://127.0.0.1:8188/
   ```
   Or just add `H3 Memory Patch Settings` to your own graph between the model loader and the sampler.
7. **Optional core patches:** see [patches/README.md](patches/README.md).
8. **Read [docs/production-stack.md](docs/production-stack.md)** for why each setting is what it is, and [docs/findings.md](docs/findings.md) before trying things we already found don't work.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Noir-Lime/h3-rtx5090
<ComfyUI python> -m pip install -r h3-rtx5090/requirements.txt   # CuTe DSL, used by VDN-H3 Fast only
```

Restart ComfyUI. Each module registers independently, so a missing optional dependency only disables its own nodes.

**Note:** the memory patches install at import and apply process-wide. Their defaults are on (`int8_chunk_rows=32768`, `vae_tile_batch=56`) and rotated attention is off. Set any of them to `0` in the settings node to restore stock behaviour. Don't install this alongside an older copy of `h3_chunked_int8_linear`.

Requires a ComfyUI with native MiniMax H3 support and Comfy Kitchen (INT8 / ConvRot / Sol).

## The production stack

The reference graph is `workflows/hyperflow_sol_speed.py`. [docs/production-stack.md](docs/production-stack.md) explains each choice.

| Layer | Source | Measured effect on the RTX 5090 |
|---|---|---|
| FL2VA INT8 ConvRot checkpoint | MiniMax H3 + Comfy Kitchen ConvRot | INT8 dense attention is 2.94× BF16 Flash (synthetic attention benchmark, below) |
| HyperFlow 8-step, **baked** into the INT8 checkpoint | HyperFlow LoRA; bake via `tools/bake_lora_int8.py` | 1088p 12 s walk: 24:37 → 13:35 for the whole job (bake plus dropping `--lowvram`) |
| Kitchen INT8 Sol sparse attention | Comfy Kitchen (PR117) + core `BlockSparseAttention` | −16% sampling time on a reference-heavy 5 s shot |
| SPEED (low-res first steps) | ComfyUI-MiniMax-H3-SPEED | 768p job: 56.5 s → 43.4 s with the manual 0.71× schedule |
| Memory patches | this repo | Long clips and latent 2× refines fit in 32 GB; VAE decode 1.5× |

Single-run measurements on one machine (RTX 5090, mostly power-capped at 400 W). They're not cross-hardware guarantees.

## Credits

- **MiniMax H3**: the model.
- **[Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI)**: native H3 support and the `BlockSparseAttention` node (kijai).
- **[Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)**: INT8 / ConvRot kernels and INT8 Sol (PR 117).
- **HyperFlow**: the 8-step distillation LoRA baked by our tool. Weights are not redistributed.
- **[ComfyUI-MiniMax-H3-SPEED](https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED)** by StanLukuvka. **PolyForm Noncommercial 1.0.0:** check its terms before commercial use.
- **VDN-H3**: [OpenVDN/vdn-minimax-h3](https://github.com/OpenVDN/vdn-minimax-h3), ported by [Saganaki22/ComfyUI-VDN-H3](https://github.com/Saganaki22/ComfyUI-VDN-H3) and [xmarre/ComfyUI-VDN-H3-Plus](https://github.com/xmarre/ComfyUI-VDN-H3-Plus); pruned-AdaLN reinjection derives in part from [Larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo). VDN-H3 Fast is a modified derivative, see `vdn_h3_fast/NOTICE`.
- **[ComfyScript](https://github.com/Chaoses-Ib/ComfyScript)**: Python graph authoring.

No model weights or reference media are included.

## License

GPL-3.0 (see `LICENSE`), matching ComfyUI, whose code the patches modify. `vdn_h3_fast/` stays Apache-2.0 with its own `LICENSE` and `NOTICE`.
