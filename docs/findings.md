# Findings

Results that surprised us, including experiments that didn't make it into the stack. The code for the failed experiments isn't in this repo; the measurements are. RTX 5090 throughout; synthetic micro-benchmarks are labelled as such.

## Round-to-nearest erases a LoRA baked into INT8

HyperFlow changes each weight by about 0.02%, far less than one INT8 step. The base weights already sit exactly on the INT8 grid, so dequantize → add LoRA → round-to-nearest snaps every weight back to where it started. You get the base checkpoint under a new name, with no error.

Stochastic rounding (what ComfyUI itself uses when it re-applies a LoRA at runtime) preserves the change in expectation. `tools/bake_lora_int8.py` measures the fraction of the intended delta that survives, per layer: **1.000 mean** with stochastic rounding.

## LoRAs cost far more than their VRAM under offloading

With layers offloaded, ComfyUI re-applies the LoRA to each offloaded layer on every step (dequantize, rank-256 matmul, requantize). Baking HyperFlow cut a 1088p Sol step from 94 s to ~76 s with `--lowvram` unchanged.

## SPEED at 0.5× loses framing at 768p, but low resolution alone doesn't

At 768p, automatic SPEED (0.5× first stage, 24×42 latent) pulled a close chest-up shot back to a wide full-body shot within a second, at both 6 and 8 steps. A plain 672×384 run on the same 24×42 grid held the framing. So the failure is SPEED's 0.5× stage plus its upsample/re-noise transition, not the small grid. A 0.71× first stage (34×60) holds framing for ~2.5 s more job time than 0.5×.

## Kitchen's INT8 attention is at the practical ceiling

The stock SM120 kernel runs at ~69 ms / ~630 dense TOPS for the representative 8192-query × 370k-key × 28-head call, with one CTA per SM (16.7% occupancy) and ~68% tensor-pipe activity.

- A 20-step ground-up warp-specialized rewrite (TMA two-stage K/V rings, twelve compute warps, released-on-read P packets, persistent scheduling) exceeded stock's tensor-pipe activity but was **7.2% slower** (78.0 vs 72.8 ms, paired).
- Forcing two resident CTAs doubled occupancy and ran **~1.8× slower** because of register spills. Smaller K tiles and less PV unrolling also regressed.
- All variants were byte-identical or within one BF16 ULP of stock on unit shapes.

Higher occupancy is not the lever here. This is an empirical result for these implementations and settings, not proof of a hardware limit.

## MLP glue isn't worth fusing

In the row-chunked ConvRot INT8 MLP at 370k rows (277 ms), non-GEMM kernels take 28 ms and all glue including copies 32 ms: **0.48%** of a layer's attention + MLP time.

## "Sol is slower on short shots" was an adapter artifact

An earlier adapter for NVIDIA's Sol made it ~3× slower than dense on short, reference-heavy shots. The cost came from its forced-dense repair of the conditioning prefix. Kitchen's in-kernel exact-KV sink has no such cost: −16% sampling time on the same kind of shot.

## A faster sparse baseline can be invalid

The old adapter's 255 s walk skipped conditioning protection on cached forwards. With protection on all 8 forwards, the correct time was 324 s. Check that every forward is protected before comparing times.

## Two nodes together can silently disable Sol

Zironic H3-Optimizations' Memory Optimization before 0.2.43 called `block.attn` directly, bypassing attention injected through `patches_replace`. Combined with core Sol, it silently ran full dense attention: no error, no log line. Fixed upstream in 0.2.43 (`ece65a4`).

## Sol inside VDN windows doesn't pay off

A rectangular/mapped Kitchen Sol kernel for VDN's local windows was correct (117 tests; cosine ≥ 0.99995 on real activations) but failed a 10% speed gate on a reference-heavy shot. Some windows were ~5% faster and others ~10% slower; with 256 augmentation tokens it was slower than dense on every window.

## Zironic sparse attention gives no gain over core Sol

Same shot and seed, attention node the only variable: no speed gain over core Kitchen Sol.

## `torch.compile` is slower here

Compiling the INT8 ConvRot model (after keeping Dynamo out of Kitchen's DLPack INT8 linear wrapper) was 7.8% slower in the sampler and 26.9% slower warm end to end, and changed output materially (608×352, 56 frames, 20 steps).
