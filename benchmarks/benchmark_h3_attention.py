#!/usr/bin/env python3
"""Benchmark SageAttention 2 against Flash SDPA with MiniMax H3-like shapes."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def flash(q, k, v):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION], set_priority=True):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def sage(q, k, v):
    from sageattention import sageattn
    return sageattn(q, k, v, tensor_layout="HND", is_causal=False)


def kitchen(q, k, v):
    from comfy_kitchen.sage_attention import int8_attention
    return int8_attention(q, k, v)


def kitchen_chunked(q, k, v):
    """Prove reuse of rotated INT8 K/V with independently quantized query chunks.

    Synthetic test only: inputs here already exist in BF16. H3 must separately
    avoid assembling full BF16 QKV in its projection to reduce its memory peak.
    """
    from dataclasses import replace
    from comfy_kitchen.sage_attention import prequantize_int8_attention, int8_attention_from_prequantized
    if q.shape[-1] != 128 or k.shape[2] <= 2048:
        raise ValueError("This proof targets H3 D128 and long-sequence attention")
    chunk = 1024
    packed = prequantize_int8_attention(q, k, v)
    dummy = torch.zeros((q.shape[0], k.shape[1], 2048, 128), device=q.device, dtype=q.dtype)
    out = torch.empty_like(q)
    for start in range(0, q.shape[2], chunk):
        stop = min(start + chunk, q.shape[2])
        query = prequantize_int8_attention(q[:, :, start:stop], dummy, dummy)
        # Before accepting the dummy-K/V quantization route, require exactly the
        # same rotated query bytes/scales as a full-sequence quantization.
        if not torch.equal(query.q, packed.q[:, :, start:stop]):
            raise RuntimeError("Independent query quantization changed Q bytes")
        scale_start = (start // 128) * 32
        if not torch.equal(query.q_scale, packed.q_scale[:, :, scale_start:scale_start + query.q_scale.shape[-1]]):
            raise RuntimeError("Independent query quantization changed Q scales")
        out[:, :, start:stop] = int8_attention_from_prequantized(
            replace(packed, q=query.q, q_scale=query.q_scale))
    return out


def benchmark(function, q, k, v, warmup, iterations):
    for _ in range(warmup):
        function(q, k, v)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function(q, k, v)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def h3_block_proof(args):
    """Synthetic H3-width projection/RoPE proof, not trained-model quality."""
    import importlib.util
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import comfy.ldm.minimax.model as h3
    import comfy.quant_ops
    spec = importlib.util.spec_from_file_location("h3_memory_prototype", args.h3_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(30902026)
    attention = h3.Attention(5376, args.heads, 128, 1e-5,
                            dtype=torch.bfloat16, device="cuda", operations=torch.nn).eval().requires_grad_(False)
    if args.convrot_weights:
        from comfy_kitchen.tensor import QuantizedTensor
        for projection in (attention.qkv_proj, attention.out_proj):
            quantized_weight = QuantizedTensor.from_float(projection.weight.detach(),
                "TensorWiseINT8Layout", is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256)
            projection.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    results = {"gpu": torch.cuda.get_device_name(), "synthetic_weights": True,
               "convrot_weights": args.convrot_weights, "tests": []}
    for count in args.sequences:
        x = torch.randn((count, 5376), dtype=torch.bfloat16, device="cuda")
        rope = h3.rope_rotation_table(torch.randn((count, 96), device="cuda"), x.dtype)
        retained = [torch.zeros_like(x) for _ in range(args.retain_hidden_buffers)]

        def full(backend):
            qkv = attention.qkv_proj(x)
            q, k, v = qkv.split(args.heads * 128, dim=-1)
            q = q.view(1, count, args.heads, 128)
            k = k.view(1, count, args.heads, 128)
            comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope, attention.q_norm.weight,
                attention.k_norm.weight, epsilon=1e-5, rot_dim=96)
            attended = backend(q.transpose(1, 2), k.transpose(1, 2),
                               v.view(1, count, args.heads, 128).transpose(1, 2))
            return attention.out_proj(attended.transpose(1, 2).reshape(count, args.heads * 128))

        def measure(fn):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            stats = {"ms": start.elapsed_time(end), "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
                     "incremental_peak_GiB": (torch.cuda.max_memory_allocated() - baseline) / 2**30}
            if args.stream_only:
                stats["finite"] = all(bool(torch.isfinite(part).all()) for part in result.split(8192, dim=0))
                if not stats["finite"]:
                    raise RuntimeError("Nonfinite target-size output")
                del result
                return None, stats
            cpu = result.float().cpu()
            del result
            return cpu, stats

        if args.stream_only:
            _, stats = measure(lambda: module.rotated_int8_attention_forward(attention, x, rope, rows=args.chunk_rows))
            results["tests"].append({"tokens": count, "retained_hidden_buffers": args.retain_hidden_buffers,
                "chunk_rows": args.chunk_rows, "streamed_rotated_int8": stats,
                "scope": "synthetic attention only; excludes remaining model weights, MLP, sampler and VAE"})
            del x, rope, retained
            continue
        dense, dense_stats = measure(lambda: full(flash))
        quantized, quantized_stats = measure(lambda: full(kitchen))
        streamed, stream_stats = measure(lambda: module.rotated_int8_attention_forward(attention, x, rope, rows=args.chunk_rows))
        def errors(reference):
            delta = (streamed - reference).abs()
            if not torch.isfinite(streamed).all():
                raise RuntimeError("Nonfinite streamed H3 output")
            return {"mean_abs": delta.mean().item(), "median_abs": delta.median().item(),
                    "std_abs": delta.std().item(), "max_abs": delta.max().item(),
                    "relative_rmse": ((streamed-reference).square().mean() / reference.square().mean()).sqrt().item(),
                    "cosine": ((streamed.double() * reference.double()).sum() /
                        (streamed.double().square().sum() * reference.double().square().sum()).sqrt()).item()}
        entry = {"tokens": count, "vs_dense": errors(dense), "vs_full_rotated_int8": errors(quantized),
                 "dense": dense_stats, "full_rotated_int8": quantized_stats, "streamed_rotated_int8": stream_stats}
        # This gates only mechanical equivalence to the existing INT8 path.
        if entry["vs_full_rotated_int8"]["relative_rmse"] > 0.005:
            raise RuntimeError(f"Streamed path differs unexpectedly from full INT8: {entry}")
        results["tests"].append(entry)
        del x, rope, retained, dense, quantized, streamed
    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate", choices=["sage", "kitchen", "kitchen-chunked"], default="sage")
    parser.add_argument("--query-length", type=int, help="Use unequal Q/K lengths")
    parser.add_argument("--h3-module", type=Path, help="Test streamed H3 projection/RoPE module")
    parser.add_argument("--convrot-weights", action="store_true", help="Use synthetic ConvRot INT8 projection weights")
    parser.add_argument("--stream-only", action="store_true", help="Memory proof only; skip full-allocation reference")
    parser.add_argument("--retain-hidden-buffers", type=int, default=0)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    args = parser.parse_args()
    candidate_fn = {"sage": sage, "kitchen": kitchen, "kitchen-chunked": kitchen_chunked}[args.candidate]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.set_grad_enabled(False)
    if args.h3_module:
        h3_block_proof(args)
        return

    torch.manual_seed(30902026)
    results = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "candidate": args.candidate,
        "query_length": args.query_length,
        "batch": 1,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "shapes": [],
    }

    for sequence in args.sequences:
        shape = (1, args.heads, sequence, args.head_dim)
        q_shape = (1, args.heads, args.query_length or sequence, args.head_dim)
        q = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        # Compile/warm both paths before accuracy and timing comparisons.
        flash(q, k, v)
        candidate_fn(q, k, v)
        torch.cuda.synchronize()
        reference = flash(q, k, v).float()
        candidate = candidate_fn(q, k, v).float()
        if not torch.isfinite(candidate).all():
            raise RuntimeError("Non-finite attention output")
        delta = candidate - reference
        accuracy = {
            "mean_abs_error": delta.abs().mean().item(),
            "median_abs_error": delta.abs().median().item(),
            "std_abs_error": delta.abs().std().item(),
            "max_abs_error": delta.abs().max().item(),
            "rmse": delta.square().mean().sqrt().item(),
            "reference_rms": reference.square().mean().sqrt().item(),
            "cosine_similarity": torch.nn.functional.cosine_similarity(
                reference.flatten(), candidate.flatten(), dim=0
            ).item(),
        }
        del reference, candidate, delta

        # Alternate order to reduce ordering bias.
        if sequence == max(args.sequences):
            sage_result = benchmark(candidate_fn, q, k, v, args.warmup, args.iterations)
            flash_result = benchmark(flash, q, k, v, args.warmup, args.iterations)
        else:
            flash_result = benchmark(flash, q, k, v, args.warmup, args.iterations)
            sage_result = benchmark(candidate_fn, q, k, v, args.warmup, args.iterations)

        results["shapes"].append({
            "sequence": sequence,
            "shape": list(shape),
            "accuracy_vs_flash": accuracy,
            "results": {"flash": flash_result, args.candidate: sage_result},
            "candidate_speedup_x": flash_result["median_ms"] / sage_result["median_ms"],
            "candidate_faster_percent": (flash_result["median_ms"] / sage_result["median_ms"] - 1.0) * 100.0,
        })
        del q, k, v
        torch.cuda.empty_cache()

    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
