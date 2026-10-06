# Benchmarks

`benchmark_h3_attention.py` times attention candidates against forced BF16 Flash SDPA at MiniMax H3 shapes (D128) and reports numerical error. It runs synthetic tensors, not the full model.

```bash
python benchmark_h3_attention.py --candidate kitchen --sequences 32769 97751 --heads 56 --output kitchen.json
python benchmark_h3_attention.py --candidate kitchen-chunked ...   # rotated-K/V reuse with chunked queries
python benchmark_h3_attention.py --h3-module ../h3_memory/__init__.py --convrot-weights --stream-only ...
```

## Results (RTX 5090)

**Synthetic attention, full 56 heads, shape (1, 56, 97751, 128), 12 interleaved repetitions:**

| Path | Time | vs BF16 Flash |
|---|---:|---:|
| BF16 Flash (forced) | 1279.73 ms | 1.00× |
| Kitchen dense INT8 | 435.81 ms | **2.94×** |
| Kitchen dense INT8 + BF16 query-prefix repair | 577.76 ms | 2.22× |

Kitchen INT8 relative RMSE 1.715%. Peak allocated 7.71 GB for INT8 versus 5.63 GB for BF16. A full-head 32,769-token control gave 3.01×.

**Same shape, conditioning-protected sparse attention (includes prep/repair):** Kitchen safe Sol 325.10 ms, Kitchen dense INT8 438.09 ms.

**Whole-video results** are in [../docs/production-stack.md](../docs/production-stack.md). Synthetic attention speedups do not translate one-to-one into end-to-end time: attention is only part of each step.
