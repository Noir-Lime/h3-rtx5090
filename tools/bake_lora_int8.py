"""Bake a LoRA into an INT8 ConvRot H3 checkpoint: dequantize (fp32), add B@A*alpha/rank, requantize with the same
ConvRot layout using stochastic rounding (the LoRA delta is far below one INT8 step, so round-to-nearest
would snap every weight back to the base grid, as the runtime's own requantize avoids). Streams tensor by tensor; untouched tensors are copied byte-for-byte.
Usage: bake_lora_int8.py BASE LORA OUT"""
import json
import struct
import sys
import torch
import zlib
from safetensors import safe_open
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

BASE, LORA, OUT = sys.argv[1:4]
dev = torch.device('cuda')


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


header, data_start = read_header(BASE)
lora = safe_open(LORA, 'pt', device='cuda')
base = safe_open(BASE, 'pt', device='cuda')
targets = {k[len('diffusion_model.'):-len('.alpha')] for k in lora.keys() if k.endswith('.alpha')}
assert all(f'{t}.weight' in header for t in targets), [t for t in targets if f'{t}.weight' not in header]

meta = dict(header.pop('__metadata__', {}) or {})
meta['baked_lora'] = LORA.rsplit('/', 1)[-1]
meta['baked_lora_strength'] = '1.0'
out_header = dict(header, __metadata__=meta)
blob = json.dumps(out_header, separators=(',', ':')).encode()
blob += b' ' * (-len(blob) % 8)

done, stats = {}, []


def baked(t):
    """Return {key: tensor} for layer t with the LoRA added."""
    a = lora.get_tensor(f'diffusion_model.{t}.lora_A.weight').float()
    b = lora.get_tensor(f'diffusion_model.{t}.lora_B.weight').float()
    scale = lora.get_tensor(f'diffusion_model.{t}.alpha').float().item() / a.shape[0]
    delta = (b @ a) * scale
    if header[f'{t}.weight']['dtype'] == 'I8':
        q, s = base.get_tensor(f'{t}.weight'), base.get_tensor(f'{t}.weight_scale')
        w = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(q, s, 256, 0).float()
        target = w + delta
        nq, params = TensorWiseINT8Layout.quantize(target, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256,
                                                    stochastic_rounding=zlib.crc32(t.encode()) + 1)
        ns = params.scale.reshape(s.shape).to(s.dtype)
        back = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(nq, ns, 256, 0).float()
        # how much of the LoRA delta survived: projection of the realized change onto the intended delta
        got = back - w
        stats.append((t, ((got * delta).sum() / (delta * delta).sum()).item(), (delta.norm() / w.norm()).item()))
        return {f'{t}.weight': nq.to(torch.int8), f'{t}.weight_scale': ns}
    w = base.get_tensor(f'{t}.weight')
    return {f'{t}.weight': (w.float() + delta).to(w.dtype)}


order = sorted(header, key=lambda k: header[k]['data_offsets'][0])
with open(BASE, 'rb') as src, open(OUT, 'wb') as dst:
    dst.write(struct.pack('<Q', len(blob)))
    dst.write(blob)
    for k in order:
        begin, end = header[k]['data_offsets']
        layer = k.rsplit('.', 1)[0]
        if layer in targets and k.endswith(('.weight', '.weight_scale')):
            if layer not in done:
                done[layer] = baked(layer)
            t = done[layer].pop(k).contiguous().cpu()
            raw = t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bfloat16 else t.view(torch.int16).numpy().tobytes()
            assert len(raw) == end - begin, (k, len(raw), end - begin)
            dst.write(raw)
            if not done[layer]:
                del done[layer]
        else:
            src.seek(data_start + begin)
            dst.write(src.read(end - begin))
assert not done, list(done)
print(f'baked {len(targets)} layers -> {OUT}')
kept = [e for _, e, _ in stats]
print(f'int8 layers {len(stats)}: delta kept (1.0 = exact) mean {sum(kept)/len(kept):.3f} min {min(kept):.3f} max {max(kept):.3f}; '
      f'lora delta/weight mean {sum(d for *_, d in stats)/len(stats):.4f}')
