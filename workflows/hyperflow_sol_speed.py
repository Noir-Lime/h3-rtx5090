"""Reference MiniMax H3 production graph (ComfyScript, virtual mode).

HyperFlow-baked FL2VA INT8 ConvRot checkpoint + official 8-step Euler sigmas
+ core BlockSparseAttention (Kitchen INT8 Sol) + SPEED + this pack's memory
settings. Builds the graph and, with --queue, submits it to a running ComfyUI.

    pip install "comfy-script[runtime]"          # into any client environment
    python hyperflow_sol_speed.py --dry-run      # writes graph.json only
    python hyperflow_sol_speed.py --queue        # submits to http://127.0.0.1:8188/

Inputs (place in ComfyUI/input/): one frame-0 image, optional identity stills,
optional voice reference clips. Required custom nodes: this pack and
ComfyUI-MiniMax-H3-SPEED (PolyForm Noncommercial 1.0.0, check its license).
See ../docs/production-stack.md for why each setting is what it is.
"""
import json
import sys

from comfy_script.runtime import *

SERVER = "http://127.0.0.1:8188/"
PROMPT = "Describe the shot here. Dialogue goes inside <d></d> tags."
FRAME0 = "frame0.png"             # pinned first frame
REF_IMAGES = ["frame0.png"]       # identity / scene stills
REF_AUDIOS = []                   # voice reference clips, e.g. ["voice-0.wav"]
WIDTH, HEIGHT, LENGTH, SEED = 1344, 768, 192, 1

# Official HyperFlow 8-step schedule. Do not run HyperFlow at other step counts.
HYPERFLOW_SIGMAS = "1.0, 0.9939097854, 0.9842874953, 0.9660637197, 0.9230769231, 0.8349423898, 0.6968520491, 0.4687533149, 0.0"

load(SERVER)
from comfy_script.runtime.nodes import *

wf = Workflow(queue=False)
with wf:
    model = UNETLoader("minimax_h3_fl2va_pruned_int8_convrot_hyperflow8.safetensors", "default")
    clip = CLIPLoader("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax", "default")
    vae = VAELoader("minimax_h3_video_vae_int8_convrot.safetensors")
    audio_vae = VAELoader("minimax_h3_audio_vae_fp32.safetensors")
    model, _ = H3MemoryPatchSettings(model, rotated_attention=False, vae_tile_batch=56, int8_chunk_rows=32768)
    # Kitchen INT8 Sol. The conditioning prefix is an exact-KV sink inside the kernel.
    model = BlockSparseAttention(model, start_percent=0.2, end_percent=1.0, dense_blocks="",
                                 min_tokens=12288, extra_tokens=256,
                                 sink_conditioning="exact_kv_and_rows", verbose=True,
                                 **{"selection": "sol-attn", "selection.tau": 1.0})
    refs = {}
    for i, name in enumerate(REF_IMAGES):
        image, _ = LoadImage(name)
        refs[f"ref_images.ref_image_{i}"] = image
    for i, name in enumerate(REF_AUDIOS):
        refs[f"ref_audios.ref_audio_{i}"] = LoadAudio(name)
    positive, latent = MiniMaxH3ReferenceToVideo(clip=clip, vae=vae, audio_vae=audio_vae, prompt=PROMPT,
                                                 width=WIDTH, height=HEIGHT, length=LENGTH,
                                                 ref_image_size="max", **refs)
    pin, _ = LoadImage(FRAME0)
    positive = MiniMaxH3AddGuide(positive, latent, frame_idx=0, vae=vae, image=pin)
    # 768p: manual SPEED, 3 steps at 0.71x then 5 at full size. Automatic 0.5x at
    # 768p drops below the ~34x60 latent floor and loses prompt framing.
    # At 1088p use the automatic two-stage 0.5x mode instead.
    latent, _ = MiniMaxH3SPEEDSamplerManual(
        noise=RandomNoise(SEED), guider=BasicGuider(model, positive),
        sigmas=ManualSigmas(HYPERFLOW_SIGMAS), latent_image=latent,
        noise_policy="direct_coarse", seed_offset=10000, ratio_mode="steps",
        transition_goal_1=3, transition_resolution_1=0.71,
        transition_goal_2=8, transition_resolution_2=1.0,
        transition_goal_3=0, transition_resolution_3=0,
        transition_goal_4=0, transition_resolution_4=0, sampler_name="euler")
    video = CreateVideo(VAEDecode(latent, vae), 24.0, VAEDecodeAudio(latent, audio_vae), bit_depth=8)
    SaveVideo(video, "video/h3", format="mp4", codec={"codec": "h264"})

with open("graph.json", "w") as f:
    json.dump({"prompt": wf.api_format()}, f, indent=1)
print("wrote graph.json")
if "--queue" in sys.argv:
    task = wf.queue()
    print("queued", task.prompt_id)
    task.wait()
