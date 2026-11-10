# Native integration handoff

## Identity and existing ownership

Target: the Comfy-Org repack of official MiniMax-H3-Image, released November 10, 2026. Pin the exact model revision and component bytes when native work is authorized.

Existing HyperGAN products inspected without modification:

- `p-h3-image-particle-sliders`: Pruna P-H3-Image. Its fixed-bank work, checkpoints, native dimensions, eight-step schedule, conditioning and result claims are not evidence for this release.
- `minimax-h3-particle-sliders`: H3 video/audio and Pika H3 Flex. Its nested audiovisual latent and audio sampling must not be reused for the new image-only model.
- `particle-sliders`: shared core. Dependency direction stays product → shared core → ParticleGAN. Do not copy the shared training loop or GAN losses into this product.

## Source-inspected behavior, not executed qualification

The source lock points to exact official metadata and ComfyUI files. The official transformer metadata says `audio_enabled=false` despite retaining an `audio_latents_dim` configuration value. ComfyUI constructs `MiniMaxH3ImageModel` with audio absent, uses a single image latent, and selects its image PDD sampler. Do not infer audio support from the unused configuration field.

The inspected attention has fused `qkv_proj` and `out_proj`, not independent Q/K/V Linear leaves. Candidate attachment locations are model block attention projections, but this version does not attach to them or claim quantized gradient support. The 56×128 attention inner width differs from the 5376 residual width; do not assume a square fused QKV mapping.

Native reference conditioning creates ordered Qwen vision inputs and optional VAE reference latents. Qwen references are scaled to a 1024-pixel short edge; VAE references use a 1024–2048 short-edge range at the inspected ComfyUI pin. The target latent is distinct from reference latents. The image node's default is 2048 square, while the official model card describes approximately 16 MP. Preserve this distinction in future tests.

The image final layer has a checkpoint-specific PDD head bank, FP32 output layers, and requires the sampler's sigma schedule. Head selection/combination is tied to intervals of the native grid. The final native response sign and timestep conventions must be preserved, not inferred from a generic flow model. This source-only product records the six-step grid but implements neither a substitute solver nor PDD math.

## Required work before native capability activation

1. Establish applicable model rights for the concrete user, execution location and use, including any hosted-service or revenue authorization. No weights have been fetched by this setup.
2. Lock the complete runtime and component identities. ComfyUI source inspection is not a qualified runtime lock. Keep the BF16, int8-convrot, W4A8 and text-encoder precision variants separate; no training support is inferred from repack availability.
3. Build an explicit product factory around verified native loading and cleanup. Freeze the backbone, conditioning and the exact paired expanded prompts. Preserve reference pixels/order, modality tags, packed positions and shape. Identify participating adapter/bank parameters exactly once.
4. Prove matched, unchanged base outputs at slider zero and initialization; establish gradients to every proposed trainable branch on actual native forward passes. Quantized offload/custom kernels may require different attachment/admission. No silent fallback to unrelated Linear or reduced-width responses.
5. Select the shared core's exact declared ParticleGAN response domain and profile. Admit real full output geometry before expensive allocation; do not call hidden responses images or label a toy scalar metric native qualification.
6. Implement train/resume/evaluate/export/inference through the shared lifecycle. Use RAW-only weights and preserve source/cohort identity in all packets. Validate strict restores and failure cleanup.
7. Preserve the native six-step PDD route, FP32 islands, sign and full image decode. Test neutral/positive paired prompts, multiple references, signed behavior only if intentionally supported, repeat requests, failures, restore and off/on comparison.
8. Run an explicitly authorized native campaign and independent visual checks before a useful-slider claim. CPU software checks are insufficient. Hosting and weight publication require their own authorization and license compliance.

No native execution, deployment, public service, model publication, broad owner reassignment, or migration of old results is requested by this scaffold.
