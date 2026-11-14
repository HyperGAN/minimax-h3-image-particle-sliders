# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
| --- | ---: | --- | ---: | --- |
| Public source-only planner | 0 | 0 | 1 | No native weights; request planning only |
| Separate native FP4 training, short conditioning | 1 | 32 | 32 | B2; 2048²; text≤256; full52 blocks; checkpointed; FP4/BF16 |
| Separate native FP4 training, full text bound | 1 | 40 | 32 | B2; 2048²; text≤4096; checkpointed; FP4/BF16 |
| Separate native PDD6 sampling / teachers | 1 | 24 | 24 | B1; 2048²; text≤4096; FP4/BF16; TE/VAE absent |
| Separate native conditioning phase | 1 | 32 | 64 | One prompt; text≤4096; full Qwen3-VL32B FP4/BF16; no KV |
| Separate native VAE decode | 1 | 16 | 24 | B1; one2048² image; FP32; DiT already closed; untiled |


These are **calculated planning budgets, not measured peaks or hard runtime ceilings**.
They apply only to the bounded profiles in the table, for one process/slider at a
time. GiB means 2^30 bytes. Add concurrent jobs' resident allocations; do not sum
mutually exclusive loading, conditioning, training and decoding phases as if all
their models were resident together. Allow the stated host RAM independently of
VRAM. Larger batches, longer inputs, extra references, FP32 bases, different
attention kernels or allocator fragmentation can require more. There is no finite
universal maximum for arbitrary inputs. These calculations start no model jobs.

## Public scaffold versus separate native implementation

Public main currently contains a source-only planner, not a downloadable native
training runtime. Its original Comfy reference in [source-lock](src/minimax_h3_image_sliders/source-lock.json)
must not be confused with the separate November14 official-image implementation.
The nonzero native rows above describe that **separate, not yet shipped here**
implementation and are not executable features of this checkout.

Native model: `MiniMaxAI/MiniMax-H3-Image@5f90473023975e6271b0ed1dd84c4e88e4bbc6d3` with
Diffusers `43e30028af5120d8b0868cb6975b9fad39cc4fab`; image-only PDD6,
no audio stream, no CFG, default2048² canvas. Native Source is captured by path
and hashes in [inputs](docs/memory/inputs.json); relevant files are
`h3-image-five-20261114/product/src/minimax_h3_image_sliders/native_runtime.py`,
`native_adapter.py`, `native_pipeline.py`, `native_backend.py`, `native_cache.py`,
`native_encode.py` and `native_runner.py`. These paths describe the separate
campaign, not files shipped in this public scaffold.

## Calculation

* Exact header-derived model count: 33,105,770,752 BF16 +19,453,632 FP32
  elements = **61.737 GiB** native tensor bytes. The364 packable
  matrices contain 33,100,333,056 scalars. Frozen FP4 storage is
  `ceil(Q/2)+4*sum(ceil(each_matrix/64))+other_BF16+FP32_islands+364*64`
  = **17.423 GiB**. The last term covers per-module16-value FP32
  codebooks. No nested quantization is assumed. Input/time/PDD heads remainFP32;
  native compute and biases/norms use their declared precision. There is no
  persistent whole-model dequantization, but bnb backward can temporarily
  dequantize an individual large layer. Reserve1 GiB for this separately.
* Attention rank8: `52*4*8*(5376+7168)` = 20,873,216;208 routed bridges
  ×6476 +512 bank + width32 FieldCritic (339,329, condition5127) give
  **22,560,065 trainables**, with 1.681 GiB combined trainable/update allowance.
* Native field `[1,24,1,128,128]` has4096 image tokens; T=4096+text.
  B2 checkpoint boundaries cost `B*T*5376*2*53` =
  4.619 GiB for text256, 8.695 GiB
  at4096. The active-block envelope
  `2*B*T*(12*5376+4*7168+3*14336)` adds2.20/4.16 GiB respectively;
  include row512 FP32 branches, FF512 chunks, penalty graphs and live fields,
  rounding boundary+active totals to7/14 GiB. Add1 layer dequant +4 allocator/kernel
  workspace; round upward to32/40 GiB.
  These activation envelopes are approximations, not established runtime ceilings.
* Conditioning is separate: full Qwen3-VL32B is FP4-loaded and unloaded before
  DiT construction. `0.5625*33,357,390,064` = 17.475 GiB is a
  conservative all-parameter packing term; add6 GiB for retained BF16 embedding,
  output/vision islands, 2.54 GiB for65 hidden-state outputs at4096 tokens and
  5 GiB attention/loader workspace. Round to32. `use_cache=False`: no decoding
  KV cache is retained. Host64 covers the46.49 GiB largest original encoder
  shard/load residency plus frozen conditioning and runtime temporaries.
* Image VAE: exactly584,228,408 FP32 elements = 2.176 GiB.
  Untiled2048 decode reserves8 GiB live convolution fields (one2048²×128 FP32
  map alone is2 GiB), 1.5 transformer working and4 runtime GiB; round to16.
  DiT is closed before VAE load, so sampling uses max(24,16), not their sum.
* Teacher cache:48 rows ×3 full FP32 fields ×24×128×128×4 bytes =216 MiB,
  plus conditioning/metadata. Training host32 allows one6 GiB DiT shard plus
  loader copies, <1 GiB current concept/cache, rolling checkpoint snapshots,
  initialization/optimizer tensors and8 GiB runtime margin. PNGs/checkpoints on
  disk are not resident VRAM. Only rolling recovery plus0/256/512/768/1024
  immutable checkpoints are part of this profile.

No existing BF16 AV/Pika FULL, teacher cache or adapter is reused as image-model
state. The table covers one concept process; five serialized concepts use the
same peak budget, while concurrent workers add their independent allocations.
There is no FP32 full-DiT training promise or native1024 downscaling shortcut.


## Optimizer and estimate policy

Training figures freeze pretrained weights, use RAW current adapters and explicitly
disable weight EMA. They include no full-base gradients or EMA shadow model.
The PG 0.9.5 `bispf_optimized` recipe uses zero-momentum DualNorm, not Adam's two
full-size moment tensors. Its direction blend still retains protected gradients,
clones originals/displacements, and converts up to two protected normals to
FP64. For a conservative planning allowance, the tables reserve **80 bytes per
trainable scalar in total**: 8 for FP32 weights+gradients, 16 for two FP64 normals,
and 56 for protected FP32 gradients, FP32/FP64 direction/projection/polar,
rollback and serialization temporaries. This is an explicit allowance, not a claim
that every temporary is simultaneously allocated or that it is an exact peak.
Small scalar optimizer statistics remain enabled. Recipe changes need a new ledger.
See released [PG 0.9.5 recipes](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/recipes.py),
[DualNorm](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/optim/dualnorm.py)
and [direction blend](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/optim/direction_blend.py).

Activation entries below are architecture-based envelopes. Fused/memory-efficient
SDPA is assumed for the pretrained model; a math attention fallback that materializes
all score matrices can exceed them. No model benchmark or runtime measurement was
performed for this documentation. Machine-readable inputs, arithmetic and all table
rows are in [docs/memory/inputs.json](docs/memory/inputs.json).

## Explicit rounding ledger

| Operation | Calculated GPU subtotal (GiB) → planning budget | Host subtotal (GiB) → planning budget |
| --- | --- | --- |
| Public source-only planner | 0.000 → 0 | 1.000 → 1 |
| Separate native FP4 training, short conditioning | 31.104 → 32 | 31.000 → 32 |
| Separate native FP4 training, full text bound | 38.104 → 40 | 31.000 → 32 |
| Separate native PDD6 sampling / teachers | 23.423 → 24 | 21.000 → 24 |
| Separate native conditioning phase | 31.015 → 32 | 60.494 → 64 |
| Separate native VAE decode | 15.676 → 16 | 20.454 → 24 |

The detailed host terms and per-device arithmetic are included in the machine-readable inputs.
