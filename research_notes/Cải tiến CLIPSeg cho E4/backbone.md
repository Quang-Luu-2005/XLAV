# Backbone-side (image/text encoder) upgrades for CLIPSeg on PhraseCut (E4)

Scope note: the WebSearch/WebFetch tools failed in this session, so every source below was downloaded directly (arXiv PDFs, and the HF `modeling_clipseg.py` source) and read in full text. Papers I could not download are listed as gaps rather than quoted from memory. Numbers marked "inference" are my own estimates, not from a paper.

Baseline reference numbers (CLIPSeg paper, Table 3, PhraseCut test, t=0.3): CLIPSeg (PC+) mIoU 43.4 / IoU_FG 54.7 / AP 76.7; CLIPSeg (PC, D=64) 46.1 / 56.2 / 78.2; CLIPSeg (PC, D=128) 48.2 / 56.5 / 78.2; CLIP-Deconv 37.7 / 49.5 / 71.2; ViTSeg (PC, ImageNet ViT) 38.9 / 51.2 / 74.4; MDETR 53.7 mIoU; HulaNet 41.3 / 50.8 — [CLIPSeg, arXiv 2112.10003](https://arxiv.org/abs/2112.10003)

## Q1. Swapping CLIP ViT-B/16 for a stronger or more dense-aware backbone (ViT-L/14, OpenCLIP, EVA-CLIP, SigLIP/SigLIP2, MaskCLIP/CLIPSelf)

### Takeaway
Every paper that reports both sizes finds ViT-L/14 better than ViT-B/16 for dense or referring segmentation, by roughly +3 to +5 points. Dense-aware CLIP weights (CLIPSelf, SigLIP2) add about +1 to +4 more. None of these papers uses the CLIPSeg architecture or PhraseCut, though. In HF transformers, only an OpenAI/OpenCLIP-format ViT (same CLIP architecture) can be swapped in without rewriting the model. Even then, ViT-L/14 breaks the "refined" decoder's upsampler (patch 14), and it costs several times the compute, which is risky on a T4 within 2 weeks.

### Cited Findings
- The CLIPSeg authors used ViT-B/16 only because "the larger ViT-L/14@336px CLIP variant" weights "were not publicly released as of writing". They list not using ViT-L/14@336px as a limitation, so ViT-L is the backbone upgrade the authors themselves left open. — [CLIPSeg](https://arxiv.org/abs/2112.10003)
- CLIP pre-training matters a lot. Using the same architecture with an ImageNet ViT (ViTSeg) drops PhraseCut (PC) mIoU from 46.1 to 38.9. In the ablation, "no CLIP pre-training" collapses text-prompt mIoU to 13.1 / AP 12.6 (vs 43.6 / 76.7 for the full PC+ model, best-threshold protocol). — [CLIPSeg Tables 3, 9](https://arxiv.org/abs/2112.10003)
- **RISCLIP** (frozen CLIP + adapters + cross-modal modules, a referring segmentation model). RISCLIP-B (CLIP ViT-B/16) vs RISCLIP-L (CLIP ViT-L/14), oIoU:
  - RefCOCO val: 73.57 → 76.92 (+3.35)
  - RefCOCO+ val: 65.53 → 71.24 (+5.71)
  - RefCOCOg val: 64.10 → 67.96 (+3.86)

  The authors say ViT-L "advances the frontier set by RISCLIP-B". — [Kim et al., arXiv 2306.08498, Table 1](https://arxiv.org/abs/2306.08498)
- **SAN** (frozen CLIP + side adapter, open-vocab semantic segmentation), ViT-B/16 → ViT-L/14: ADE-150 mIoU 27.5 → 32.1; PASCAL Context-59 53.8 → 57.7. **CAT-Seg**, B → L: ADE-150 27.2 → 31.5; PC-59 57.5 → 62.0. (These are COCO-Stuff-trained open-vocab segmentation numbers, not referring segmentation.) — [CLIPSelf paper, Table 4 (arXiv 2310.01403)](https://arxiv.org/abs/2310.01403)
- **CLIPSelf** (self-distillation that aligns patch-level dense features with crop-level image embeddings, applied to EVA-CLIP ViTs). Plugged into CAT-Seg:
  - ViT-B/16: ADE-150 27.2 → 29.0; ADE-847 → 10.8; PC-59 57.5 → 58.0
  - ViT-L/14: ADE-150 31.5 → 34.5; PC-59 62.0 → 62.3

  The paper argues that a CLIP ViT's dense features "tend to encode the global image", which hurts local region recognition. — [Wu et al., CLIPSelf, ICLR 2024, arXiv 2310.01403](https://arxiv.org/abs/2310.01403)
- **SigLIP 2** frozen-feature dense probing (segmentation mIoU, PASCAL / ADE20k):
  - CLIP L/14 @224: 74.5 / 39.0
  - OpenCLIP G/14 @224: 71.4 / 39.3
  - SigLIP So/14 @224: 72.0 / 37.6
  - SigLIP 2 So/14 @224: 77.1 / 41.8
  - SigLIP So/14 @384: 73.8 / 40.8
  - SigLIP 2 So/14 @384: 78.1 / 45.4

  The SigLIP 2 paper also reports that, used inside CAT-Seg, SigLIP 2 L/16 beats SigLIP and even OpenCLIP G/14 on open-vocab segmentation. The extra self-distillation and masked-prediction losses are described as giving "better dense features". — [Tschannen et al., SigLIP 2, arXiv 2502.14786, Table 2 and Sec. 3.3](https://arxiv.org/abs/2502.14786)
- **MaskCLIP** (training-free). Modifying CLIP's last attention layer (using value embeddings directly), with key smoothing and prompt denoising, gives usable dense labels from a frozen CLIP. ViT-B/16 almost always beats CLIP-RN50, and MaskCLIP+ (ViT-B/16) reaches 31.1 mIoU in annotation-free segmentation (PASCAL Context). — [Zhou et al., arXiv 2112.01071](https://arxiv.org/abs/2112.01071)
- **CRIS** (CLIP-driven referring segmentation; fine-tunes CLIP ResNet encoders with a vision-language decoder): RefCOCO val oIoU 69.52 with CLIP-RN50 and 70.47 with CLIP-RN101. — [Wang et al., CRIS, arXiv 2111.15174](https://arxiv.org/abs/2111.15174)
- **HF implementation constraints**, read from `transformers/models/clipseg/modeling_clipseg.py`:
  - `CLIPSegModel.__init__` requires `config.vision_config` to be a `CLIPSegVisionConfig`, and builds `CLIPSegVisionTransformer` (same layout as the CLIP ViT).
  - The decoder's `reduces` are `nn.Linear(config.vision_config.hidden_size, config.reduce_dim)`, so the input dim follows the vision config automatically.
  - With `use_complex_transposed_convolution` (the "refined" checkpoint), the upsampler uses kernels `(patch_size // 4, patch_size // 4)` with matching strides. That gives ×16 upsampling for patch 16, but only ×9 for patch 14.
  - The decoder reshapes tokens with `size = int(sqrt(seq_len))` after dropping the CLS token at `output[:, 1:, :]`. A backbone without a CLS token (SigLIP) would therefore drop a real patch token and break the reshape.

  — [HF transformers modeling_clipseg.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py)

### Inferences
- **ViT-L/14 is the most-cited upgrade, but it is expensive here.** ViT-L/14 has about 3.5× the parameters of ViT-B/16 (approx. 304M vs 86M, from general knowledge, not verified this session). It also has 24 layers at width 1024.
  - Grid size: at 336px it produces 24×24 = 576 tokens, vs 22×22 = 484 at 352px with ViT-B/16. 352 is not a multiple of 14; use 336 or 350.
  - Compute: per-image forward FLOPs are likely about 4–5× the baseline. Since CLIP is frozen and run under `no_grad`, activation memory is modest, but throughput on a T4 would probably drop several-fold.
  - Turning the 6–10 h baseline run into about 20–40 h exceeds the ~30 GPU-h/week budget for a single run. **Flag: probably does NOT fit** unless iterations are cut, or features are cached (not possible with random augmentation).
- **Feasible HF route for ViT-L (moderate code, about 50–100 lines):**
  1. Build a `CLIPSegConfig` with ViT-L/14 text/vision sub-configs (hidden 1024, 24 layers, patch 14, image_size 224 or 336, projection_dim 768).
  2. Set `extract_layers` to roughly [7, 15, 19], i.e. the same relative depth as [3, 7, 9] / 12.
  3. Set `use_complex_transposed_convolution=False`, or add a final bilinear upsample.
  4. Load `openai/clip-vit-large-patch14(-336)` `CLIPModel` weights into `model.clip` with `load_state_dict(strict=False)`. Parameter names match since the CLIPSeg vision/text towers mirror CLIP.
  5. Train the decoder from scratch. The pretrained rd64 decoder weights do not transfer because input dims change.
- **Cheaper variant with the same architecture:** an OpenCLIP LAION-2B ViT-B/16 checkpoint in HF CLIP format (e.g. `laion/CLIP-ViT-B-16-laion2B-s34B-b88K`; existence not verified this session) could be dropped in with identical dims and compute. This is the cheapest backbone swap. I found no paper reporting its effect on referring segmentation, so the gain is unknown and could be ±1 point.
- **SigLIP/SigLIP2 and EVA-CLIP are not drop-in.** SigLIP has no CLS token and uses a MAP pooling head. EVA-CLIP uses a different block design (and is not native in transformers 4.55). Both need a custom wrapper replacing `self.clip` and patching the decoder's CLS handling, roughly 150–300 lines plus debugging. Too risky for the timeline.
- **CLIPSelf weights** are released for EVA-CLIP (per the paper's setup), so they inherit the EVA porting problem.
- **MaskCLIP-style tweak:** use the value path (`v`) of the last extracted layer instead of its output. It is training-free and cheap. However, CLIPSeg already reads intermediate layers [3, 7, 9] and not the final layer where MaskCLIP's issue is strongest, so the expected gain is small and unproven.

### Gaps
- No paper found that reports CLIPSeg itself with ViT-L/14, OpenCLIP, EVA-CLIP or SigLIP on PhraseCut. All backbone-size gains above come from different architectures and datasets (RefCOCO family, ADE/PC open-vocab).
- EVA-CLIP and OpenCLIP papers were not retrieved (web tools failed), so their dense-task numbers are not quoted.
- No measured T4 throughput for ViT-L/14 at 336px. The cost figures above are estimates.

## Q2. Partial unfreezing / parameter-efficient tuning of CLIP (LoRA, adapters, prompt tuning, last blocks)

### Takeaway
Tuning CLIP gives large in-domain gains: CAT-Seg +5 to +11 mIoU, RISCLIP's frozen decoder-only → adapters +14 mIoU on RefCOCOg. Full fine-tuning tends to hurt unseen and open-vocabulary generalization (ZegCLIP). Attention-only tuning (query and value projections), adapters, or deep prompt tuning keep most of the gain with lower risk.

In HF CLIPSeg the vision forward runs inside `torch.no_grad()`, so any tuning means re-implementing step 1 of `forward` (about 30–60 lines) and paying backward-pass memory through CLIP. That is feasible on a T4 for ViT-B/16 with LoRA, gradient checkpointing and a smaller batch, but it is the riskiest "one module" change for time.

### Cited Findings
- **CAT-Seg, Table 6** (ViT-B/16, trained on COCO-Stuff; columns A-847 / PC-459 / A-150 / PC-59 / PAS-20 / PAS-20b, plus learnable CLIP params and training memory):

  | Variant | A-847 | PC-459 | A-150 | PC-59 | PAS-20 | PAS-20b | Params | Memory |
  |---|---|---|---|---|---|---|---|---|
  | Freeze | 10.4 | 15.0 | 31.8 | 52.5 | 92.2 | 71.3 | 5.8M | 20.0 GiB |
  | Prompt tuning | 8.8 | 14.3 | 30.5 | 55.8 | 93.2 | 74.7 | — | — |
  | Full fine-tune | 13.6 | 22.2 | 34.0 | 61.1 | 97.3 | 79.7 | 393.2M | 26.8 GiB |
  | Attention-only | 15.7 | 23.7 | 37.1 | 63.1 | 97.1 | 81.5 | 134.9M | 20.9 GiB |
  | QV both encoders (theirs) | 16.0 | 23.8 | 37.9 | 63.3 | 97.0 | 82.5 | 70.3M | 20.9 GiB |
  | QV image encoder only | 13.9 | 22.8 | 35.1 | 62.0 | 96.3 | 82.0 | 56.7M | — |

  — [Cho et al., CAT-Seg, arXiv 2303.11797, Table 6](https://arxiv.org/abs/2303.11797)
- **CAT-Seg, Table 3:** with a simple feature-aggregation decoder (closest to CLIPSeg's design), fine-tuning CLIP vs freezing raises A-150 from 16.6 to 23.6 and PC-59 from 46.8 to 58.1. — [CAT-Seg Table 3](https://arxiv.org/abs/2303.11797)
- **ZegCLIP** (VOC zero-shot, mIoU seen / unseen):
  - Baseline-Fix: 71.1 / 16.3
  - Baseline-FT (full fine-tuning): 76.5 / 13.8, better on seen classes but worse on unseen
  - Baseline-DPT (deep prompt tuning): 75.9 / 28.3
  - With all their designs: FT+NEL+RD 90.2 / 42.4 vs DPT+NEL+RD (ZegCLIP) 91.9 / 77.8

  The authors conclude "fine-tuning could lead to overfitting to seen classes while prompt tuning prefers to retain the inherent zero-shot" ability. — [Zhou et al., ZegCLIP, arXiv 2212.03588, Table 5](https://arxiv.org/abs/2212.03588)
- **RISCLIP-B ablation** (RefCOCOg-UMD test, 240px quick setting, mIoU / oIoU):
  - Frozen CLIP: 23.86 / 33.13
  - Frozen + Adapters: 48.29 / 50.98
  - Full RISCLIP with frozen CLIP + adapters: 62.64 / 62.02
  - Same with CLIP fully fine-tuned: 57.88 / 55.75 (lower than keeping CLIP frozen with adapters)

  Adapters on only the last N layers (mIoU): N=3 → 61.40, N=6 → 62.15, N=9 → 62.31, N=12 → 62.64. — [RISCLIP, arXiv 2306.08498, Tables 3 and 5](https://arxiv.org/abs/2306.08498)
- **ETRIS** (Bridger adapters between frozen CLIP encoders) updates "only 1.61% to 3.38%" of backbone parameters. It reaches RefCOCO val oIoU 70.51 with CLIP-B, vs CRIS 70.47 with a fine-tuned CLIP-RN101. — [Xu et al., ETRIS, arXiv 2307.11545](https://arxiv.org/abs/2307.11545); numbers as tabulated in [RISCLIP Table 1](https://arxiv.org/abs/2306.08498)
- **CRIS** shows "naive" direct fine-tuning of CLIP is worse than their contrastive text-to-pixel alignment design (Fig. 2). — [CRIS](https://arxiv.org/abs/2111.15174)
- **HF code fact:** `CLIPSegForImageSegmentation.forward` wraps `self.clip.vision_model(...)` and `visual_projection` in `with torch.no_grad():`, and `get_conditional_embeddings` also computes text features under `torch.no_grad()`. Setting `requires_grad=True` or attaching LoRA alone therefore has no effect. — [HF modeling_clipseg.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py)

### Inferences
- **Best fit for E4, if tuning CLIP:** LoRA (rank 4–16) on q_proj/v_proj of the vision tower, in the last ~4–6 blocks only, or in all blocks up to layer 9. Layers 10–11 are never read by the decoder (extract_layers stops at 9), so tuning them is wasted compute.
  - Tooling: PEFT `LoraConfig(target_modules=["q_proj","v_proj"])` works on `CLIPSegAttention` since it uses `q_proj`/`k_proj`/`v_proj`. Then subclass or monkey-patch `forward` to drop the `no_grad`.
  - Size: about 40–80 lines.
  - Risk: VRAM rises because of backward through 10 blocks at 485 tokens. Use AMP plus `gradient_checkpointing_enable()` (CLIPSeg layers subclass `GradientCheckpointingLayer`) and likely batch 32 with grad accumulation.
  - Keep a lower LR for LoRA (e.g. 1e-4) than for the decoder (1e-3).
- **Keep the text tower frozen.** Text embeddings are the FiLM conditioning, and the visual-prompt branch (PC+) relies on image and text embeddings sharing CLIP's space. Tuning either tower can break the text/visual interchangeability that CLIPSeg's PC+ training depends on.
- **Generalization risk:** given the ZegCLIP and RISCLIP evidence, report both PhraseCut and an out-of-distribution check (e.g. CLIPSeg's generalized affordance/attribute prompts, or the PASCAL zero-shot split) to show that open-vocabulary ability is not lost.

### Gaps
- No paper found that applies LoRA or partial unfreezing to CLIPSeg itself, and none that reports PhraseCut numbers for such tuning.
- No measured T4 memory or throughput for CLIPSeg with LoRA. Feasibility is estimated.

## Q3. Which CLIP layers to extract, how many, and input resolution / positional-embedding interpolation

### Takeaway
The only primary CLIPSeg evidence:
- Using only layer 3 drops text-prompt PhraseCut mIoU from 43.6 to 31.9, so higher-level features matter.
- Decoder width matters: D=16 gives 37.4, and D=128 beats D=64 on PC.
- ViT-B/16 CLIP accuracy peaks around 350px and degrades above it.

No published ablation of alternatives like [3,7,9] vs [6,9,12] or 4–6 layers was found. Changing `extract_layers` is a one-line config change in HF, plus decoder retraining, so it is the cheapest E4 option. The upside is unproven but plausibly small to moderate.

### Cited Findings
- **CLIPSeg Table 9** (PhraseCut, best threshold; text mIoU / AP; visual mIoU / AP):

  | Variant | Text mIoU | Text AP | Visual mIoU | Visual AP |
  |---|---|---|---|---|
  | CLIPSeg PC+ | 43.6 | 76.7 | 25.4 | 55.6 |
  | No CLIP pre-training | 13.1 | 12.6 | 12.7 | — |
  | No visual | 46.4 | 77.8 | 14.4 | 31.0 |
  | D=16 | 37.4 | 71.5 | 24.7 | 51.2 |
  | Only layer 3 | 31.9 | 64.9 | 21.5 | 48.6 |
  | Highlight mask | 43.4 | 75.4 | 23.3 | 43.8 |

  (Row-to-value mapping reconstructed from the PDF's two-column layout. It is consistent with the text, e.g. "no visual" raises text and lowers visual scores.) — [CLIPSeg Table 9](https://arxiv.org/abs/2112.10003)
- The default is S = [3, 7, 9], "consequently our decoder has only three layers". The decoder has 1,122,305 trainable params at D=64, and "a high capacity decoder (D = 128) seems to be beneficial for PhraseCut" (PC mIoU 46.1 → 48.2). — [CLIPSeg Sec. 3, 5.1](https://arxiv.org/abs/2112.10003)
- **Resolution:** CLIPSeg interpolates positional embeddings to allow other image sizes. The authors "find that for ViT-B/16 performance only decreases for images larger than 350 pixels". In the appendix, CLS-token linear-probe accuracy on a 67-class ImageNet subset peaks near 350×350 for ViT-B/16. Training used 352×352, 20k iterations, batch 64, LR 1e-3 cosine to 1e-4, AMP, BCE. — [CLIPSeg Sec. 3 and Appendix](https://arxiv.org/abs/2112.10003)
- **RISCLIP layer-count ablation** (cross-modal modules on the last N layers, RefCOCOg mIoU): CFE N=2 → 62.33, N=4 → 62.41, N=6 → 62.64. Adding adapters to more of the later layers helps only marginally (61.40 at N=3 → 62.64 at N=12). — [RISCLIP Table 5](https://arxiv.org/abs/2306.08498)
- **HF code:** `activations = [hidden_states[i + 1] for i in self.extract_layers]`. The decoder builds `len(config.extract_layers)` reduce layers and decoder blocks, and applies FiLM at `config.conditional_layer` (default 0, i.e. the deepest extracted layer, since activations are reversed). `interpolate_pos_encoding=True` is the default in the CLIPSeg vision forward. — [HF modeling_clipseg.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py)

### Inferences
- **Cheap E4 candidates, each a 1–5 line change; same CLIP cost, decoder grows slightly, one training run each:**
  - (a) `extract_layers=[3,6,9,11]` or `[2,5,8,11]`. Uses the deeper layers that are currently computed but discarded, and adds one decoder block.
  - (b) `[3,7,9]` + `reduce_dim=128`. The paper's own D=128 finding.
  - (c) Change `conditional_layer` to apply FiLM at a different depth.

  Option (a) changes the module count, so the pretrained rd64 decoder cannot be loaded; train from scratch, which the E1 pipeline presumably already does. Note that in CLIP ViT-B/16, layers 10–11 are already computed in the frozen forward (`output_hidden_states=True` runs all 12), so extracting them adds no backbone cost.
- **Higher input resolution** (e.g. 448–480px) at ViT-B/16 gives 784–900 tokens vs 484. That roughly doubles attention cost in both CLIP and decoder. The CLIPSeg appendix suggests CLIP-B/16 semantics degrade above ~350px, so the gain is uncertain. Also, PhraseCut evaluation would need resizing logic, and the "refined" upsampler is fixed ×16, which is fine. Medium feasibility, low confidence.

### Gaps
- No ablation found of different multi-layer sets for CLIPSeg (e.g. [3,7,9] vs [4,8,12] or 5 layers). The GitHub issues of timojl/clipseg could not be retrieved (the raw Readme fetch returned only 14 bytes).
- The CLIPSeg supplementary resolution study measures classification, not segmentation. No PhraseCut-at-higher-resolution numbers were found.

## Q4. Follow-up works that cite CLIPSeg and modify the backbone; PhraseCut / RefCOCO / PASCAL-5i numbers

### Takeaway
I found no follow-up that keeps the CLIPSeg decoder and only upgrades its backbone while reporting PhraseCut. The relevant CLIP-based referring segmentation works (CRIS, ETRIS, RISCLIP) report on the RefCOCO family, where CLIP-B-based methods reach about 70–74 RefCOCO val oIoU and CLIP-L reaches about 77. These are not directly comparable to CLIPSeg's PhraseCut mIoU.

### Cited Findings
- RefCOCO val oIoU of CLIP-based referring segmentation models (different datasets and protocols from PhraseCut):
  - CRIS: 69.52 (CLIP-RN50), 70.47 (CLIP-RN101) — [CRIS](https://arxiv.org/abs/2111.15174)
  - ETRIS (CLIP-B, frozen + Bridger): 70.51
  - RISCLIP-B: 73.57; RISCLIP-L: 76.92
  - Non-CLIP reference: LAVT (Swin-B + BERT) 72.73
  - — [RISCLIP Table 1](https://arxiv.org/abs/2306.08498)
- PhraseCut state reported in the CLIPSeg paper: MDETR 53.7 mIoU (full resolution, two rounds of fine-tuning) vs CLIPSeg (PC, D=128) 48.2. MDETR remains the stronger PhraseCut reference cited there. — [CLIPSeg Table 3](https://arxiv.org/abs/2112.10003)

### Inferences
- For the E4 write-up, cite RISCLIP's B → L (+3.4 RefCOCO val oIoU) and CAT-Seg/SAN's B → L (+4 to +4.6 ADE-150 mIoU) as indirect evidence. Cite CAT-Seg QV-tuning and RISCLIP adapters as the evidence for tuning. State clearly that none of it was measured on PhraseCut with the CLIPSeg decoder.

### Gaps
- PASCAL-5i numbers for backbone-modified CLIPSeg variants: none found.
- Papers with Code PhraseCut leaderboard: not retrieved (web tools failed).
- Other CLIPSeg-citing works that modify the backbone (e.g. medical CLIPSeg variants, CLIPSeg + SAM pipelines): not surveyed.

## Q5. Implementability in HF transformers 4.55.4 and code size per option (summary)

### Takeaway
Ranked by effort for one person in 2 weeks on a T4:
1. **Layer-set / D change** — config only.
2. **LoRA or adapters on the vision tower** — forward override plus PEFT, about 40–80 lines; moderate VRAM risk.
3. **Same-architecture OpenCLIP ViT-B/16 swap** — weight remapping, about 20–40 lines; same compute.
4. **CLIP ViT-L/14 swap** — config plus weight load plus upsampler fix, about 50–100 lines; likely exceeds the GPU budget.
5. **SigLIP/SigLIP2, EVA-CLIP, CLIPSelf** — replace `self.clip` and patch CLS handling, 150–300+ lines; not recommended.

### Cited Findings
- The HF CLIPSeg code exposes `extract_layers`, `reduce_dim`, `conditional_layer`, `use_complex_transposed_convolution` and the vision/text sub-configs. The decoder input dims are derived from `config.vision_config.hidden_size`. — [HF modeling_clipseg.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py)
- In the code: the CLIP vision forward and the text-feature computation are inside `torch.no_grad()`; the decoder strips index 0 as CLS (`output[:, 1:, :]`); and the "refined" upsampler uses `patch_size // 4` kernels and strides. — [HF modeling_clipseg.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clipseg/modeling_clipseg.py)

### Inferences
- The source inspected was the current `main` `modeling_clipseg.py`. The locally installed transformers is 5.16.1, not 4.55.4. The structures quoted (`no_grad` blocks, `extract_layers` indexing, decoder construction) have been stable across 4.x, but verify against 4.55.4 before coding.
- **Recommendation for E4 under the budget:** first try the zero-backbone-cost options. For example, extract [3,6,9,11] or 4 layers, with D=128 as a decoder-side alternative if allowed. If a true backbone change is required, LoRA-QV on vision blocks ≤9 is the option with the strongest literature support (CAT-Seg, RISCLIP) that still fits a T4. ViT-L/14 has the clearest published gain but likely does not fit 30 GPU-h/week.

### Gaps
- Exact 4.55.4 source not diffed against main.
- No measured VRAM or step-time on a T4 for any option.
