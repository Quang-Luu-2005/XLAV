# Upgrades to the CLIPSeg decoder and text–image fusion/conditioning module (for E4)

Scope note: the notes compare options for replacing or upgrading ONE module of CLIPSeg (HF `CIDAS/clipseg-rd64-refined`, frozen CLIP ViT-B/16) under a budget of about 2 weeks on a Kaggle T4 (about 30 GPU-h/week). Most follow-up papers report on RefCOCO/+/G (overall IoU, usually with a fine-tuned backbone) or on open-vocabulary semantic segmentation (ADE/PASCAL mIoU), not PhraseCut. Their gains are **not directly comparable** to PhraseCut mIoU / IoU_FG / AP, and the protocol is flagged on each number.

Research-process caveat: in this session the web search tool and the WebFetch summarizer both failed with a model error. All facts below come from primary sources downloaded directly with curl (arXiv HTML full texts, the HF transformers v4.55.4 source, the HF model config, the timojl/clipseg README). I could not run a broad literature search for newer PhraseCut results (see Gaps).

---

## 0. Baseline facts: what the reproduced model actually is (needed for every option)

### Takeaway
The HF checkpoint differs from the paper text in a few details. It extracts CLIP layers **[3, 6, 9]**, not [3, 7, 9]. FiLM is applied **once**, before the first decoder block, and uses only the **pooled, projected** 512-d text/image embedding. The decoder works on a 22×22 token grid (352 px / 16) with D=64. The "refined" head is a 3×3 conv followed by two ×4 transposed convs.

### Cited Findings
- The HF config of `CIDAS/clipseg-rd64-refined` has `extract_layers: [3, 6, 9]`, `reduce_dim: 64`, `conditional_layer: 0`, `decoder_num_attention_heads: 4`, `decoder_intermediate_size: 2048`, `projection_dim: 512`, `use_complex_transposed_convolution: True`. The vision hidden size is 768 (patch 16) and the text hidden size is 512 — [HF config.json](https://huggingface.co/CIDAS/clipseg-rd64-refined/resolve/main/config.json)
- The paper says "We extract CLIP activations at layers S=[3,7,9], consequently our decoder has only three layers", with D=64 by default. This is a discrepancy with the HF config ([3,6,9]) — [CLIPSeg paper, arXiv 2112.10003](https://arxiv.org/abs/2112.10003)
- In HF, the activations are `hidden_states[i + 1] for i in self.extract_layers` ("we add +1 here as the hidden states also include the initial embeddings"), computed under `torch.no_grad()`. CLIP is therefore frozen by construction in the forward pass — [modeling_clipseg.py v4.55.4](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/clipseg/modeling_clipseg.py)
- `CLIPSegDecoder.forward(hidden_states, conditional_embeddings, ...)` reverses the activations (deepest first). For each activation it does `output = reduce(activation) + output`, a residual sum of linear 768→D projections; this is CLIPSeg's "U-Net-like skip". It applies FiLM only at `i == self.conditional_layer` (= 0): `film_mul(cond) * output + film_add(cond)`. It then runs one `CLIPSegDecoderLayer` (a standard pre-LN transformer block with `hidden_act="relu"` forced in code). Finally it drops the CLS token, reshapes to `[B, D, 22, 22]` and applies `transposed_convolution` — [modeling_clipseg.py v4.55.4](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/clipseg/modeling_clipseg.py)
- Refined head (`use_complex_transposed_convolution=True`): `Conv2d(D,D,3,pad=1) → ReLU → ConvTranspose2d(D, D/2, k=4, s=4) → ReLU → ConvTranspose2d(D/2, 1, k=4, s=4)`. The plain head is a single `ConvTranspose2d(D, 1, 16, stride=16)` — [modeling_clipseg.py v4.55.4](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/clipseg/modeling_clipseg.py)
- The README describes the refined weights as follows: "a more complex module for transforming tokens into predictions that allow for more refined predictions (in contrast to the square-like predictions of other weights)", loaded as `CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)`. No quantitative comparison of refined vs plain is given there — [timojl/clipseg README](https://github.com/timojl/clipseg/blob/master/README.md)
- The conditioning vector comes from `self.clip.get_text_features(...)` (pooled EOS token → `text_projection`, 512-d) or `get_image_features(...)`, both under `torch.no_grad()`. **Word-level token states are not used anywhere** in CLIPSeg — [modeling_clipseg.py v4.55.4](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/clipseg/modeling_clipseg.py)
- The forward also accepts precomputed `conditional_embeddings` (shape checked against `projection_dim`). The loss is plain `BCEWithLogitsLoss` — [modeling_clipseg.py v4.55.4](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/clipseg/modeling_clipseg.py)
- Original training recipe: 20,000 iterations, batch 64, lr 1e-3 with cosine decay to 1e-4 and no warmup, AMP, BCE as the only loss — [CLIPSeg paper appendix](https://arxiv.org/abs/2112.10003)

### Inferences
- `self.decoder` is a plain `nn.Module` attribute. You can replace it with a subclass of `CLIPSegDecoder`, or with any module that has the same `forward(activations, conditional_embeddings, ...)` signature, while keeping `model.clip` frozen and loaded from the checkpoint. Changes that keep tensor shapes (adding modules, zero-initialized residual branches) can load the reproduced decoder's `state_dict` with `strict=False` and warm-start.
- Word-level fusion cannot reuse `get_conditional_embeddings`. You need to call `model.clip.text_model(input_ids, attention_mask).last_hidden_state` (B×L×512, before `text_projection`) yourself and pass it to a custom decoder. Two ways to do this: subclass `CLIPSegForImageSegmentation.forward`, or write a thin wrapper that computes `activations` with `model.clip.vision_model(..., output_hidden_states=True)` and calls your decoder. The text encoder stays frozen and can run under `no_grad`. Token states for the PhraseCut phrases could also be cached offline.
- Rough parameter arithmetic from the code at D=64: 3 reduces (3×768×64 ≈ 147k), FiLM (2×512×64 ≈ 66k), 3 decoder blocks (each ≈ 4×64² attention + 2×64×2048 MLP ≈ 279k → ≈ 0.84M), plus the refined head. This is consistent with the ~1.1M trainable parameters the user quoted. The MLP (64→2048→64) dominates the parameter count.

### Gaps
- No official number compares rd64 vs rd64-refined on PhraseCut. The README is qualitative only.
- Whether the user's reproduction uses [3,6,9] (HF) or [3,7,9] (paper) should be checked in their notebook. It changes which layers any "skip" option would add.

---

## 1. Conditioning / fusion: FiLM vs cross-attention, word-level, contrastive (CRIS), pixel-word attention (LAVT), bidirectional, multi-stage

### Takeaway
In referring segmentation, the two consistently reported gains are (a) word-level cross-attention from pixels to text tokens instead of a single pooled sentence vector, and (b) injecting language at several stages instead of once. Each is worth roughly +1.7 to +4.6 IoU in ablations on RefCOCO-family benchmarks: CRIS decoder +2.2–4.6 IoU; LAVT pooled-vs-word −1.7 oIoU, no multi-stage language pathway −1.95 oIoU. Combined with a pixel–text alignment loss, CRIS reports about +4–8 IoU over its naive baseline. All of these numbers come from RefCOCO with fine-tuned backbones, not PhraseCut with frozen CLIP. Bidirectional/other attention variants (BCAM, GARAN) did *worse* than LAVT's simpler PWAM in a controlled swap.

### Cited Findings
**CRIS (Wang et al., CVPR 2022)**
- Architecture: multi-level visual features (ResNet C3–C5) fused with the global sentence feature by elementwise multiplication in a "cross-modal neck" (this is FiLM-like), plus 2D coordinate features. Then a **vision-language decoder of n transformer layers**: pixel self-attention → **cross-attention with word-level text features Ft ∈ R^{L×C} as keys/values** → MLP. Fixed sine positional encodings are added to both modalities — [CRIS, arXiv 2111.15174](https://arxiv.org/abs/2111.15174)
- "Text-to-pixel contrastive learning": pixel embeddings z_v are taken at H/4×W/4 after 4× upsampling and the sentence embedding z_t is projected. The loss is per-pixel −log σ(z_t·z_v^i) for foreground and −log(1−σ(z_t·z_v^i)) for background. **The prediction itself is σ(z_t·z_v)** reshaped to H/4×W/4 and then upsampled — [CRIS](https://arxiv.org/abs/2111.15174)
- Ablation (RefCOCO / RefCOCO+ / G-Ref val, IoU): adding the contrastive scheme alone gives **+1.98 / +2.98 / +3.43**. Adding a 1-layer VL decoder alone gives **+3.65 / +4.56 / +2.22**. Both together beat the baseline by "about 4%–8%". More decoder layers helped up to n=3, which was chosen as the default; n=4 "introduces more parameters, which could increase the risk of over-fitting" — [CRIS](https://arxiv.org/abs/2111.15174)
- CRIS final results (ResNet-50 CLIP): RefCOCO val/testA/testB **69.52 / 72.72 / 64.70**, RefCOCO+ **61.39 / 67.10 / 52.48**, G-Ref val/test **59.35 / 59.39**. With ResNet-101: 70.47 / 73.18 / 66.10. Settings: 416×416 input, decoder 8 heads with FFN 2048, 50 epochs, batch 64 on **8× V100**, Adam with lr 1e-4. The framework is trained end-to-end, so the CLIP encoders are not frozen — [CRIS](https://arxiv.org/abs/2111.15174)

**LAVT (Yang et al., CVPR 2022)**
- Early fusion inside a Swin backbone. A pixel-word attention module (PWAM) sits at each of 4 stages: single-head attention with visual features as queries and word features as keys/values gives a position-specific language feature, which is multiplied elementwise with projected visual features. A "language gate" (2-layer MLP with tanh) controls how much fused signal flows back into the backbone. The authors note PWAM has a much smaller memory footprint than image-to-image cross-modal attention — [LAVT, arXiv 2112.02244](https://arxiv.org/abs/2112.02244)
- Ablation, RefCOCO val (oIoU / mIoU): full model **72.73 / 74.46**. Without the language pathway (no multi-stage feedback): **70.78 / 71.96**, i.e. −1.95 / −2.50. Replacing word-level G_i with a **globally pooled sentence vector**: **71.03 / 72.31**, i.e. −1.70 / −2.15, and P@0.5–0.9 drop by 1–2 points. Removing both: 68.82 / 68.87 — [LAVT Table 2](https://arxiv.org/abs/2112.02244)
- Controlled swap of the attention module in the same framework (RefCOCO val oIoU): PWAM 72.73, **BCAM (bidirectional cross-modal attention, BRINet) 70.19**, GA/GARAN 71.20 — [LAVT Table 3d](https://arxiv.org/abs/2112.02244)
- Normalization matters: InstanceNorm in the PWAM projections gives 72.73 oIoU vs no norm 70.66, LayerNorm 71.92, BatchNorm 71.59. A tanh gate beats a sigmoid gate (72.73 vs 70.49) — [LAVT Table 3a,b](https://arxiv.org/abs/2112.02244)
- LAVT final results: RefCOCO val/testA/testB **72.73 / 75.82 / 68.79**, RefCOCO+ 62.14 / 68.38 / 55.10, G-Ref (UMD) 61.24 / 62.09 (oIoU) — [LAVT Table 1](https://arxiv.org/abs/2112.02244)

**CLIPSeg's own evidence on conditioning**
- CLIP-Deconv (FiLM + linear projection + deconvolution only, no transformer decoder) on PhraseCut: **37.7 mIoU / 49.5 IoU_FG / 71.2 AP**, vs CLIPSeg (PC) **46.1 / 56.2 / 78.2**. The transformer decoder after FiLM is therefore worth about +8 mIoU on PhraseCut — [CLIPSeg Table 3](https://arxiv.org/abs/2112.10003)
- Masking-based visual prompt engineering inside CLIP ("constraining interactions") worked poorly; the authors conclude "more complex strategies are necessary to combine image and mask internally" — [CLIPSeg](https://arxiv.org/abs/2112.10003)

**ZegCLIP (relationship descriptor = an alternative to FiLM/raw text queries; open-vocab semantic seg, not referring)**
- ZegCLIP uses a light transformer decoder that matches text queries with CLIP patch embeddings. Its "Relationship Descriptor" replaces the raw text embedding t_c with a combination involving t_c ⊙ g, where g is the image CLS embedding. On VOC with a **frozen CLIP** (Baseline-Fix): hIoU **26.5 → 59.6** with RD, 51.2 with the non-mutually-exclusive (sigmoid) loss, and **73.9** with both. On COCO-Stuff: 16.2 → 23.3 (RD), 32.4 (both) — [ZegCLIP Table 5, arXiv 2212.03588](https://arxiv.org/abs/2212.03588)
- ZegCLIP total learnable parameters: 13.8M (VOC), compared with ZegFormer's 60.3M — [ZegCLIP Table 3](https://arxiv.org/abs/2212.03588)

### Inferences
- **Why word-level cross-attention helps:** a pooled EOS vector compresses relations and attributes ("the man *left of* the red car"). Cross-attention lets each pixel select the words relevant to it. PhraseCut phrases are template-generated (category + attribute + relation, per [GRES](https://arxiv.org/abs/2306.00968)), so attribute/relation words are exactly what a pooled vector blurs. The expected gain on PhraseCut is plausibly similar in direction to the LAVT −1.7 oIoU (pooled vs word) ablation, but this has not been measured on PhraseCut with frozen CLIP.
- **Cheapest high-value E4 option ("FiLM + word cross-attention, multi-stage"):** add a cross-attention sublayer (queries = 64-d pixel tokens; keys/values = a Linear(512→64) projection of the CLIP text `last_hidden_state`, using the attention mask) inside each of the 3 `CLIPSegDecoderLayer`s, or as a residual block after each one. Keep FiLM. Zero-init the output projection (or add a tanh gate initialized at 0, à la LAVT's gate) so the network starts exactly at the reproduced baseline and can be **fine-tuned from the reproduced decoder** rather than trained from scratch. Extra parameters ≈ 3 × (4×64² + 512×64) ≈ 0.15M if the text projection is shared or not. VRAM impact is negligible: 22×22=484 queries × ≤77 keys.
- **Multi-stage FiLM** (apply FiLM at every block instead of only `conditional_layer=0`) is a ~10-line subclass change, adds about 2×66k parameters, and is the cheapest test of the "condition at multiple stages" finding. It is weaker evidence than word-level attention: the LAVT language-pathway result is the closest analogue, but it also feeds back into the backbone, which a frozen CLIP cannot do.
- **CRIS-style "contrastive" head:** with one positive phrase per image, the CRIS loss is mathematically a per-pixel sigmoid BCE on the dot product between a projected text vector and projected pixel embeddings at 1/4 resolution. For CLIPSeg it amounts to replacing the transposed-conv logit head with "pixel embedding · text embedding". CLIPSeg already applies BCE to its logits. The CRIS gain (+2–3.4 IoU) was measured against a naive baseline with a fine-tuned ResNet backbone, so the expected gain on top of CLIPSeg's FiLM+transformer decoder is uncertain and likely smaller. It may also pair poorly with PC+ visual prompts. A cheap variant is an *auxiliary* pixel–text loss on the decoder tokens while keeping the main head.
- **LAVT-style early fusion inside the backbone is not compatible with a frozen CLIP** unless adapters are inserted into CLIP layers (that falls in the adapter/PEFT area, outside this module-swap scope). Only the PWAM *operator* (pixel→word attention + elementwise product + InstanceNorm + tanh gate) is transferable into the decoder.
- Bidirectional fusion (text tokens also attend to pixels) has no supporting evidence of gains: BCAM lost to PWAM by 2.5 oIoU in LAVT's controlled swap. It is a low-priority option.
- The ZegCLIP RD gains are huge but measured on zero-shot semantic segmentation with seen/unseen splits. They mostly fix overfitting to seen classes, so they are not evidence for PhraseCut. A cheap analogue is to FiLM with [t, t⊙g], where g is the CLIP image embedding already computed as `pooled_output` in the HF forward.

### Gaps
- I found no paper that ablates FiLM vs cross-attention **with a frozen CLIP ViT-B/16 on PhraseCut**. All the fusion ablations above use RefCOCO with fine-tuned backbones (CRIS: CLIP ResNet; LAVT: Swin + BERT).
- I found no ablation of conditioning at multiple decoder blocks in CLIPSeg itself. The HF `conditional_layer` config exists, but the paper reports no ablation of it.

---

## 2. Decoder capacity: reduce_dim, depth, number/choice of extracted layers, skip connections

### Takeaway
D is the strongest documented capacity knob in CLIPSeg. On PhraseCut (text-only training, PC), D=128 gives **48.2 mIoU vs 46.1 for D=64** (+2.1 mIoU, +0.3 IoU_FG, AP equal at 78.2). D=16 loses 6.2 mIoU vs D=64 (PC+ model). Using only layer 3 loses 11.7 mIoU, so deeper layers matter. Changing D requires training from scratch, but at roughly 2.3M parameters it fits easily on a T4.

### Cited Findings
- PhraseCut test, Table 3 (t = binary threshold), reported as mIoU / IoU_FG / AP: CLIPSeg (PC+, D=64, t=0.3) **43.4 / 54.7 / 76.7**; CLIPSeg (PC, D=128) **48.2 / 56.5 / 78.2**; CLIPSeg (PC, D=64) **46.1 / 56.2 / 78.2**; CLIP-Deconv 37.7 / 49.5 / 71.2; ViTSeg (PC) 38.9 / 51.2 / 74.4; MDETR 53.7 mIoU; HulaNet 41.3 / 50.8. The authors write: "Especially, a high capacity decoder (D=128) seems to be beneficial for PhraseCut." They attribute MDETR's lead to it operating "at full image resolution" and receiving "two rounds of fine-tuning on PhraseCut" — [CLIPSeg Table 3](https://arxiv.org/abs/2112.10003)
- Ablation, Table 9 (PhraseCut; best threshold per model), text mIoU / AP and visual mIoU / AP: CLIPSeg (PC+) 43.6 / 76.7 | 25.4 / 55.6; **D=16: 37.4 / 71.5** | 24.7 / 51.2; **only layer 3: 31.9 / 64.9** | 21.5 / 48.6; no visual (PC): 46.4 / 77.8 | 14.4 / 31.0; no CLIP pre-training: 13.1 / 12.6 — [CLIPSeg Table 9](https://arxiv.org/abs/2112.10003)
- The paper's comment on the D=16 result: "When the number of parameters is reduced to 16 ('D=16') performance decreases substantially, which indicates the importance of the information processing in the decoder." On using only layer 3: "higher level features of CLIP are useful for segmentation" — [CLIPSeg §5.5](https://arxiv.org/abs/2112.10003)
- **There is no rd128 row in the ablation table, and no rd16/rd64/rd128 sweep under one protocol.** D=128 appears only in Table 3, trained on PC rather than PC+. The README releases D=64 (×2: plain and refined) and D=16 weights only — [CLIPSeg](https://arxiv.org/abs/2112.10003); [README](https://github.com/timojl/clipseg/blob/master/README.md)
- SAN (frozen CLIP ViT-B/16, side network; ADE-150 mIoU) on fusing CLIP layers: no fusion 21.1. Single-layer fusion: stem 20.0, 3rd 24.1, 6th 26.2, **9th 27.1**. Multi-layer fusion: {6,9} 27.0, {3,6,9} 27.7, {stem,3,6,9} 27.8. Deeper layers matter most, and multi-layer fusion adds about +0.8 — [SAN Table 4, arXiv 2302.12242](https://arxiv.org/abs/2302.12242)
- CRIS decoder depth: performance improves from n=1 to n=3; n=4 risks overfitting — [CRIS](https://arxiv.org/abs/2111.15174)

### Inferences
- **D=128 is the best-documented "safe" E4 option.** The authors' own PC-trained comparison shows +2.1 mIoU on PhraseCut (46.1 → 48.2), although the AP gain is 0 and the IoU_FG gain is only +0.3, so most of the gain is in the per-phrase mean. Caveats: one run per setting (no variance reported), and it was trained on PC, not PC+. The user trains on PC+, where the D=64 model scores 43.4–43.6 mIoU.
- Cost: from the code arithmetic, D=128 has ≈ 3×768×128 + 2×512×128 + 3×(4×128² + 2×128×2048) ≈ 0.30M + 0.13M + 1.77M ≈ 2.2–2.3M parameters plus the head. Activations at 22×22×128 are tiny next to frozen CLIP's. VRAM and step time should be within a few percent of baseline, so the cost is one full training run (6–10 h). Shapes change, so it **must be trained from scratch**; you cannot warm-start from the rd64 decoder, except by partial or Net2Net-style widening, which is not worth the effort.
- Adding a 4th extracted layer (e.g. [3,6,9,11]) adds one reduce plus one block (≈0.28M at D=64) and also needs retraining. SAN's ablation suggests diminishing returns from more layers once [3,6,9] is used (+0.1 from adding the stem). Layer 11/12 features are more "CLIP-semantic", but their usefulness is untested in CLIPSeg.
- `decoder_intermediate_size=2048` with D=64 is a 32× MLP expansion and holds about 75% of the decoder parameters. If capacity is the bottleneck, widening D is the documented lever; changing the MLP ratio is untested.

### Gaps
- No published PhraseCut number for D=128 trained on PC+, and no D=256.
- No published ablation of decoder depth or extract-layer set beyond "only layer 3".

---

## 3. Upsampling / resolution: refined transposed conv, FPN-style, CARAFE/FeatUp/JBU, high-resolution skips

### Takeaway
CLIPSeg decodes at 22×22 (1/16) and upsamples ×16 in the head. Every CLIP-ViT layer has the same 1/16 grid, so "skip connections from early CLIP layers" add semantics, not spatial resolution. Real resolution gains need either (a) learned upsampling of CLIP features (CAT-Seg's transposed conv to 2×/4×), or (b) image-guided upsampling using the RGB input (FeatUp JBU, or a small conv stem as in SAN, which runs a side network at higher resolution). Reported gains for better upsamplers are about +1–4.5 mIoU on semantic segmentation benchmarks. None are measured on PhraseCut.

### Cited Findings
- CLIPSeg appendix: on 5,000 PhraseCut test samples, "there is a clear trend towards better performance on larger objects" (Fig. 8). Small objects are the weak spot — [CLIPSeg appendix](https://arxiv.org/abs/2112.10003)
- CLIP ViT-B/16 handles larger inputs well, with an "optimal image size of around 350×350" in a classification probe (motivation for the 352 px input) — [CLIPSeg appendix](https://arxiv.org/abs/2112.10003)
- **CAT-Seg upsampling decoder:** "Similar to FPN, we employ bilinear upsampling on the aggregated cost volume and concatenate it with the corresponding level of feature map extracted from CLIP, followed by a 3×3 conv", iterated N_U times. The high-resolution guidance maps come from **intermediate CLIP ViT layers upsampled with a single learnable transposed convolution**: 24×24 → 48×48 and → 96×96. No extra backbone is used — [CAT-Seg, arXiv 2303.11797](https://arxiv.org/abs/2303.11797)
- CAT-Seg Table 5 (ViT-B; mIoU on A-847 / PC-459 / A-150 / PC-59 / PAS-20 / PAS-20b): without the upsampling decoder 9.9 / 16.1 / 28.4 / 52.9 / 93.2 / 73.3; **with it 12.0 / 19.0 / 31.8 / 57.5 / 94.6 / 77.3**, i.e. +1.4 to +4.6 mIoU — [CAT-Seg Table 5](https://arxiv.org/abs/2303.11797)
- **FeatUp (ICLR 2024)**, linear probe on frozen low-res ViT features upsampled 16×, COCO-Stuff 27 classes (Acc / mIoU): Low-res 65.17 / 40.65; Bilinear 66.95 / 42.40; Resize-conv 67.72 / 42.95; FADE 67.1 / 42.39; SAPA 65.69 / 41.17; **FeatUp JBU 68.77 / 43.41**; FeatUp Implicit 71.58 / 47.37. Implicit FeatUp is per-image optimization, so it is impractical for training. The probe was trained at low resolution and frozen — [FeatUp Table 1, arXiv 2403.10516](https://arxiv.org/abs/2403.10516)
- FeatUp end-to-end in a Segformer decoder (ADE20K val mIoU; added params): Bilinear 39.7; Resize-conv 41.1 (+3.54M); IndexNet 41.5 (+12.6M); A2U 41.5 (+0.12M); **CARAFE 42.4 (+0.78M)**; SAPA 41.6 (+0.20M); FADE 43.6 (+0.29M); **FeatUp JBU 44.2 (+0.16M)** — [FeatUp Table 2](https://arxiv.org/abs/2403.10516)
- The JBU guidance uses the **raw input image** through a 2-layer MLP (30-d) over 3×3 neighborhoods. FeatUp ships a CUDA kernel; the naive `nn.Unfold` implementation uses up to about two orders of magnitude more memory (e.g. 6628 MB vs 372 MB peak for a 16×32×32×2048 input) — [FeatUp §3, Table 6](https://arxiv.org/abs/2403.10516)
- **CARAFE** (content-aware reassembly, ICCV 2019): "consistent and substantial gains" of 1.2% (detection), 1.3% (instance seg), **1.8% (semantic seg)**, 1.1 dB (inpainting) "with negligible computational overhead" — [CARAFE, arXiv 1905.02188](https://arxiv.org/abs/1905.02188)
- SAN runs CLIP at low resolution (320²) and its side network at high resolution (640²). Raising CLIP's own input beyond 320² hurt ADE-150 (448²: 26.1; 640²: 24.6 vs 27.8 at 320²). This is evidence that feeding CLIP larger images is not the way to gain resolution — [SAN Table 8](https://arxiv.org/abs/2302.12242)

### Inferences
- **Upsampling head swap (fits budget):** replace `transposed_convolution` with an FPN-like head. Steps: (i) take the 22×22×D decoder output, (ii) 2× bilinear + conv, fused with a 44×44 map obtained from a CLIP intermediate layer via a learned ConvTranspose (CAT-Seg recipe) or from a tiny RGB conv stem (stride 8/4), (iii) repeat to 88×88, then predict and bilinearly upsample to 352. Extra params ≈ 0.1–0.5M. Memory grows with the 88×88×(D/2) maps, which is still small at batch 64 on a T4 with AMP. The decoder transformer weights can be **kept and warm-started** from the reproduced checkpoint; only the head is new. A head-only fine-tune (frozen transformer decoder) for a few thousand iterations is a cheap first test.
- An image-guided upsampler (RGB stem or JBU) is the only way to recover boundaries finer than 16 px. CLIP-layer skips are all 1/16, and the CAT-Seg-style transposed conv of CLIP features adds learned but not image-detail resolution. FeatUp JBU's advantage over CARAFE in Segformer was +1.8 mIoU at fewer parameters. FeatUp's pretrained CLIP upsampler exists, but using it requires its CUDA extension, which is a compile risk on Kaggle. A pure-PyTorch `unfold` JBU at 22→88 with D=64 channels is cheap enough at this scale.
- PhraseCut mIoU averages per-phrase IoU, and many targets are small (Fig. 8 trend), so a better head plausibly helps mIoU more than AP. This is untested. Boundary gains also show up only weakly in region IoU metrics, so you may want a boundary F-score or a small-object IoU breakdown to show the effect.

### Gaps
- No quantitative refined-vs-plain comparison for CLIPSeg.
- No upsampler study on referring/phrase segmentation with frozen CLIP. All upsampler numbers are on COCO-Stuff linear probes or ADE20K Segformer/CAT-Seg.
- FeatUp Kaggle compatibility (CUDA build) was not verified.

---

## 4. Follow-up works that keep CLIP (mostly) frozen and redesign the decoder

### Takeaway
The most decoder-centric follow-ups (ZegCLIP, SAN, CAT-Seg) target open-vocabulary *semantic* segmentation, not phrase segmentation. Their transferable ideas are: deeper and multi-layer CLIP features, an FPN/upsampling decoder with CLIP-feature guidance, cost-volume (cosine similarity) inputs, and sigmoid (non-exclusive) losses. With frozen CLIP, their decoder redesigns give modest-to-large gains in their own settings. Their biggest jumps come from *adapting* CLIP (prompt tuning, attention fine-tuning), which is outside a pure decoder swap.

### Cited Findings
- **CAT-Seg** (CVPR 2024). Feature aggregation vs cost aggregation with **frozen CLIP**, mIoU on A-847 / PC-459 / A-150 / PC-59 / PAS-20 / PAS-20b: feature-agg + freeze 3.1 / 8.7 / 16.6 / 46.8 / 92.3 / 69.7; **cost-agg + freeze 10.0 / 14.5 / 26.0 / 46.9 / 94.2 / 65.1**. With fine-tuning: 5.6 / 12.8 / 23.6 / 58.1 / 96.3 / 77.7 (feature) vs 14.7 / 23.2 / 35.3 / 60.3 / 96.7 / 78.9 (cost) — [CAT-Seg Table 3](https://arxiv.org/abs/2303.11797)
- CAT-Seg CLIP-tuning ablation (#params / memory): Freeze 10.4 / 15.0 / 31.8 / 52.5 / 92.2 / 71.3 (5.8M, 20.0 GiB); Full FT 13.6 / 22.2 / 34.0 / 61.1 / 97.3 / 79.7 (393.2M, 26.8 GiB); attention-only FT 15.7 / 23.7 / 37.1 / 63.1 / 97.1 / 81.5 (134.9M) — [CAT-Seg Table 6](https://arxiv.org/abs/2303.11797)
- CAT-Seg training: 80k iterations, batch 4, 4× RTX 3090; lr 2e-4 (model) and 2e-6 (CLIP) — [CAT-Seg](https://arxiv.org/abs/2303.11797)
- **SAN** (CVPR 2023): frozen CLIP plus a side ViT (8 blocks, dim 240, 100 queries) that fuses CLIP layers {stem,3,6,9} and predicts masks plus attention biases for the last 3 CLIP blocks. It has **8.4M trainable params**, 64.3 GFLOPs and 15.2 FPS. End-to-end ("CLIP-aware") training gives +4.5 mIoU over two-stage training, and a decoupled head gives +1.7 (ADE-150) — [SAN Tables 3, 6, 7](https://arxiv.org/abs/2302.12242)
- **ZegCLIP** (CVPR 2023): a frozen-CLIP baseline with a light transformer decoder fails on unseen classes (VOC hIoU 26.5). Relationship descriptor plus sigmoid loss bring it to 73.9. Deep prompt tuning of the CLIP image encoder, rather than fine-tuning, is used in the final model — [ZegCLIP Table 5](https://arxiv.org/abs/2212.03588)
- **CRIS / LAVT** (referring seg, RefCOCO): see §1. Both fine-tune their encoders, so their final numbers are not "frozen CLIP" results — [CRIS](https://arxiv.org/abs/2111.15174); [LAVT](https://arxiv.org/abs/2112.02244)
- **GRES/ReLA** note on datasets: PhraseCut multi-target expressions are only a "fallback", and its expressions are "written using templates rather than free natural language" — [GRES, arXiv 2306.00968](https://arxiv.org/abs/2306.00968)
- **PhraseCut reference points** (from CLIPSeg Table 3): MDETR 53.7 mIoU (full-resolution, fine-tuned detector), HulaNet 41.3 / 50.8, Mask-RCNN top 39.4 / 47.4, RMI 21.1 / 42.5 — [CLIPSeg](https://arxiv.org/abs/2112.10003)
- FC-CLIP (arXiv 2308.02487) uses a **frozen convolutional** CLIP (ConvNeXt) backbone with a Mask2Former-style pixel/mask decoder. It depends on a CNN CLIP and a Mask2Former decoder, so it does not apply to a ViT-B/16 CLIPSeg swap. Its paper was downloaded, but I did not extract numbers from it.

### Inferences
- **Cost-volume input** (CAT-Seg idea) adapted to CLIPSeg: compute the cosine similarity between each CLIP patch token (projected with CLIP's own `visual_projection` from the final layer) and the text embedding. This gives a 22×22 "rough mask". Concatenate it as an extra channel to the decoder tokens or the head. It is cheap (no new big modules) and adds an explicit text–pixel alignment signal similar in spirit to CRIS. The evidence for the gain (CAT-Seg frozen: cost-agg ≫ feature-agg on A-847/A-150) comes from open-vocab semantic segmentation, where feature aggregation overfits seen classes. On in-domain PhraseCut the benefit is uncertain.
- SAN-style or CAT-Seg-style full systems need multi-day 3090/V100 training (CAT-Seg: 4× 3090, 80k iterations; CRIS: 8× V100, 50 epochs) or tens of millions of parameters with attention fine-tuning. **They do not fit 2 weeks on a T4.** Only their sub-components (upsampling decoder, multi-layer fusion, cost map) are realistic.

### Gaps
- I could not identify (search tools unavailable) post-2022 papers reporting **PhraseCut** mIoU/IoU_FG/AP for frozen-CLIP decoder redesigns. The comparison set for PhraseCut remains CLIPSeg Table 3 (MDETR 53.7 as an upper reference).
- FC-CLIP and DenseCLIP numbers were not extracted. DenseCLIP's pixel-text score-map auxiliary loss is related to the CRIS idea, but it fine-tunes the backbone.

---

## 5. Feasibility summary per option (HF transformers 4.55.4, Kaggle T4, ~2 weeks)

### Takeaway
Ranked by expected gain per GPU-hour:
1. **D=128 retrain**: documented +2.1 mIoU on PhraseCut (PC); one from-scratch run.
2. **Word-level cross-attention added to the existing decoder**: strongest literature support for referring segmentation; can be warm-started from the reproduced decoder with zero-init gates.
3. **FPN/image-guided upsampling head**: targets the documented small-object weakness; warm-startable, head-only.
4. Multi-stage FiLM: trivial to implement, weak evidence.
5. CRIS-style dot-product head / auxiliary pixel–text loss: evidence comes from a non-frozen setting.

SAN, CAT-Seg, LAVT-style early fusion and CRIS end-to-end are **not feasible** in the budget.

### Cited Findings
- The HF decoder is called as `self.decoder(activations, conditional_embeddings, ...)` and the CLIP forward runs under `torch.no_grad()`. A replacement decoder only has to accept the activation list and a conditioning tensor — [modeling_clipseg.py v4.55.4](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/clipseg/modeling_clipseg.py)
- The text hidden size of CLIP in this checkpoint is 512, and so is `projection_dim`. Word-token states (`text_model(...).last_hidden_state`) are therefore 512-d, the same width as the pooled embedding before projection — [HF config.json](https://huggingface.co/CIDAS/clipseg-rd64-refined/resolve/main/config.json)
- Original training budget: 20k iterations at batch 64 (the CLIPSeg default) — [CLIPSeg appendix](https://arxiv.org/abs/2112.10003)

### Inferences (option table; parameter and VRAM figures are estimates from the code, not measured)

| Option | Code change in HF 4.55.4 | Extra params | VRAM / time vs baseline | Warm-start from reproduced decoder? | Evidence for gain | Fits budget? |
|---|---|---|---|---|---|---|
| D=128 (rd128) | `CLIPSegConfig(reduce_dim=128)` + new decoder; reuse `model.clip` weights | ≈ +1.2M (≈2.3M total) | ≈ same (CLIP dominates) | No (shape change) → full run 6–10 h | +2.1 mIoU, +0.3 IoU_FG, +0 AP on PhraseCut, PC-trained ([CLIPSeg T3](https://arxiv.org/abs/2112.10003)) | Yes, 1 run |
| Multi-stage FiLM | Subclass `CLIPSegDecoder.forward`, apply FiLM (separate Linear pair per block) at every i | ≈ +0.13M | ≈ same | Yes (init block-0 FiLM from checkpoint, others to identity: mul=1, add=0) | Indirect: LAVT language pathway −1.95 oIoU when removed (RefCOCO, different arch) | Yes |
| Word-level cross-attention (CRIS-decoder / PWAM-style) in each decoder block | Subclass the decoder plus a custom forward that also returns `clip.text_model(...).last_hidden_state` and the attention mask | ≈ +0.1–0.2M at D=64 | Small (484 queries × ≤77 keys) | Yes, with zero-init output projection / tanh gate (LAVT-style) | CRIS 1-layer decoder +2.2–4.6 IoU; LAVT pooled→word +1.7 oIoU (RefCOCO, fine-tuned backbones) | Yes; best literature support |
| CRIS-style pixel·text logit head / aux text-pixel loss | Replace or add to `transposed_convolution`: project tokens (upsampled ×4) and text, take the dot product | ≈ +0.05M | Small | Mostly yes | CRIS contrastive +2.0–3.4 IoU vs naive baseline (fine-tuned CLIP-RN) | Yes, uncertain gain |
| Cost-map channel (CAT-Seg idea) | Compute cos-sim(final CLIP patch tokens via `visual_projection`, text emb), concatenate into the decoder | ≈ +few k | ≈ same | Yes (new input channel zero-init) | CAT-Seg frozen: cost-agg ≫ feature-agg on open-vocab mIoU; not referring | Yes, speculative |
| FPN / CAT-Seg-style upsampling head (CLIP-feature transposed-conv guidance) | Replace `transposed_convolution`; also needs `activations` passed to the head | ≈ +0.1–0.5M | Moderate (88×88 maps) | Yes (keep transformer; train head first) | CAT-Seg +1.4–4.6 mIoU (open-vocab sem-seg) | Yes |
| Image-guided upsampler (RGB stem / JBU / CARAFE) | As above, plus `pixel_values` passed to the head | ≈ +0.1–0.8M | Moderate; JBU via `unfold` is memory-heavy at high res, FeatUp CUDA kernel is a build risk | Yes | FeatUp JBU 44.2 vs bilinear 39.7 / CARAFE 42.4 mIoU (ADE20K Segformer); CARAFE +1.8 sem-seg | Yes (pure-PyTorch variant) |
| Extra extracted layer (e.g. [3,6,9,11]) | Config `extract_layers` + new decoder | ≈ +0.28M | ≈ same | No (new block in sequence) → full run | SAN: multi-layer fusion +0.8 over best single layer (ADE-150) | Yes, 1 run, low expected gain |
| LAVT early fusion in CLIP backbone | Requires adapters inside the frozen CLIP | Large | High | No | LAVT gains rely on backbone fusion | No (out of scope / budget) |
| SAN / CAT-Seg full / CRIS end-to-end | New framework | 5.8M–135M+ | CAT-Seg 20–27 GiB; CRIS 8×V100 | No | Large, but on other tasks | **No: exceeds T4 / 2-week budget** |

- Suggested E4 plan under ~60 GPU-h:
  - (a) A short warm-start fine-tune of "word cross-attention + zero-init gate" from the reproduced checkpoint, a few thousand iterations (~2–3 h), to check the direction.
  - (b) Then one full run of the best variant, and optionally one D=128 run as a "documented-capacity" control.
  - Report PhraseCut mIoU / IoU_FG / AP at the same threshold protocol as the baseline. CLIPSeg Table 9 uses the best threshold per model, while Table 3 uses t=0.3 (0.1 for ViTSeg PC+).
- Warm-start caveat: if the decoder is changed and only fine-tuned while the baseline got a full schedule, the comparison is confounded by total training. Train the baseline for the same extra iterations as a control.

### Gaps
- None of the parameter, VRAM or time figures in the table were measured. They come from code-level arithmetic and should be profiled with one batch on the T4.
- No source quantifies the PhraseCut effect of any upsampling-head change, or of word-level fusion with frozen CLIP ViT-B/16.
