# Inference-time / post-hoc refinement modules on top of CLIPSeg (for E4)

Research note on method: WebSearch and the WebFetch summarizer were unavailable during this session (tool errors). All findings below come from primary PDFs downloaded directly from arXiv and read with pdftotext, plus the HF transformers docs pages, GitHub READMEs, and the PyPI JSON API fetched with curl. No blog-post benchmarks were consulted, so there are no third-party "CLIPSeg+SAM" numbers on PhraseCut here (see Gaps).

## 1. Known failure modes of CLIPSeg and the evidence for them

### Takeaway
The CLIPSeg paper itself reports four weaknesses: (a) worse results on small objects, (b) strong threshold dependence, which is why it also reports AP, (c) a fixed working resolution of about 350 px, and (d) masks that come from 16-px patches. Outside evidence adds weak spatial-relation ("left/right") and negation understanding in CLIP-family models, plus a PhraseCut profile with many small and multi-instance targets. Boundary and small-object errors are the ones a post-hoc module can realistically fix. Relations and negation are semantic errors that a mask refiner cannot fix, and it may even make a wrong mask look sharper.

### Cited Findings
- CLIPSeg uses CLIP ViT-B/16 (patch size 16) with a frozen encoder and a small decoder. It handles other image sizes by interpolating the positional embeddings, and "performance only decreases for images larger than 350 pixels". All experiments use 352×352 inputs — [Lüddecke & Ecker, arXiv 2112.10003](https://arxiv.org/abs/2112.10003)
- Supplement: CLIP ViT-B/16 classification accuracy peaks at about 350×350 input. The paper's Limitations section says "Image size may vary but only within certain limits" — [CLIPSeg paper](https://arxiv.org/abs/2112.10003)
- **Small objects:** the supplementary analysis (Fig. 8, 5,000 PhraseCut test samples, CLIPSeg PC+) finds "a clear trend towards better performance on larger objects". Performance across classes is "fairly balanced", and prompt wording has only a "small effect" — [CLIPSeg paper](https://arxiv.org/abs/2112.10003)
- **Threshold sensitivity:** the authors say the optimal binary threshold "can strongly deviate from 0.5" when the prior probability of a query matching differs between training and inference. They therefore report thresholds tuned per task and model, and add AP as a threshold-independent metric — [CLIPSeg paper](https://arxiv.org/abs/2112.10003)
- CLIPSeg's pixel AP is not an exact sort over all pixels. It uses "a fixed set of thresholds", sums TP/FP statistics per threshold across images, and integrates the PR curve with Simpson's rule — [CLIPSeg paper, supplement](https://arxiv.org/abs/2112.10003)
- Reference PhraseCut numbers (t=0.3, mIoU / IoU_FG / AP): CLIPSeg (PC+) 43.4 / 54.7 / 76.7; CLIPSeg (PC) 46.1 / 56.2 / 78.2; CLIPSeg (PC, D=128) 48.2 / 56.5 / 78.2; ViTSeg (PC) 38.9 / 51.2 / 74.4 — [CLIPSeg paper, Table 3](https://arxiv.org/abs/2112.10003)
- **PhraseCut dataset profile:** 17.6% of phrases refer to two or more instances. 20.2% of targets cover less than 2% of the image, while the smallest RefCOCO target is 2.4%. Cumulative IoU "is dominated by large targets", and mean IoU "balances the performance on small and large objects" — [Wu et al., PhraseCut, arXiv 2008.01187](https://arxiv.org/abs/2008.01187)
- PhraseCut subset mean-IoU for HULANet (cat+att+rel): small 17.3, mid 41.7, large 68.2; single 42.6, multi 38.4, many 35.7. The RMI baseline scores 1.2 mIoU on small vs 49.5 on large and "predicts empty masks on challenging phrases and small targets". These are pre-CLIPSeg models, but they show where the dataset is hard — [PhraseCut paper, Table 4](https://arxiv.org/abs/2008.01187)
- **Spatial relations:** "What's 'up' with vision-language models?" evaluates 18 VL models on left/right/on/under benchmarks and finds "all perform poorly". For example, BLIP fine-tuned on VQAv2 reaches 56% vs 99% for humans. The authors trace this to too little reliable spatial data in corpora such as LAION-2B — [Kamath et al., arXiv 2310.19785](https://arxiv.org/abs/2310.19785)
- **Negation:** NegBench (79k examples, 18 task variants) finds that modern VLMs "struggle significantly with negation, often performing at chance level". Fine-tuning CLIP on synthetic negated captions gives about +10% recall on negated queries — [Alhamoud et al., arXiv 2501.09425](https://arxiv.org/abs/2501.09425)
- CLIP-based approaches align global features, "making it less suitable for fine-grained segmentation", and cannot precisely find the edges of local regions. ClipSAM states this as its motivation for adding SAM — [ClipSAM, arXiv 2401.12665](https://arxiv.org/abs/2401.12665)

### Inferences
- With a 352×352 input and patch 16, the encoder grid is 22×22, so each token covers about 16×16 input pixels. Upsampling makes boundaries blurry by construction. This is the error mode that refinement modules (SAM, CRF, PAMR, CascadePSP) target directly.
- On PhraseCut, mIoU (per-sample average) rewards fixes on small objects much more than cumulative IoU does. A refiner that only sharpens the boundaries of large objects will mostly move cum-IoU. One that recovers small objects will move mIoU.
- Left/right, negation, and wrong-instance choice come from the text-image alignment, not from mask quality. A post-hoc refiner keeps whatever region CLIPSeg chose, so these errors persist and may become "confidently wrong" crisp masks. That can lower AP if refinement overrides the soft scores.

### Gaps
- I found no paper that breaks CLIPSeg (CIDAS rd64-refined) PhraseCut results down by relation type, negation, or instance count. Only the object-size trend from the CLIPSeg supplement is documented. The project's own miniv evaluation could produce this by slicing on PhraseCut's metadata (size buckets, multi-instance flags).
- No direct measurement of CLIPSeg itself on left/right or negation prompts was found. The evidence above is for CLIP-family VLMs in general.

## 2. SAM / SAM 2 / SAM 2.1 refinement of CLIPSeg masks (prompt types, gains, failure cases)

### Takeaway
The strongest primary evidence on turning a coarse mask into SAM prompts is SAMRefiner (ICLR 2025). Each prompt type alone is worse than the coarse mask: tight-box only 68.8 IoU and mask-only 37.3, against 81.4 for the coarse mask. Combining points, boxes, and a Gaussian-style mask prompt raises it to 86.9 (+5.5) on DAVIS-585, and MaskCLIP/CLIP-ES pseudo-masks improve by +9.5/+8.5 mIoU on VOC. ClipSAM also finds points+boxes beat either alone, and fuses SAM masks with the CLIP map as soft scores. That soft fusion matters for keeping pixel AP meaningful. Known failure modes: multi-object/multi-instance masks, extremely noisy coarse masks, the choice among SAM's whole/part/subpart outputs (the "handle of the cup" problem), and SAM's poorly calibrated IoU head.

### Cited Findings
**SAMRefiner (Lin et al., ICLR 2025): the most relevant recipe**
- Mines three prompt types from a coarse mask: distance-guided points (the point farthest from the background, plus a negative point), "context-aware elastic" boxes, and Gaussian-style mask prompts. It adds a split-then-merge (STM) step for multi-object masks and an optional IoU-head LoRA adaptation (SAMRefiner++). It uses SAM ViT-H — [SAMRefiner, arXiv 2502.06756](https://arxiv.org/abs/2502.06756)
- Prompt ablation on DAVIS-585 (coarse mask = 81.4 IoU / 71.4 boundary IoU), SAMRefiner / SAMRefiner++ IoU: Point 53.7/56.4; Box 68.8/70.8; Mask 37.3/40.4; Point+Box 76.7/79.1; Point+Mask 77.5/80.6; Box+Mask 84.6/85.1; **All 86.9/87.1** (boundary IoU 75.1/75.4) — [SAMRefiner Table 1](https://arxiv.org/abs/2502.06756)
- "The mask prompt ... shows poor performance on its own but can bring an obvious advantage of nearly 20% IoU for point and box." SAM "fails to work by merely using the mask as an input prompt", because during SAM pre-training the mask prompt only acts as an auxiliary in the cascade — [SAMRefiner](https://arxiv.org/abs/2502.06756)
- Figure 2 shows failure cases of the naive tight box around the coarse mask, which "fails to obtain satisfactory performance". The box center is sensitive to distant false positives: box-center point 22.2 IoU vs random point 37.8 vs distance-guided 53.7 — [SAMRefiner, Fig. 2 and Table 2a](https://arxiv.org/abs/2502.06756)
- Split-then-merge on semantic masks: MaskCLIP coarse masks 51.1 → 57.3 mIoU (+6.2); CLIP-ES 79.1 → 79.3 — [SAMRefiner Table 2c](https://arxiv.org/abs/2502.06756)
- Pseudo-mask mIoU on the VOC train set: MaskCLIP 47.8 → 57.3 (+9.5); BECO 66.3 → 71.8 (+5.5); CLIP-ES 70.8 → 79.3 (+8.5) — [SAMRefiner Table 4](https://arxiv.org/abs/2502.06756)
- The multimask output combined with selection by SAM's predicted IoU "falls short of the upper limit" (selecting by ground-truth IoU), because SAM's IoU head is not trained for the target categories. Choosing the mask with the highest IoU against the coarse mask works better for single point/box prompts but "performs poorly in multi-prompt cases" — [SAMRefiner, Sec. 3.3 / Fig. 5](https://arxiv.org/abs/2502.06756)
- Limitations: it "still fails to work when the initial masks are extremely noisy". There is semantic ambiguity about what a category should include (for example, items on a table). "SAM struggles to process multiple objects in semantic segmentation", and STM only partly mitigates this — [SAMRefiner, App. D.3](https://arxiv.org/abs/2502.06756)
- Runtime: refining COCO train5K (about 5K images, 37K masks) took 0.6 h for SAMRefiner, against 1.0 h CRF (16 CPU workers), 3.4 h CascadePSP, 1.5 h CRM, and 1.4 h SegRefiner, all on one RTX 3090. SAM batches several masks per image — [SAMRefiner Table 5](https://arxiv.org/abs/2502.06756)
- Code is public (linked as "SAMRefiner" in the paper) — [SAMRefiner](https://arxiv.org/abs/2502.06756)

**ClipSAM (Li et al., 2024): zero-shot anomaly segmentation, not referring segmentation**
- Pipeline: a CLIP-based rough map is binarized with a threshold. Point prompts (m random points inside connected regions) and box prompts (one per connected region) are extracted. SAM produces 3 masks with scores per box, and the final output is `Norm(O + Σ_i Σ_j m_ij · s_ij)`, a **soft fusion** of the rough CLIP map O with the score-weighted SAM masks — [ClipSAM, arXiv 2401.12665](https://arxiv.org/abs/2401.12665)
- "Employing either points or boxes alone leads to biased results ... the combined application of both points and boxes can yield more precise and detailed segmentation" — [ClipSAM, Sec. 3.3](https://arxiv.org/abs/2401.12665)
- Gain from the SAM refinement module (MMR) on MVTec-AD (AUROC / F1-max / AP / PRO): without MMR 91.8 / 46.7 / 44.7 / 84.8; full ClipSAM 92.3 / 47.8 / 45.9 / 88.3, i.e. +0.5 / +1.1 / +1.2 / +3.5. Different task, dataset, and metrics from PhraseCut. Removing their trained CLIP module (UMCI) collapses results to 60.4 AUROC, which shows SAM refinement depends on good coarse localization — [ClipSAM Table 2](https://arxiv.org/abs/2401.12665)
- Binarization threshold sensitivity: thr 0.45 / 0.47 / 0.50 gives F1-max 43.3 / 47.8 / 44.6. "A small value may cause the non-anomalous regions to be misclassified ... a large value may cause some anomalies to be ignored" — [ClipSAM Table 3](https://arxiv.org/abs/2401.12665)
- ClipSAM uses CLIP ViT-L-14-336 and trains its UMCI module on a 3090. Only the SAM stage is training-free — [ClipSAM](https://arxiv.org/abs/2401.12665)

**SAM / SAM 2 mechanics relevant to the "part mask" failure**
- SAM returns 3 masks for ambiguous prompts because "nested masks are often at most three deep: whole, part, and subpart". During training it backprops only the minimum loss over the three. Masks are ranked by a predicted-IoU head — [Kirillov et al., SAM, arXiv 2304.02643](https://arxiv.org/abs/2304.02643)
- SAM prompts: a point is a positional encoding plus a learned foreground/background embedding. A box is a pair of corner embeddings. Masks are dense prompts. The prompt encoder and mask decoder run in about 50 ms on CPU in a browser once the image embedding exists — [SAM paper](https://arxiv.org/abs/2304.02643)
- SAM 2 vs SAM on 23 image/video datasets (1-click / 5-click mIoU): SAM 58.1 (81.3), SAM 2 (SA-1B only) 58.9 (81.7), SAM 2 (full data mix) 61.9 (83.5). FPS on A100: SAM 21.7 vs SAM 2 130.1, "6× faster" — [SAM 2, arXiv 2408.00714, Table 5](https://arxiv.org/abs/2408.00714)
- SAM 2.1 checkpoints (params / FPS from the README): hiera_tiny 38.9M / 91.2; small 46M / 84.8; base_plus 80.8M / 64.1; large 224.4M / 39.5 — [facebookresearch/sam2 README](https://github.com/facebookresearch/sam2)
- HQ-SAM (ViT-L) adds less than 0.5% parameters and runs at 4.8 FPS vs SAM-L 5.0 FPS, with 7.6 GB memory for both. On four high-quality boundary datasets, mIoU averages 79.5 → 89.1 and mBIoU 71.1 → 81.8 — [Ke et al., HQ-SAM, arXiv 2306.01567, Tables 1 and 3](https://arxiv.org/abs/2306.01567)

**Grounded-SAM (detector + SAM) as a comparison point**
- Grounding DINO boxes fed to SAM give 48.7 mean AP on the zero-shot SegInW benchmark (25 datasets) with Grounding-DINO-Base + SAM-H, and 49.6 with HQ-SAM. Box prompts come from a detector, not from a coarse mask — [Ren et al., Grounded SAM, arXiv 2401.14159, Table 1](https://arxiv.org/abs/2401.14159)
- Existing open-source pipelines use CLIPSeg for "stuff" classes and Grounding DINO + SAM for "things", for example segments-ai/panoptic-segment-anything (about 438 GitHub stars). This is a demo, with no benchmark numbers — [segments-ai/panoptic-segment-anything](https://github.com/segments-ai/panoptic-segment-anything)

### Inferences
- **Recommended SAM recipe for CLIPSeg on PhraseCut** (drawn from the SAMRefiner and ClipSAM evidence above): threshold the CLIPSeg probability (keep 0.3), take connected components, then build prompts per component: a slightly expanded box, a distance-transform positive point, optionally a negative point, plus a mask prompt made from the CLIPSeg logits. Use `multimask_output=True` and choose the output by IoU against the CLIPSeg mask, not by SAM's predicted IoU. Union the components. Finally, soft-fuse (for example `p_final = α·p_clipseg + (1-α)·SAM_mask`) so the result stays a probability map for AP and for the 0.3 threshold.
- **Part vs whole ("handle of the cup"):** for part phrases a box around the whole cup tends to return the whole object, while a point inside the handle may return the handle. Choosing among the 3 masks by agreement with the coarse CLIPSeg mask mitigates this. If CLIPSeg's mask already covers the whole cup, however, SAM will make the error crisper. Gating is needed: keep CLIPSeg's mask when the IoU between the SAM and CLIPSeg masks is below a threshold, which is the same idea as CascadePSP's fallback when IoU < 0.5.
- **Multi-instance phrases (17.6% of PhraseCut):** a single tight box around all instances is the worst case. Per-component prompting is necessary (ClipSAM boxes per connected region; SAMRefiner STM).
- **Empty predictions:** PhraseCut evaluation includes cases where CLIPSeg predicts few or no foreground pixels. SAM must not be run with a box when the mask is empty, or it will hallucinate an object.
- Expected size of the gain on PhraseCut is uncertain. Published gains (+5 to +10 mIoU) are on pseudo-masks of semantic/instance segmentation with known categories. The parts of CLIPSeg's PhraseCut error that are semantic (wrong region chosen) cannot be fixed. A plausible estimate is a few mIoU points, mostly on mid and large objects and boundaries. That is an estimate, not a cited number.

### Gaps
- I found no peer-reviewed paper that reports CLIPSeg+SAM on PhraseCut, RefCOCO, or similar with an mIoU delta. "Prompt-SAM" in the sense the user mentioned could not be identified as a single primary source (web search unavailable).
- No primary source documents the "handle of the cup" part-mask failure specifically. The mechanism above (whole/part/subpart multimask) is from the SAM paper; the specific example is not verified.
- SAMRefiner's numbers use SAM ViT-H. No evidence was found for how much is lost with SAM ViT-B or SAM 2.1-tiny/small as the refiner.

## 3. Classical refinement: dense CRF, guided filter, bilateral solver, PAMR, TTA, sliding window / higher resolution

### Takeaway
Classical refiners give small, unreliable gains. Dense CRF can even hurt: it was below the coarse mask on 3 of 6 settings in SAMRefiner's comparison. PAMR gives about +1.4 to +3.5 mIoU on average for training-free CLIP segmenters. Sliding-window or higher-resolution inference helps CLIP-based dense prediction by roughly 1 to 3 mIoU, but CLIPSeg's own paper warns that quality drops above about 350 px when the positional embedding is interpolated. Sliding windows also break global context, which referring phrases need. Flip TTA is cheap, but for "left/right" phrases a horizontal flip changes the meaning, so the prompt must be flipped too or those samples excluded.

### Cited Findings
- **Dense CRF:** in the comparison on 6 sources (coarse mask → CRF): DAVIS-585 81.4 → 81.0; COCO NB 15.2 → 13.9; PointWSSIS 29.1 → 24.4; MaskRCNN 35.2 → 31.5; VOC MaskCLIP 47.8 → 48.2; BECO 66.3 → 66.5; CLIP-ES 70.8 → 72.6. "CRF shows inferior performance due to the lack of high-level semantic context and unfitness for the binary mask." Runtime was 1.0 h for COCO train5K with 16 CPU workers — [SAMRefiner Table 5](https://arxiv.org/abs/2502.06756)
- SCLIP and TCL do not allow dense CRF post-processing for fair comparison because of its "very heavy computation cost" — [SCLIP, arXiv 2312.01597](https://arxiv.org/abs/2312.01597)
- **PAMR** (Pixel-Adaptive Mask Refinement) is a parameter-free recurrent module that propagates labels with local RGB affinities and multiple dilated 3×3 kernels. Each iteration adds less than 1% GPU footprint, and 10 iterations is the default. The configuration used is dilations [1,2,4,8,12,24] — [Araslanov & Roth, arXiv 2005.08104](https://arxiv.org/abs/2005.08104)
- PAMR gains for CLIP-based zero-shot segmentation: TCL average mIoU over 8 benchmarks 33.9 → 37.2 (+3.3; VOC 51.2 → 55.0); GroupViT +1.4 average; SCLIP 38.0 → about 40 (+1.9 average); MaskCLIP +3.5 after tuning PAMR hyperparameters. TCL had reported MaskCLIP degrading with PAMR. SCLIP calls PAMR "computationally intensive" and recommends against it by default — [SCLIP Table 1 / Sec. 4.2](https://arxiv.org/abs/2312.01597); [TCL, arXiv 2212.00785, Table 3](https://arxiv.org/abs/2212.00785)
- **Sliding window / resolution** for CLIP ViT-B dense inference (SCLIP, PASCAL VOC21 / Context59 / COCO-Stuff): 224 short side, 224 window, 112 stride (1× FLOPs) = 56.5 / 32.0 / 20.5; 336 short side, 224 window, 112 stride (4× FLOPs) = 59.1 / 34.2 / 21.9 (+2.6 / +2.2 / +1.4); 448 short side, 336 window = 58.6 / 34.1 / 22.1. A denser stride gives about +0.2 mIoU on Context — [SCLIP Table 4](https://arxiv.org/abs/2312.01597)
- CLIPSeg: interpolating positional embeddings allows larger images, but for ViT-B/16 "performance only decreases for images larger than 350 pixels" — [CLIPSeg paper](https://arxiv.org/abs/2112.10003)
- **Guided filter / bilateral solver:** no primary source with CLIP-segmentation numbers was retrieved in this session.
- **Flip / multi-scale TTA for CLIPSeg:** no primary source with numbers was retrieved. SCLIP and TCL use multi-scale/slide protocols as part of their standard evaluation, not as a separately ablated TTA gain — [SCLIP](https://arxiv.org/abs/2312.01597)

### Inferences
- Classical options are cheap, but on the evidence above they are unlikely to clearly beat a baseline on PhraseCut miniv. PAMR is the most defensible of them (training-free, about 20 lines of PyTorch, sub-second on GPU, and its probabilistic output keeps AP valid). SCLIP and TCL already show it helps CLIP-based maps.
- Multi-scale TTA (for example 352 + 448 + flip, averaging probabilities) is about 3 forward passes of a small model. It is very cheap on a T4 and keeps outputs as probabilities, so the threshold and AP stay comparable. Prompts containing "left" or "right" must be either excluded from flip-TTA or have the words swapped.
- Sliding windows (crops) are risky for referring expressions: each crop sees only part of the scene, so "the man on the left" or "second car" loses context, and a crop without the target will still produce false positives. Global full-image inference at a moderately higher resolution (for example 448) is safer but runs into CLIPSeg's >350 px degradation note. This has to be measured, not assumed.

### Gaps
- No primary-source numbers on guided filter or the fast bilateral solver applied to CLIP/CLIPSeg masks.
- No published CLIPSeg-specific TTA ablation (flip, multi-scale) on PhraseCut was found.
- The exact CLIPSeg size-vs-quality curve on PhraseCut (beyond the qualitative ">350 px decreases") is in the supplement figure only as a classification proxy (Fig. 5), not as a segmentation number.

## 4. Learned lightweight refinement heads usable without retraining CLIPSeg (CascadePSP, SegRefiner, HQ-SAM, etc.)

### Takeaway
CascadePSP is class-agnostic, pip-installable (`segmentation-refinement`), works "without finetuning" on new models, and gives about +1.3 to +3.9 IoU (+8 to +12 mBA) on PASCAL VOC semantic segmentation. However, it has no semantics and can follow color edges into wrong regions. On instance-style data it was at or below the coarse masks. SegRefiner is more accurate on the BIG dataset but is a diffusion model costing about 2.9 s per image for 6 steps, which is too slow for a full evaluation sweep on a T4 within the time budget. SAM-based refinement (Section 2) beats both in SAMRefiner's head-to-head comparison.

### Cited Findings
- **CascadePSP:** trained class-agnostically on MSRA-10K, DUT-OMRON, ECSSD and FSS-1000 (36,572 images, over 1,000 classes). It refines outputs of segmentation models "without finetuning" and uses a global step and a local step. Memory depends only on L: L=900 uses 3.16 GB GPU, L=500 uses 1.16 GB at −0.6% mBA. Refining the paper's Figure 1 (very high resolution) takes about 6.6 s — [Cheng et al., arXiv 2005.02551](https://arxiv.org/abs/2005.02551)
- CascadePSP on PASCAL VOC 2012 (IoU / mBA): FCN-8s 68.85 / 54.05 → 72.70 / 65.36 (+3.85 / +11.3); DeepLabV3+ 87.13 / 61.68 → 89.01 / 72.10 (+1.88 / +10.4); PSPNet 90.92 / 60.51 → 92.86 / 72.24. On ADE20K the gains are only +0.73 to +0.99 mIoU — [CascadePSP Tables 3 and 4](https://arxiv.org/abs/2005.02551)
- CascadePSP failure case: "it produces a wrong segmentation due to the lack of semantic information". It follows color boundaries even when the input region is semantically wrong. For scene parsing it falls back to the original segmentation when IoU < 0.5 — [CascadePSP, Fig. 11 / Sec. 4](https://arxiv.org/abs/2005.02551)
- In SAMRefiner's comparison, CascadePSP (PSP): DAVIS-585 81.4 → 81.9; VOC MaskCLIP 47.8 → 55.3; CLIP-ES 70.8 → 76.9; but COCO PointWSSIS 29.1 → 28.9 and MaskRCNN 35.2 → 34.6. Improvement is "limited or even worse than coarse masks on instance segmentation" — [SAMRefiner Table 5](https://arxiv.org/abs/2502.06756)
- PyPI package `segmentation-refinement` v0.6 ("Deep learning based segmentation refinement system", home page github.com/hkchengrex/CascadePSP) — [PyPI](https://pypi.org/project/segmentation-refinement/)
- **SegRefiner (NeurIPS 2023):** treats refinement as discrete diffusion with T=6 steps. On BIG (IoU / mBA): average improvement +7.43 / +17.33 vs CascadePSP +3.58 / +14.05 and CRM +4.01 / +15.12 — [Wang et al., arXiv 2312.12425, Table 1](https://arxiv.org/abs/2312.12425)
- SegRefiner cost (instance segmentation): 1 step 0.52 s / 249 GFLOPs; 6 steps 2.94 s / 1,492 GFLOPs per image; mask AP 39.8 → 41.9. "The diffusion process leads to slowdown of the inference" — [SegRefiner Table 6](https://arxiv.org/abs/2312.12425)
- SegRefiner's "performance is not stable across different settings" (for example DAVIS-585 81.4 → 80.3, below the coarse mask) — [SAMRefiner Table 5](https://arxiv.org/abs/2502.06756)
- **HQ-SAM** (a learned SAM add-on, trained for 4 h on 8×3090) is a drop-in, higher-boundary-quality SAM: +9.6 mIoU / +10.7 mBIoU on average across 4 high-quality datasets vs SAM. Grounded-HQ-SAM gives 49.6 vs Grounded-SAM 48.7 SegInW AP — [HQ-SAM](https://arxiv.org/abs/2306.01567); [Grounded SAM](https://arxiv.org/abs/2401.14159)

### Inferences
- CascadePSP is the simplest learned head to try (`pip install segmentation-refinement`, one call per image plus mask). Its output is a refined soft or binary map, so AP is still computable if the soft output is used. Its class-agnostic, color-edge behavior suits PhraseCut's object and stuff regions, but it may hurt multi-instance and small-object cases.
- SegRefiner is ruled out on budget: about 3 s per image × miniv size × several ablations, plus mmdet-style dependencies.
- HQ-SAM is mainly interesting if SAM refinement is chosen. It is not in HF transformers as far as I could verify (not checked), so it would require the original repo.

### Gaps
- Exact CascadePSP and SegRefiner runtime on a T4 was not found. CascadePSP's 6.6 s figure is for a very high-resolution image; PhraseCut images (Visual Genome) are much smaller.
- No published result of CascadePSP or SegRefiner on CLIPSeg outputs or on PhraseCut.

## 5. Feasibility on Kaggle T4, availability (transformers 4.55.4 / pip), effort, and metric comparability

### Takeaway
SAM v1 (`SamModel`/`SamProcessor`, with `input_points`, `input_boxes`, `input_masks`) is available in transformers 4.55.4. **SAM 2 / 2.1 (`Sam2Model`) is not in 4.55.4.** Its docs first appear in v4.56.0, so using it means upgrading transformers (which risks changing the CLIPSeg baseline environment) or installing Meta's `sam2` package. The best single E4 candidate within 2 weeks and the T4 budget is "SAM (ViT-B or ViT-L) refinement with mask-derived box + point + mask prompts and soft fusion". The cheaper fallbacks are PAMR or multi-scale/flip TTA. For any binary-output refiner, AP is no longer comparable unless the output is fused back into a probability map.

### Cited Findings
- transformers 4.55.4 was uploaded to PyPI on 2025-08-22 — [PyPI transformers 4.55.4](https://pypi.org/project/transformers/4.55.4/)
- The HF docs page `v4.55.4/en/model_doc/sam2` contains no `Sam2Model` references, while `v4.56.0/en/model_doc/sam2` contains 23. So SAM2 was added in 4.56.0 (verified by fetching both pages) — [HF docs v4.56.0 SAM2](https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/sam2); [HF docs v4.55.4 SAM2](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/sam2)
- The HF docs `v4.55.4/en/model_doc/sam` page exists and documents `SamModel` with an `input_masks` argument — [HF docs v4.55.4 SAM](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/sam)
- The current SAM2 HF API: `Sam2Processor`/`Sam2Model.from_pretrained("facebook/sam2.1-hiera-large")` (also `-tiny` etc.), with `input_points`, `input_labels` (1 = foreground, 0 = background, −10 = padding), `input_boxes`, and `model(..., input_masks=mask_input, image_embeddings=..., multimask_output=False)` for mask-prompt refinement. Batched multi-box prompts are supported — [HF SAM2 docs](https://huggingface.co/docs/transformers/model_doc/sam2)
- SAM 2.1 sizes and A100 FPS (README): tiny 38.9M / 91.2 FPS; small 46M / 84.8; base+ 80.8M / 64.1; large 224.4M / 39.5. SAM (v1, ViT-H) is 21.7 FPS vs SAM 2 130.1 FPS on A100 in the paper's image-task benchmark — [sam2 README](https://github.com/facebookresearch/sam2); [SAM 2 paper Table 5](https://arxiv.org/abs/2408.00714)
- SAM-L inference memory is 7.6 GB at 5.0 FPS (HQ-SAM's measurement; their GPU setting for inference is not stated in the extracted table) — [HQ-SAM Table 1](https://arxiv.org/abs/2306.01567)
- The CascadePSP pip package exists (`segmentation-refinement` 0.6). `pydensecrf2` 1.1 on PyPI ships only a source tarball and a macOS wheel, so on Kaggle Linux it must be compiled from source — [PyPI segmentation-refinement](https://pypi.org/project/segmentation-refinement/); [PyPI pydensecrf2](https://pypi.org/project/pydensecrf2/)
- CLIPSeg's AP uses fixed thresholds with TP/FP sums per threshold, which assumes a continuous score map — [CLIPSeg supplement](https://arxiv.org/abs/2112.10003)
- ClipSAM keeps a continuous output by normalizing `O + Σ m·s` (the CLIP map plus score-weighted SAM masks) — [ClipSAM Eq. 9](https://arxiv.org/abs/2401.12665)

### Inferences (T4 numbers below are estimates, not measured or cited)
| Option | Availability | Effort (1 person) | T4 cost (estimate) | Metric comparability |
|---|---|---|---|---|
| SAM v1 ViT-B/L refinement (box + point + mask prompts, per-component, soft fusion) | `SamModel` in 4.55.4 (`facebook/sam-vit-base`, `-large`, `-huge`) | 2–4 days incl. prompt-design ablation | ViT-B encoder is about 0.1–0.2 s/img on a T4, ViT-H a few times slower. Fits easily for miniv. Fits in 16 GB (SAM-L 7.6 GB reported). | Fine if fused into a probability map; binary SAM output makes AP degenerate and makes t=0.3 irrelevant |
| SAM 2.1 (tiny/small/base+) refinement | Needs transformers ≥4.56 (`Sam2Model`) or Meta `sam2` pip/repo (the CUDA-kernel compile is optional) | Same as above plus an environment-version risk | Faster than SAM v1 (6× on A100) | Same as above |
| PAMR on CLIPSeg logits | ~20 lines of PyTorch (code in visinf/1-stage-wseg) | <1 day | Negligible | Keeps probabilities, so fully comparable |
| Multi-scale + flip TTA | Native | <1 day | 2–4× CLIPSeg forward (cheap) | Comparable; "left/right" phrases need the prompt swapped |
| Higher-res / sliding window | Native (CLIPSeg interpolates pos-emb) | 1 day | 2–4× | Comparable; risks the >350 px degradation and loss of global context |
| Dense CRF | pydensecrf must be compiled; CPU | 1 day | CPU-bound, slow | Outputs marginals so AP stays possible; evidence of gains is weak or negative |
| CascadePSP | `pip install segmentation-refinement` | 1 day | Seconds per image at high resolution; fewer for VG sizes (unmeasured) | Soft output is available |
| SegRefiner | Repo (mmdet stack) | 3+ days | About 3 s/img (6 steps) | Binary-ish; **does not fit the 2-week budget well** |
| SAMRefiner(++) full | Repo (SAM ViT-H) | 3–5 days; ++ needs LoRA training | ViT-H on a T4 is heavy but feasible for miniv | Binary output plus fusion needed |

- **Comparability rules for E4:**
  1. Keep the CLIPSeg baseline's threshold (0.3) applied to the *final* probability map.
  2. Compute AP on the fused probability map, not on a binary SAM mask. Otherwise report AP as "not applicable" and do not compare it.
  3. Report the same mIoU and cum-IoU over the identical miniv sample list, including samples where SAM is gated off.
  4. Report gating rate and runtime.
  5. Tuning α, the gate threshold, or the prompt type on miniv itself is test-set tuning. Tune on a held-out subset of the PhraseCut train/val split, or fix the hyperparameters a priori from SAMRefiner's defaults (box expansion and mask thresholds 0.1/0.5, Gaussian σ factors 15 and 4).
- **Prerequisite (must have):** a gate for empty or near-empty CLIPSeg masks, and per-connected-component prompting for multi-instance phrases.
- **Recommendation for one E4 module:** SAM v1 ViT-B via `SamModel` (no environment change), with box + distance-guided point + mask prompts per component, best-of-3 selected by IoU with the CLIPSeg mask, and soft fusion. Run PAMR as a cheap secondary ablation or fallback if SAM does not beat the baseline.

### Gaps
- No measured T4 latency for SAM ViT-B/L/H or SAM 2.1 in HF transformers was found. All T4 timings in the table are estimates and should be measured with a 50-image dry run.
- I did not verify whether `facebook/sam2.1-*` HF checkpoints load in transformers 4.56 without other dependency changes, or whether upgrading transformers changes the CLIPSegForImageSegmentation outputs. Both need checking if SAM 2 is chosen.
- HQ-SAM availability in HF transformers was not checked.
