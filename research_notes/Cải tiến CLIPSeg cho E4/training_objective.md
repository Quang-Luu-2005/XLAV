# Training-objective, data and prompt-side improvements for CLIPSeg (E4 candidates)

Method note: tool access was limited. WebSearch/WebFetch failed during this session. All findings below come from primary PDFs downloaded directly from arXiv and from the timojl/clipseg GitHub repo's raw files, read as full text. No secondary blogs were used. Several key questions (SAM distillation for CLIPSeg, hard-negative ablations, LLM-generated descriptions for CLIPSeg) could not be searched broadly and are listed as gaps.

Baseline reference numbers (CLIPSeg paper, PhraseCut test, threshold t=0.3, mIoU / IoU_FG / AP):
- CLIPSeg (PC+): 43.4 / 54.7 / 76.7. CLIPSeg (PC, text-only, no negatives): 46.1 / 56.2 / 78.2. CLIPSeg (PC, D=128): 48.2 / 56.5 / 78.2. — [CLIPSeg, Table 3](https://arxiv.org/abs/2112.10003)
- Official training: AdamW, lr 1e-3, cosine to 1e-4, no warmup, 20,000 iterations, batch 64, AMP, "binary cross entropy as the only loss function", 352×352 input. — [CLIPSeg appendix](https://arxiv.org/abs/2112.10003); [experiments/phrasecut.yaml](https://github.com/timojl/clipseg/blob/master/experiments/phrasecut.yaml)

---

## 1. Loss functions: BCE vs BCE+Dice, focal, Lovász-hinge, Tversky, boundary losses

### Takeaway
Adding a region/overlap loss (Dice or Lovász-hinge) to BCE is the standard choice in recent text-prompted segmenters (LISA, SAM, medical CLIPSeg fine-tuning). The most direct evidence is that Lovász improves mIoU by about 1–2 points over cross-entropy on Pascal VOC. However, no paper found reports a controlled BCE-vs-BCE+Dice ablation on PhraseCut or CLIPSeg itself, so the size of the gain for E4 is unknown. One caution: Dice/Lovász are per-image IoU surrogates and are undefined or degenerate on CLIPSeg's 20% empty-mask negatives, so they must be masked or handled carefully.

### Cited Findings
- CLIPSeg uses only `binary_cross_entropy_with_logits` in every config (phrasecut.yaml, ablation.yaml). The authors never ablated the loss. — [phrasecut.yaml](https://github.com/timojl/clipseg/blob/master/experiments/phrasecut.yaml); [ablation.yaml](https://github.com/timojl/clipseg/blob/master/experiments/ablation.yaml)
- CRIS's "text-to-pixel contrastive loss" (Eq. 9–10) is mathematically per-pixel sigmoid BCE on dot products between text and pixel features, averaged over all pixels. It is the same form as CLIPSeg's loss. CRIS's reported gain of +1.98 / +2.98 / +3.43 IoU on RefCOCO / RefCOCO+ / G-Ref comes from adding this pixel-text alignment head to a baseline without it, not from switching the loss type. — [CRIS, Sec 3.3 & Table 1](https://arxiv.org/abs/2111.15174)
- LAVT is trained with plain cross-entropy only (AdamW, lr 5e-5, 40 epochs, batch 32). It reaches 72.73 oIoU on RefCOCO val, so strong referring-segmentation results do not require Dice. — [LAVT, Sec 4](https://arxiv.org/abs/2112.02244)
- LISA's mask loss = λ_bce·BCE + λ_dice·DICE with λ_bce = 2.0 and λ_dice = 0.5. — [LISA, Eq. 4 & Sec 5](https://arxiv.org/abs/2308.00692)
- SAM's mask loss is a linear combination of focal loss and dice loss in a 20:1 focal:dice ratio. — [SAM, Sec 3 & appendix](https://arxiv.org/abs/2304.02643)
- Medical CLIPSeg fine-tuning (MIDL 2024) used Dice loss plus BCE scaled by 0.2. CLIPSeg lr was 2e-3 and CRIS lr 2e-5, with LR ÷10 on a 5-epoch plateau. The paper does not ablate the loss. — [Poudel et al., Exploring Transfer Learning in Medical Image Segmentation using VLSMs, arXiv 2308.07706](https://arxiv.org/abs/2308.07706)
- The VLSM-Adapter paper for CLIPSeg/CRIS also uses L = λ_d·L_Dice + λ_ce·L_BCE (Eq. 2), batch 32, adapter lr 1e-3 / 3e-4, and no loss ablation. — [VLSM-Adapter, arXiv 2405.06196](https://arxiv.org/abs/2405.06196)
- Lovász-Softmax vs cross-entropy, DeepLab-v2 on Pascal VOC val mIoU:
  - single-scale: 74.64 → 76.56 (+1.9)
  - multi-scale + CRF: 76.53 → 77.99
  - with equibatch and 30K iterations: 74.97 → 77.41 (single-scale)
  — [Berman et al., Lovász-Softmax, Table 2](https://arxiv.org/abs/1705.08790)
- For binary segmentation, Berman et al. fine-tune a multi-class net into per-class binary models. They report that the Lovász hinge gives the lowest Jaccard loss across Pascal VOC binary categories (Table A.1). On a synthetic binary task, a model trained with Lovász hinge reached a Jaccard-hinge validation loss of 5.44, versus 7.11–8.37 for the other loss/metric pairings. — [Lovász paper, Sec 4.2, Table 1, App. A](https://arxiv.org/abs/1705.08790)
- Boundary loss (Kervadec et al.), added to Generalized Dice:
  - about 13% DSC improvement on ISLES over GDL alone
  - about 3% over UNet cross-entropy or focal alone
  - Hausdorff-style losses add about 10% training slowdown
  - It requires precomputed distance maps and a scheduled weight α.
  — [Boundary loss, arXiv 1812.07032](https://arxiv.org/abs/1812.07032)
- Jadon's survey compares about 15 losses (BCE, weighted CE, focal, Dice, Tversky, focal-Tversky, log-cosh Dice…) on one small skull-segmentation dataset. All Dice scores fall in about 0.94–0.97, and the authors' own log-cosh Dice is marginally best. The survey's guidance table says BCE "works best in equal data distribution among classes scenarios" and focal/Tversky-type losses suit imbalanced data. — [Jadon 2020, Tables II–III](https://arxiv.org/abs/2006.14822)
- CLIPSeg itself reports a clear trend toward better performance on larger objects (appendix Fig. 8). This is the small-object weakness that region-based losses target. — [CLIPSeg appendix](https://arxiv.org/abs/2112.10003)

### Inferences
- **BCE + λ·Dice (soft Dice computed only over positive samples) is the cheapest loss change.** It takes about 10 lines in a HF training loop, adds negligible GPU cost, and works as a short fine-tune from the reproduced checkpoint (e.g. 3–5k iterations at the reduced cosine LR).
  - Expected direction: IoU_FG and mIoU at a fixed threshold should rise, because Dice pushes calibration toward a sharper foreground. AP may move little or even drop, since AP is threshold-free and BCE is already a proper scoring rule.
  - Evidence for size: indirect (Lovász +1.9 mIoU on VOC; universal adoption in LISA/SAM/VLSM).
- **Handling negatives.** Because 20% of PC+ samples have empty masks, apply Dice/Lovász only to samples with a non-empty mask, or use a batch-level Dice (sum intersections and unions across the batch). Otherwise the loss on negatives is either 0/0 or pushes toward all-zero predictions.
- **Focal and Tversky** mainly help extreme pixel imbalance. PhraseCut objects are often small, so focal (γ=2, SAM-style with Dice) is a reasonable alternative. There is no direct CLIPSeg evidence.
- **Boundary/Hausdorff losses** need distance-map precomputation per crop. With random crops this means on-the-fly computation via scipy's distance transform (CPU cost). This is low priority within a 2-week budget.
- **Threshold confound.** The metric uses t=0.3, which was tuned for BCE-trained calibration (CLIPSeg Table 3). A Dice-trained model shifts calibration. Report both t=0.3 and best-t to stay fair, since Table 9 of the CLIPSeg paper uses the best threshold per model.

### Gaps
- No controlled BCE vs BCE+Dice / focal / Lovász ablation on PhraseCut or any CLIPSeg variant was found.
- Web search was unavailable, so papers like "loss ablations for RIS" (e.g. ReLA/GRES, PolyFormer) were not checked for loss tables.

---

## 2. Negative sampling: rate, hard negatives, effect on AP and false positives

### Takeaway
On PhraseCut text-prompt metrics, CLIPSeg variants trained without negatives (and without visual prompts) score higher than PC+ (46.1 vs 43.4 mIoU, 78.2 vs 76.7 AP at t=0.3). Negatives and visual mixing are what make the model "universal", but they cost a few points on the PhraseCut benchmark. The authors ran negative_prob = 0.0 and 0.5 ablations in their repo but did not publish the numbers. Hard negatives (deceptive, image-related phrases) are the design used by GRES, but no numbers isolate their effect.

### Cited Findings
- PC+ replaces the phrase with a different phrase with probability q_neg. The final model uses 20% negatives. — [CLIPSeg Sec 3.1, Sec 5.1](https://arxiv.org/abs/2112.10003)
- Results on PhraseCut test (t=0.3, mIoU / IoU_FG / AP):
  - PC+ (negatives + visual + interpolation): 43.4 / 54.7 / 76.7
  - PC (no negatives, text only): 46.1 / 56.2 / 78.2
  — [CLIPSeg Table 3](https://arxiv.org/abs/2112.10003)
- The ablation config `ablation.yaml` lists `rd64-no-negatives (negative_prob: 0.0)` and `rd64-neg0.5 (negative_prob: 0.5)` runs, but these are not reported in the paper. — [ablation.yaml](https://github.com/timojl/clipseg/blob/master/experiments/ablation.yaml)
- One-shot on COCO-20i (Table 6): CLIPSeg (COCO) gets 33.2 mIoU; CLIPSeg (COCO+N, 10% negatives) gets 33.3 mIoU. Negatives were roughly neutral on mIoU there, and the other columns shift by a few points in mixed directions. Different dataset and protocol. — [CLIPSeg Table 6](https://arxiv.org/abs/2112.10003)
- GRES/gRefCOCO adds 32,202 no-target expressions with two rules:
  1. The expression must not be totally irrelevant to the image (e.g. "the kid in blue" when kids exist but none in blue; "dog" is disallowed if no dog).
  2. Otherwise, use a "deceptive expression drawn from other images" in the same split.
  The authors note unconstrained annotators write trivial negatives like "dog" for dog-free images, which they call "undesirable and purposeless". — [GRES, arXiv 2306.00968, Sec 3](https://arxiv.org/abs/2306.00968)
- GRES evaluates no-target handling with N-acc / T-acc. The paper states that classic RES models "are likely to output an instance incorrectly" for no-target expressions. — [GRES](https://arxiv.org/abs/2306.00968)
- Red-circle visual prompting finds that subtracting the score averaged over negative (other) queries improves REC. In its Table 9 ablation (RN50x16 + ViT-L/14@336, top-1 accuracy), RefCOCO val goes 42.01 → 43.67 with subtraction and → 49.84 with subtraction plus marker ensemble. This is an inference-time analog of contrasting against negatives. — [Shtedritski et al., arXiv 2304.06712, Table 9](https://arxiv.org/abs/2304.06712)

### Inferences
- If E4's goal is purely to "beat the reproduced baseline on PhraseCut text metrics", **lowering q_neg (e.g. 0.2 → 0.05–0.1) is a cheap data change.** It is a one-line config change, works as a fine-tune, and costs no extra GPU. Paper evidence points in that direction (PC > PC+), but PC also drops visual prompts, so the effect is confounded.
- **Hard negatives** (phrases from other objects in the same image that do not overlap the target, or phrases sharing the head noun but a different attribute) should reduce false positives. They may raise AP, since AP penalizes confident FP pixels.
  - Implementation: PhraseCut annotations already contain all phrases per image, so the dataset `__getitem__` changes by about 20–40 lines.
  - There is no numeric evidence of the gain, and the PhraseCut test set has no negative samples, so FP reduction only shows in AP/mIoU indirectly.
- Raising q_neg probably hurts PhraseCut metrics, because the test set contains no negatives.

### Gaps
- No published numbers for CLIPSeg with q_neg = 0 vs 0.2 vs 0.5 (the runs exist in the repo, but results are not public).
- No paper found that ablates hard vs random negatives for referring segmentation with numbers.

---

## 3. Visual-prompt engineering during training (k-shot, support processing, red circle, interpolation schedule)

### Takeaway
CLIPSeg already uses the best of its tested support-image transforms: crop + 10% background intensity + background blur, which gives +23.50 object probability, the highest in Table 2. Joint visual training barely helps text-prompt PhraseCut results; the paper calls the gap "negligible". Red circles beat cropping for CLIP-based zero-shot REC, but CLIPSeg's own experiment found a red outline weak (+2.29). There is no evidence that k-shot averaging or a different interpolation schedule improves text-prompt PhraseCut scores.

### Cited Findings
- CLIPSeg Table 2: average improvement of CLIP object probability over 1,600 samples, by support-image transform.

  | Transform | Improvement |
  |---|---|
  | dye object red in grayscale image | +1.21 |
  | add red object outline | +2.29 |
  | BG intensity 10% | +13.85 |
  | BG intensity 0% | +23.40 |
  | BG blur | +13.15 |
  | crop | +13.60 |
  | crop & BG intensity 10% | +21.73 |
  | crop + BG intensity 10% + BG blur | **+23.50** |
  | crop large context | +6.27 |

  — [CLIPSeg Table 2](https://arxiv.org/abs/2112.10003)
- CLIPSeg Table 9 ablation (best threshold per model; text-based mIoU / AP, then visual-based mIoU / AP). Row alignment was reconstructed from the PDF text layout, so verify against the PDF:

  | Variant | Text mIoU / AP | Visual mIoU / AP |
  |---|---|---|
  | CLIPSeg (PC+) | 43.6 / 76.7 | 25.4 / 55.6 |
  | no CLIP pre-training | 13.1 / 12.6 | 12.7 / – |
  | no visual | 46.4 / 77.8 | 14.4 / 31.0 |
  | D=16 | 37.4 / 71.5 | 24.7 / 51.2 |
  | only layer 3 | 31.9 / 64.9 | 21.5 / 48.6 |
  | highlight mask | 43.4 / 75.4 | 23.3 / 43.8 |

  The paper's text says "the gap in text-based performance to the hybrid version (PC+) is negligible". — [CLIPSeg Table 9, Sec 5.5](https://arxiv.org/abs/2112.10003)
- Interpolation in the paper is x = a·s + (1−a)·t with a ~ U[0,1], used "as a data augmentation strategy". The released configs use `mix: True, mix_text_min: 0.0, mix_text_max: 0.5`. — [CLIPSeg Sec 3](https://arxiv.org/abs/2112.10003); [phrasecut.yaml](https://github.com/timojl/clipseg/blob/master/experiments/phrasecut.yaml)
- Red circle (zero-shot REC with CLIP, top-1 accuracy, RefCOCO val/testA/testB):
  - Red Circle: 49.8 / 58.6 / 39.9
  - ReCLIP crop-based: 42.0 / 43.5 / 39.0
  - ReCLIP with relations: 45.8 / 46.1 / 47.1
  - On RefCOCOg: 59.4 / 58.9 vs ReCLIP 57.8 / 57.2
  The authors find marking "significantly more effective than cropping", and ensembling markers (circle, circle + blur outside, circle + grey outside) adds about 6 points. — [Red circle, arXiv 2304.06712, Tables 5 & 9](https://arxiv.org/abs/2304.06712)
- Different task caveat: red circle is evaluated as zero-shot region ranking with large CLIP models (ViT-L/14@336, RN50x16), which the authors say matter. They report that the smallest model's improvements are much weaker. — [Red circle](https://arxiv.org/abs/2304.06712)

### Inferences
- **Support processing is already near-optimal per CLIPSeg's own measurement.** Switching to a red circle is not supported: the red outline scored +2.29 vs +23.50 in CLIPSeg's ViT-B/16 test, and red-circle gains come from large CLIP models.
- **Training-time k-shot averaging** (averaging CLIP embeddings of k≥2 support crops) is easy: a few lines, and k extra CLIP forward passes per sample, which is cheap since support images are small and the visual encoder is frozen. It would mainly help visual-prompt evaluation (E2), not text PhraseCut.
- **Interpolation schedule** (e.g. annealing w toward pure text late in training, or w ~ U(0, 0.25)) could recover part of the PC vs PC+ text gap (43.4 → 46.1 mIoU). This is speculative. It is cheap (one line) and fine-tunable.

### Gaps
- No paper found that ablates k-shot support averaging during CLIPSeg training, or the interpolation range (w max 0.5 vs 1.0 vs annealed).
- Red-circle training of a segmentation decoder (rather than zero-shot ranking) was not found.

---

## 4. Text-prompt side: learned prompts (CoOp/CoCoOp/DenseCLIP), ensembles, LLM descriptions

### Takeaway
Learnable context tokens give modest gains in dense prediction: +0.6 mIoU on ADE20K in DenseCLIP's ablation (42.9 → 43.5). They give large gains in few-shot classification (CoOp: about +15% average at 16 shots). For CLIPSeg the text encoder is frozen and the decoder already conditions via FiLM, so a learned prefix is feasible but the expected gain on PhraseCut is small. Medical CLIPSeg work finds that richer prompts barely matter after fine-tuning.

### Cited Findings
- CoOp (16-shot, 11 classification datasets): learned context vectors beat hand-crafted prompts "by around 15%" on average, "with the highest reaching over 45%". — [CoOp, arXiv 2109.01134](https://arxiv.org/abs/2109.01134)
- CoCoOp: CoOp's learned context "is not generalizable to wider unseen classes". CoCoOp's instance-conditional prompts generalize "much better than CoOp to unseen classes" in a base-to-new setting. — [CoCoOp, arXiv 2203.05557](https://arxiv.org/abs/2203.05557)
- DenseCLIP Table 7 (ADE20K semantic segmentation, ResNet-50): hand template "a photo of a [CLS]." gives 42.9 mIoU; CoOp-style learnable contexts give 43.5 mIoU (+0.6). Context length is 8. Its auxiliary pixel-text loss is BCE on sigmoid(score/τ) (Eq. 8). — [DenseCLIP, arXiv 2112.01518](https://arxiv.org/abs/2112.01518)
- DenseCLIP's headline gains are larger: "+3.9%, +2.4% and +1.2% mIoU improvement compared to vanilla fine-tuning". These come from fine-tuning the whole backbone with context-aware prompting, not from prompts alone. That is not applicable with CLIPSeg's frozen CLIP. — [DenseCLIP Sec 1](https://arxiv.org/abs/2112.01518)
- CLIPSeg appendix: "a small effect on performance for alternative prompt forms" when evaluating on 5,000 PhraseCut test samples with a model trained on randomly sampled prompt forms. — [CLIPSeg appendix, Fig. 7](https://arxiv.org/abs/2112.10003)
- Medical VLSM fine-tuning: "Making prompts richer does not always help during finetuning"; DSC "variation across prompt type is minimal in the finetuned setting". Replacing attribute values with random ones at inference changed CLIPSeg very little, "reinforcing the minimal impact of attributes in CLIPSeg". — [arXiv 2308.07706](https://arxiv.org/abs/2308.07706)
- Red-circle REC ensembles across marker types add about 6 points top-1 on RefCOCO val (43.67 → 49.84). This is a prompt ensemble at test time, with a different task and model. — [arXiv 2304.06712, Table 9](https://arxiv.org/abs/2304.06712)

### Inferences
- **CoOp-style learnable prefix for CLIPSeg in HF.** `CLIPSegForImageSegmentation` exposes `conditional_embeddings`, so one can compute text embeddings with a custom forward through the frozen `clip.text_model`, prepending n learnable token embeddings (n=4–8) after BOS.
  - Effort: about 50–100 lines, since it must bypass the tokenizer → embedding path.
  - Cost: backprop through the 12-layer text transformer. The text encoder is only about 63M params at sequence length 77, so a moderate increase.
  - Fine-tunable from the checkpoint.
  - Expected gain on free-form PhraseCut phrases: small (DenseCLIP +0.6 for class names). PhraseCut is open-vocabulary, so the CoCoOp-style overfitting risk is lower but nonzero.
- **Test-time prompt ensembling** (averaging conditional embeddings over the 9 templates, or averaging the output logits) needs zero training and 9× text-encoder cost at eval only. This is likely the easiest "free" gain, but it is an inference change. Whether it counts as an E4 "training" change is a project decision.
- LLM-generated descriptions: medical evidence suggests minimal effect after fine-tuning. No PhraseCut evidence found.

### Gaps
- No paper found applying CoOp/learned context to CLIPSeg on PhraseCut, or LLM-generated descriptions for CLIPSeg training. Web search was unavailable to check CLIPSeg follow-ups.
- CoOp exact per-dataset numbers were not extracted, only the abstract-level ~15% / >45% claims.

---

## 5. Augmentation, resolution, training length / LR

### Takeaway
The released "refined" CLIPSeg (`rd64-uni-refined`, the HF `CIDAS/clipseg-rd64-refined` lineage) trains 50k iterations instead of 20k and uses a more complex transposed-conv head. Decoder width also matters a lot: D=128 trained on PC gets 48.2 vs 46.1 mIoU for D=64. ViT-B/16 CLIP performs best around 350 px, which is why 352 is used. Larger input helps CLIP classification only up to about 350 px. The paper uses no color jitter or multi-scale training.

### Cited Findings
- Refined config: `rd64-uni-refined`: `complex_trans_conv: True, T_max: 50000, max_iterations: 50000`, otherwise like rd64-uni. The comment says it is "almost the same training setting as the refined model except for transformer dropout of 0.1". — [phrasecut.yaml](https://github.com/timojl/clipseg/blob/master/experiments/phrasecut.yaml)
- Decoder width: CLIPSeg (PC, D=128) gets 48.2 / 56.5 / 78.2 vs CLIPSeg (PC, D=64) 46.1 / 56.2 / 78.2 (PhraseCut t=0.3). D=16 in Table 9 drops to 37.4 text mIoU. — [CLIPSeg Tables 3 & 9](https://arxiv.org/abs/2112.10003)
- Layers: using only layer 3 drops text mIoU to 31.9. The default is extract_layers [3, 7, 9] with depth 3. — [CLIPSeg Table 9](https://arxiv.org/abs/2112.10003); [config](https://github.com/timojl/clipseg/blob/master/experiments/phrasecut.yaml)
- Image size: "ViT-B/16 showing a slightly better performance at an optimal image size of around 350 × 350 pixels" (CLS-token logistic regression on 67 ImageNet vehicle classes). — [CLIPSeg appendix, Fig. 5](https://arxiv.org/abs/2112.10003)
- Augmentation in the config: `pre_crop_image_size: [sample, 1, 1.5]`, `aug: 1new`, random crop "making sure the object remains at least partially visible". No color jitter is mentioned. — [phrasecut.yaml](https://github.com/timojl/clipseg/blob/master/experiments/phrasecut.yaml); [CLIPSeg Sec 3.1](https://arxiv.org/abs/2112.10003)
- Lovász paper side-note: longer training (30K iterations) helped Lovász (76.53 → 77.41) but not cross-entropy (75.53 → 74.97) on VOC. Different setting. — [Lovász Table 2](https://arxiv.org/abs/1705.08790)
- LR references:
  - CLIPSeg 1e-3 → 1e-4 cosine
  - medical CLIPSeg fine-tune 2e-3 with plateau decay
  - VLSM-Adapter 1e-3 / 3e-4
  — [CLIPSeg](https://arxiv.org/abs/2112.10003); [2308.07706](https://arxiv.org/abs/2308.07706); [2405.06196](https://arxiv.org/abs/2405.06196)

### Inferences
- **Longer training (continue 20k → 50k, i.e. +30k iterations)** directly mirrors the authors' refined model. On a T4 it costs about 1.5× a full run (≈9–15 h by the user's 6–10 h/20k estimate). That fits in 30 GPU-h/week only as one big job, split across Kaggle 12-hour sessions with checkpoint resume.
  - A cheaper variant: restart cosine from 3e-4 → 1e-5 for 5–10k iterations.
  - The paper does not report the refined model's PhraseCut numbers, so the gain is unquantified. Note the refined model also changes the head (complex_trans_conv).
- **reduce_dim 64 → 128** is the change with the clearest published gain on PhraseCut mIoU (+2.1 for PC). But it is an architecture change, and it must be trained from scratch, since the decoder shapes change (a full run).
- Multi-scale training or larger input (e.g. 416) is unlikely to help given the ~350 px CLIP optimum. Color jitter can mislead colour-word phrases ("red car"), so avoid hue jitter.

### Gaps
- No published PhraseCut numbers for the 50k refined model vs the 20k model.
- No CLIPSeg ablation of color jitter or multi-scale training.

---

## 6. Distillation / pseudo-labels (e.g. SAM) to sharpen masks

### Takeaway
No primary source was found that distills SAM into CLIPSeg during training. SAM's own mask loss (focal:dice 20:1) is a usable recipe. Generating SAM pseudo-labels for PhraseCut's ~70k training images on a T4 is likely too costly for the 2-week budget.

### Cited Findings
- SAM's mask decoder is supervised with focal + dice in a 20:1 ratio. — [SAM](https://arxiv.org/abs/2304.02643)
- LISA builds a text-prompted segmenter on SAM's decoder with BCE (2.0) + Dice (0.5). Training took "less than 3 days" on 8× RTX 3090. — [LISA](https://arxiv.org/abs/2308.00692)

### Inferences
- SAM ViT-H on a T4 runs at roughly seconds per image; precomputing refined masks for PhraseCut's training set (box/point prompts from GT polygons) would take many GPU-hours. This is an inference and an estimate, not measured. PhraseCut GT masks are polygons, so SAM "refinement" mainly sharpens boundaries, while evaluation is against the original polygons. A mismatch could even hurt IoU against the PhraseCut GT.
- **Not recommended for E4 given constraints.**

### Gaps
- Web search was unavailable. Papers combining CLIPSeg + SAM (e.g. pseudo-label pipelines, "Grounded-SAM"-style) could not be verified and no numbers are reported here.

---

## 7. Feasibility summary for one E4 change (HF transformers 4.55.4, Kaggle T4)

### Takeaway
The best risk/effort candidates, all doable as a 2–4 h fine-tune from the reproduced checkpoint:
1. BCE + Dice, applied to positive samples only.
2. Lowering the negative rate and/or the text–visual mix ratio.
3. Continuing training with a fresh cosine schedule (the "refined" 50k recipe).

The option with the strongest published PhraseCut gain (D=128, +2.1 mIoU) needs a full retrain.

### Cited Findings
- Published PhraseCut deltas available:
  - D 64 → 128: +2.1 mIoU, +0.3 IoU_FG, AP equal
  - no negatives/visual (PC vs PC+): +2.7 mIoU, +1.5 IoU_FG, +1.5 AP
  - "no visual" in Table 9 (best-t): +2.8 text mIoU, +1.1 AP
  — [CLIPSeg Tables 3 & 9](https://arxiv.org/abs/2112.10003)
- Loss changes have only cross-dataset evidence: Lovász +1.9 mIoU on VOC [Lovász](https://arxiv.org/abs/1705.08790), and boundary loss gains on medical data [Boundary loss](https://arxiv.org/abs/1812.07032).

### Inferences

| Option | Code effort (HF loop) | Added GPU cost | Fine-tune from ckpt? | Evidence for PhraseCut gain | Fits 2 weeks? |
|---|---|---|---|---|---|
| BCE + Dice (pos-only), λ_dice ≈ 0.5–1 | ~10–20 lines | ≈0 | Yes (3–5k it) | Indirect (LISA/SAM/VLSM use it; Lovász +1.9 on VOC) | Yes |
| BCE + Lovász-hinge (per-image, pos-only) | ~40 lines (port `lovasz_hinge`) | small (sort per image) | Yes | Indirect (+1.9 VOC) | Yes |
| Focal + Dice (SAM 20:1) | ~20 lines | ≈0 | Yes | Indirect | Yes |
| Lower q_neg (0.2 → 0.05–0.1) | 1 line | 0 | Yes | Indirect (PC > PC+ by +2.7 mIoU, confounded) | Yes |
| Narrower text–visual mix (w max 0.5 → 0.25 / anneal) | 1–5 lines | 0 | Yes | Indirect (no-visual +2.8 text mIoU) | Yes |
| Hard negatives (same-image phrases) | ~30 lines dataset | 0 | Yes | None numeric (GRES rationale) | Yes, risky |
| Continue training (+10–30k it, new cosine) | config only | 3–15 h | Yes | Authors' refined = 50k, no public numbers | Partly (split sessions) |
| reduce_dim 128 | config | ~same per-iter | **No** (new decoder) | +2.1 mIoU (PC) | One full run (6–10 h) |
| CoOp learned prefix | ~50–100 lines | moderate (text backprop) | Yes | +0.6 mIoU ADE20K (DenseCLIP) | Yes, low payoff |
| Test-time prompt ensemble | ~20 lines eval | 9× text enc at eval | N/A (no training) | small per CLIPSeg appendix | Yes (but inference-only) |
| k-shot support averaging | ~20 lines | k× support CLIP fwd | Yes | none for text metrics | Yes, targets E2 |
| Boundary loss | ~60 lines + distance maps | CPU dist-transform per crop | Yes | medical only | Marginal |
| SAM pseudo-label distillation | large | many GPU-h preprocessing | Yes | none found | **No** |

- Caveat for any fine-tune comparison: a fair E4 comparison needs the baseline also continued for the same extra iterations with plain BCE. Otherwise "more training" is confounded with "the change".

### Gaps
- No direct evidence for any of the cheap options on PhraseCut with the exact CLIPSeg PC+ recipe. Expected gains are inferred from different datasets or confounded ablations.
