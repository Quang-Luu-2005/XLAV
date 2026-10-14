# Lấy CLIPSeg làm trục, thắng điểm bằng thực nghiệm gọn

> **Quản lý triển khai:** [Scrum cho một người](docs/agile/scrum.md), 3 sprint trong 07–27/10/2026. S1: chuẩn bị dữ liệu, đo thử và bắt đầu train E1; S2: train xong E1/E1b, chạy E2–E4, khóa số liệu 20/10; S3: báo cáo, slide, demo và gói bàn giao. [Daily log](docs/agile/daily-log.md) · [Sprint Review/Retro](docs/agile/review-retro.md).

> **Cập nhật 09/10/2026, đổi hướng:** Đồ án chuyển từ "chỉ inference trên checkpoint có sẵn" sang **tự train lại CLIPSeg theo công thức PhraseCut+ của tác giả, rồi cải tiến một module**. Checkpoint `CIDAS/clipseg-rd64-refined` chỉ còn là mốc tham chiếu.
> - **E1**: train decoder trên PhraseCut+. PhraseCut+ không có file công bố mà được sinh ngẫu nhiên từ PhraseCut lúc train. Kèm ablation **E1b** chỉ học text.
> - **E2 / E3**: text vs visual prompt và template câu, chạy trên decoder E1.
> - **E4**: thay khối fusion của decoder bằng word-level cross-attention (dự phòng: `reduce_dim` 128), theo [báo cáo nghiên cứu E4](reports/Cải%20tiến%20CLIPSeg%20cho%20E4.md).
> - **Bỏ hoặc hoãn**: so sánh Grounding DINO + SAM / SAM 3 (E4 cũ) và các mở rộng Hạng 1–4 bên dưới.
>
> Chi tiết thay đổi, hiện trạng, lịch và việc cần làm tiếp: [kế hoạch thực thi](docs/implementation-plan.md). Notebook: [source/](source/).

> **Cập nhật 07/10/2026:** Triển khai một người trong 3 tuần, 5 giờ/ngày, theo [kế hoạch thực thi hiện tại](docs/implementation-plan.md). Nội dung dài bên dưới là tham khảo cho phạm vi 3 người / 10 tuần. Xem [paper cần đọc](docs/paper-reading.md), [README chạy code](README.md) và [Trello](https://trello.com/b/85LHoRTG/%C4%91%E1%BB%93-%C3%A1n-clipseg-1-ng%C6%B0%E1%BB%9Di-07-27102026).

**Kết luận chính (BLUF):** Cách đạt điểm cao mà không quá sức là lấy **CLIPSeg (HF `CIDAS/clipseg-rd64-refined`) làm trục**: khảo sát sâu một mô hình rồi đặt nó vào bức tranh rộng (open-vocabulary, promptable, referring). Phần thực nghiệm gồm **4 thí nghiệm cốt lõi, đều chỉ cần inference**: (1) tái lập số liệu gốc trên tập con, (2) so sánh text prompt và visual prompt, (3) nghiên cứu prompt engineering, (4) so sánh CLIPSeg với Grounding DINO + SAM 2 và SAM 3. Thêm **1–2 mở rộng** có tỉ lệ điểm/công sức tốt nhất. Cách này khả thi trên GPU miễn phí vì CLIPSeg chỉ có **decoder ~1,1M tham số trên CLIP ViT-B/16 đóng băng**, tổng ~0,2B tham số, thuần PyTorch, không cần compile ([CLIPSeg paper](https://arxiv.org/pdf/2112.10003); [HF model card](https://huggingface.co/CIDAS/clipseg-rd64-refined)). Grounding DINO và SAM 2 cũng chạy thuần `transformers` ([HF Grounding DINO](https://huggingface.co/docs/transformers/model_doc/grounding-dino); [HF SAM2](https://huggingface.co/docs/transformers/model_doc/sam2)). Ngược lại, LSeg, OVSeg, SAN, CAT-Seg và SEEM phụ thuộc detectron2 hoặc PyTorch cũ, nên chỉ nên dùng **HF Spaces và số liệu đã công bố** ([lang-seg](https://github.com/isl-org/lang-seg); [ov-seg](https://github.com/facebookresearch/ov-seg)). Đồ án có hai điểm khác biệt đáng giá. Thứ nhất, trong các bài đã kiểm tra, **chưa có số liệu CLIPSeg công bố trên RefCOCO**, nên nhóm có thể đưa ra con số đầu tiên chỉ với vài giờ GPU ([CaR](https://arxiv.org/pdf/2312.07661)). Thứ hai, **template ensembling cho CLIPSeg chưa từng được định lượng** ([CLIP paper](https://arxiv.org/pdf/2103.00020)). Kế hoạch dưới đây giả định **10 tuần và nhóm 3 người**; hãy co giãn theo deadline và sĩ số thật.

---

## 1. Đích đến: một khảo sát lấy CLIPSeg làm trục, kèm 4 sản phẩm bàn giao

Đề tài 3.4 có hai vế: "khảo sát" (survey) và "dựa trên VLM + câu nhắc (text/visual prompt)". Bài làm điểm cao cần phân tích sâu, có số liệu do chính nhóm đo, chứ không chỉ liệt kê các phương pháp. Phạm vi nên khoanh như sau. **Lý thuyết** gồm CLIPSeg chi tiết, cộng một taxonomy khoảng 15 phương pháp chia thành 5 nhóm. **Thực nghiệm** chỉ chạy những gì chạy được trên T4: CLIPSeg, Grounding DINO + SAM 2, và tùy chọn SAM 3. Các mô hình còn lại dùng số liệu công bố và demo trên HF Spaces. Không huấn luyện lại CLIPSeg ở quy mô bài báo: 20.000 iteration × batch 64 tương đương ~1,28M mẫu ([CLIPSeg paper](https://arxiv.org/pdf/2112.10003)), mà Kaggle chỉ cho khoảng 30 giờ/tuần theo thông tin người dùng chia sẻ ([gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)).

| Sản phẩm | Hình thức "điểm cao" | Ghi chú |
|---|---|---|
| Báo cáo | PDF có taxonomy, bảng so sánh, số liệu tái lập đặt cạnh số liệu gốc, phân tích lỗi bằng hình | Mỗi con số đều có nguồn hoặc nói rõ là "nhóm tự đo" |
| Slide | 15–20 slide, nhiều hình định tính, 1 slide demo trực tiếp | Có video demo dự phòng |
| Code / notebook | Repo GitHub gồm notebook theo thí nghiệm (E1–E4, mở rộng), `requirements` đã pin phiên bản, README hướng dẫn chạy lại | Mỗi họ mô hình một notebook riêng |
| Demo Gradio | Nhận text prompt và visual prompt, có thanh trượt ngưỡng, heatmap, so sánh với GDINO+SAM 2 | Host trên HF Space (CPU basic) và có bản Colab `share=True` dự phòng |

---

## 2. Lý thuyết cần khảo sát: CLIPSeg là "bộ phân đoạn nhị phân có điều kiện"

### CLIPSeg gồm CLIP đóng băng, FiLM và decoder 1,1M tham số

CLIPSeg (Lüddecke & Ecker, CVPR 2022, arXiv 2112.10003) dùng **CLIP ViT-B/16 đóng băng** (patch 16). Mô hình trích activation ở **3 lớp của CLIP** và đưa chúng vào **decoder transformer 3 khối**, mỗi khối nhận một lớp, nối bằng skip connection kiểu U-Net. Decoder có **token dimension D = 64** và **1.122.305 tham số huấn luyện** ([CLIPSeg paper](https://arxiv.org/pdf/2112.10003)). Vector điều kiện lấy từ CLIP text encoder (với text prompt) hoặc CLIP visual encoder (với visual prompt). Vector này **điều biến đầu vào decoder bằng FiLM** (feature-wise linear modulation). Cuối cùng, một phép chiếu tuyến tính đưa token về **một mask nhị phân W×H cho mỗi prompt** ([CLIPSeg paper](https://arxiv.org/pdf/2112.10003)). Mô hình huấn luyện ở 352×352, batch 64, 20k iteration, learning rate cosine từ 1e-3 xuống 1e-4, chỉ dùng loss BCE ([CLIPSeg paper](https://arxiv.org/pdf/2112.10003)).

Dữ liệu huấn luyện là **PhraseCut+**. Đây là PhraseCut mở rộng với 20% mẫu âm (prompt không khớp ảnh), visual prompt lấy từ ảnh khác có cùng cụm từ, nội suy tuyến tính giữa embedding text và embedding ảnh, random crop, và tiền tố cố định để tăng cường cụm từ ([CLIPSeg paper](https://arxiv.org/pdf/2112.10003)). Về **visual prompt engineering**, đo bằng ΔP(object) trên 1.600 ảnh LVIS: crop +13,60; làm mờ nền +13,15; làm tối nền còn 10% +21,73; **kết hợp crop + blur + tối nền 10% đạt +23,50**. Ngược lại, che mask trực tiếp trên token cho kết quả **−14,44** ([CLIPSeg paper](https://arxiv.org/pdf/2112.10003)). Đây là bảng tái lập tốt nhất cho thí nghiệm E2.

Có ba biến thể. `rd64-uni` (~4 MB) và `rd16` (~1 MB) cho mask "vuông vức" theo lưới patch. `rd64-uni-refined` dùng transposed convolution phức tạp hơn để mask mịn hơn ([timojl/clipseg](https://github.com/timojl/clipseg)). Trên HF, bản refined là `CIDAS/clipseg-rd64-refined`, tương ứng `use_complex_transposed_convolution=True` ([HF docs](https://huggingface.co/docs/transformers/model_doc/clipseg)).

**Kết quả gốc cần trích dẫn trong báo cáo:**

| Tác vụ | CLIPSeg | Đối chứng mạnh | Nguồn |
|---|---|---|---|
| PhraseCut (mIoU / IoU_FG / AP, t=0,3) | PC+: 43,4 / 54,7 / 76,7; PC: 46,1 / 56,2 / 78,2 | MDETR 53,7 mIoU; HulaNet 41,3 | [paper](https://arxiv.org/pdf/2112.10003) |
| Zero-shot tổng quát PASCAL-VOC, 10 lớp unseen (mIoU_S / mIoU_U) | 35,7 / **43,1** | OSR 72,1 / 33,9; SPNet 59,0 / 18,1 | [paper](https://arxiv.org/pdf/2112.10003) |
| PASCAL-5i, text prompt (mIoU / IoU_BIN / AP) | **72,4** / 83,1 / 84,8 | LSeg 52,3 / 67,0 | [paper](https://arxiv.org/pdf/2112.10003) |
| PASCAL-5i one-shot visual (mIoU / IoU_BIN / AP) | 59,5 / 75,0 / 82,3 | HSNet RN50 64,0 / 76,7; RN101 66,2 / 77,6 | [paper](https://arxiv.org/pdf/2112.10003) |
| COCO-20i one-shot (mIoU / IoU_BIN / AP, t=0,1) | 33,3 / 59,1 / 41,7 | HSNet RN101 41,2 / 69,1 | [paper](https://arxiv.org/pdf/2112.10003) |
| LVIS tổng quát (affordance / attribute / meronymy) | 36,9 / 26,6 / 25,7 | — | [paper](https://arxiv.org/pdf/2112.10003) |
| Ablation PhraseCut (mIoU) | Đầy đủ 43,6; không có CLIP pre-training 13,1; D=16 37,4; chỉ lớp 3 31,9 | — | [paper](https://arxiv.org/pdf/2112.10003) |

Thông điệp phân tích nên nêu là CLIPSeg **đổi độ chính xác lấy tính linh hoạt**. Một mô hình, huấn luyện một lần, làm được cả referring, zero-shot và one-shot, nhận cả text lẫn ảnh, nhưng đều thua mô hình chuyên biệt ở từng tác vụ. Ngoài ra, PC thắng PC+ trên PhraseCut, cho thấy việc thêm visual prompt và mẫu âm làm giảm nhẹ độ chính xác khi chỉ dùng text. Ba điểm yếu cần nhấn mạnh: **output 352×352 mờ**, **không tách được instance**, và **K lớp cần K lần forward rồi lấy argmax**.

### Taxonomy năm nhóm, sắp theo loại prompt và cách dùng VLM

Khảo sát TPAMI 2024 về open-vocabulary tổ chức theo tác vụ semantic/instance/panoptic, và trong các bảng trích được không thấy CLIPSeg hay LSeg ([Awesome-Open-Vocabulary](https://github.com/jianzongwu/Awesome-Open-Vocabulary)). Vì vậy, một taxonomy **lấy prompt làm trung tâm** là ngách hợp lý cho đồ án. Đề xuất ba trục: (A) mức chi tiết của output (binary → semantic → instance/panoptic → video); (B) loại prompt (T = text, I = ảnh mẫu, G = hình học, R = suy luận); (C) cách dùng VLM (đóng băng + head nhẹ → adapter → fine-tune → làm classifier cho proposal → MLLM + SAM).

| Nhóm | Phương pháp (venue) | Cơ chế cốt lõi | Prompt | Output | Số liệu tham chiếu | Chạy trên Colab? |
|---|---|---|---|---|---|---|
| 1. Căn chỉnh pixel–text, một giai đoạn | **CLIPSeg** (CVPR'22) | CLIP đóng băng + FiLM + decoder 1,1M | T, I | Binary | PASCAL-5i text 72,4 mIoU ([paper](https://arxiv.org/pdf/2112.10003)) | **Dễ** (HF) |
| | LSeg (ICLR'22) | Embedding pixel dày đặc so với embedding nhãn, loss contrastive | T | Semantic | FSS-1000 87,8; PASCAL-5i 52,8–53,8 ([lang-seg](https://github.com/isl-org/lang-seg)) | Khó (torch 1.7, repo đã archive) |
| | MaskCLIP (ECCV'22) | Không cần huấn luyện, sửa lớp attention cuối của CLIP | T | Semantic | MaskCLIP+ VOC unseen 86,1 ([arXiv](https://arxiv.org/abs/2112.01071)) | — |
| | ZegCLIP (CVPR'23, cần kiểm tra) | Deep prompt tuning, sigmoid theo từng lớp | T | Semantic | VOC hIoU 84,3 ([arXiv](https://arxiv.org/abs/2212.03588)) | — |
| | CAT-Seg (CVPR'24) | Tổng hợp cost volume, fine-tune CLIP | T | Semantic | ViT-B: A-150 31,8 / PC-59 57,5 ([GitHub](https://github.com/cvlab-kaist/CAT-Seg)) | Trung bình–khó (detectron2) |
| 2. Proposal + CLIP phân loại | ZegFormer, OpenSeg, **OVSeg** (CVPR'23), ODISE | Mask class-agnostic, sau đó CLIP phân loại | T | Semantic/panoptic | OVSeg ViT-B: A-150 24,8 ([CAT-Seg](https://arxiv.org/html/2303.11797)) | Khó (archive, CC BY-NC) |
| 3. Adapter trên VLM đóng băng | **SAN** (CVPR'23), FC-CLIP (NeurIPS'23) | Side adapter tạo mask + attention bias | T | Semantic/panoptic | SAN ViT-B: A-150 27,5 ([SAN](https://arxiv.org/pdf/2302.12242)) | Trung bình–khó |
| 4. Referring segmentation | **CRIS** (CVPR'22), **LAVT** (CVPR'22), LISA | CRIS: CLIP fine-tune + contrastive text-pixel; LAVT: early fusion trong Swin; LISA: token `<SEG>` | T, R | Binary | RefCOCO val oIoU: CRIS 70,47; LAVT 72,73 ([CRIS](https://arxiv.org/pdf/2111.15174); [LAVT](https://arxiv.org/pdf/2112.02244)) | Không cần chạy |
| 5. Promptable generalist | **X-Decoder** (CVPR'23), **SEEM** (NeurIPS'23) | Decoder chung, không gian visual-semantic chung | T, I, G | Đa tác vụ | SEEM Ref-COCOg cIoU 62,4 ([GitHub](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)) | Khó (CUDA op tùy biến) |
| | SAM / SAM 2 | Prompt hình học, memory cho video | G | Instance | SAM2.1-tiny 91,2 FPS trên A100 ([sam2](https://github.com/facebookresearch/sam2)) | **Dễ** (HF) |
| | **Grounded-SAM** (2024) | Grounding DINO: text → box, rồi SAM: box → mask | T (qua detector) | Instance | SegInW 48,7 AP ([arXiv](https://arxiv.org/abs/2401.14159)) | **Dễ** (HF) |
| | **SAM 3** (ICLR'26) | Promptable Concept Segmentation, presence head, 848M tham số | T, I, G | Mọi instance, video | SA-Co/Gold 54,1 cgF1 so với OWLv2 24,6 ([sam3](https://github.com/facebookresearch/sam3)) | Trung bình (gated, bf16) |

Trong báo cáo cần nêu hai mối nối lý thuyết. Thứ nhất, **SAM 3 là "hậu duệ khái niệm" của CLIPSeg**: exemplar của SAM 3 là phiên bản instance-level của visual prompt, còn presence head đóng vai trò tương tự 20% mẫu âm khi huấn luyện CLIPSeg ([arXiv 2511.16719](https://arxiv.org/abs/2511.16719)). Thứ hai, phát hiện của OVSeg rằng "CLIP kém trên ảnh bị che mask" tương ứng với kết quả visual prompt engineering của CLIPSeg ([OVSeg](https://arxiv.org/abs/2210.04150)). Xu hướng 2022→2024: CLIP đóng băng + head nhẹ → hai giai đoạn proposal + CLIP → adapter / một giai đoạn → fine-tune có kiểm soát (CAT-Seg) → foundation model nhận prompt (SAM 3).

Về hậu bối của CLIPSeg, không có dòng "CLIPSeg v2". CLIPSeg trở thành baseline và công cụ, chủ yếu trong y tế: MedVLSM ([arXiv 2308.07706](https://arxiv.org/abs/2308.07706)), VLSM-Adapter ([arXiv 2405.06196](https://ar5iv.labs.arxiv.org/html/2405.06196)), Telescopic Adapters WACV'26 ([arXiv 2512.13855](https://arxiv.org/abs/2512.13855)), MedCLIPSeg CVPR'26 ([arXiv 2602.20423](https://arxiv.org/abs/2602.20423)). Nó cũng được dùng làm công cụ tạo mask cho inpainting ([diffusers text_inpainting](https://huggingface.co/datasets/diffusers/community-pipelines-mirror/blob/main/v0.33.1/text_inpainting.py)).

---

## 3. Bốn bộ dữ liệu nhỏ đủ cho toàn bộ thực nghiệm

Nguyên tắc chọn dữ liệu: dùng **tập con chính thức hoặc tập đánh giá đã đóng gói trên HF** để tránh tải hàng chục GB. Không dùng COCO-20i đầy đủ (COCO 2014 ~37,6 GiB, [TFDS](https://www.tensorflow.org/datasets/catalog/coco)) và cũng không cần ảnh PhraseCut train.

| Dataset | Quy mô / dung lượng | Nguồn tải | Dùng cho | Metric |
|---|---|---|---|---|
| **PhraseCut miniv / val** | miniv 100 ảnh / 821 cụm từ; val 2.971 ảnh / 20.316 cụm từ; toàn bộ ảnh ~11 GB | `python download_dataset.py -s miniv` ([PhraseCutDataset](https://github.com/ChenyunWu/PhraseCutDataset)) | E1 (in-distribution), E4 | mean IoU, cum IoU, Pr@0,5/0,7/0,9 qua `evaluate.py` chính thức |
| **PASCAL-5i** (VOC2012 + SBD) | 4 fold × 5 lớp, 1.000 episode mỗi fold | VOC2012 tar + mask SBD qua [HSNet repo](https://github.com/juhongm999/hsnet); bản Kaggle [pascal-voc-2012](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012) | E1, E2 (text và visual) | mIoU theo lớp, FB-IoU ([HSNet](https://arxiv.org/pdf/2104.01538)), AP |
| **PASCAL VOC 2012 val** (semantic) | ~1,5K ảnh, 20 lớp ([OVSeg](https://arxiv.org/pdf/2210.04150)) | Như trên | E3 (prompt engineering, mỗi lớp một lần forward, sau đó argmax) | mIoU |
| **RefCOCO-M** | 1.190 ảnh, 5.598 biểu thức, mask chính xác pixel, MIT | `Voxel51/RefCOCO-M` ([HF](https://huggingface.co/datasets/Voxel51/RefCOCO-M/blob/main/README.md)) | E4, mở rộng RefCOCO, stress-test | mIoU, oIoU, Pr@X |
| **RefCOCO (eval)** | val 8.811 / testA 1.975 / testB 1.810 dòng; ~2,28 GB; không có train | `lmms-lab/RefCOCO` ([HF](https://huggingface.co/datasets/lmms-lab/RefCOCO/blob/main/README.md)); hoặc `jxu124/refcoco-benchmark` ([HF](https://huggingface.co/datasets/jxu124/refcoco-benchmark)) | Mở rộng "số liệu RefCOCO đầu tiên" | oIoU + mIoU (báo cáo cả hai) |
| **Kvasir-SEG** | 1.000 ảnh polyp, 46,2 MB, chỉ dùng cho nghiên cứu/giáo dục | [Simula](https://datasets.simula.no/kvasir-seg/); HF `kowndinya23/Kvasir-SEG` (880/120); [Kaggle](https://www.kaggle.com/datasets/debeshjha1/kvasirseg) | E3 (OOD), mở rộng domain adaptation | Dice, IoU |
| **ISIC 2016** | 900 train / 379 test (602 + 232 MB), CC-0 | [ISIC](https://challenge.isic-archive.com/data/) | Mở rộng domain adaptation (thay thế) | Dice |

**Định nghĩa metric:** mean IoU = (1/N)·Σ I_t/U_t, tính theo mẫu. Cumulative IoU (oIoU) = ΣI/ΣU; chỉ số này ưu ái vật thể lớn ([PhraseCut](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_PhraseCut_Language-Based_Image_Segmentation_in_the_Wild_CVPR_2020_paper.pdf); [LAVT](https://arxiv.org/pdf/2112.02244)). Với few-shot, mIoU = trung bình IoU_c theo lớp trong fold, và FB-IoU = ½(IoU_F + IoU_B) ([HSNet](https://arxiv.org/pdf/2104.01538)). AP của CLIPSeg là diện tích dưới đường precision–recall ở mức pixel và **không phụ thuộc ngưỡng** ([CLIPSeg paper](https://arxiv.org/pdf/2112.10003)). Luôn báo cáo AP kèm metric có ngưỡng.

**Số liệu tham chiếu ngoài CLIPSeg:**
- Referring segmentation có giám sát (oIoU, RefCOCO val/testA/testB): CRIS R101 70,47 / 73,18 / 66,10 ([CRIS](https://arxiv.org/pdf/2111.15174)); LAVT 72,73 / 75,82 / 68,79 ([LAVT](https://arxiv.org/pdf/2112.02244)).
- Training-free, so sánh công bằng hơn với CLIPSeg zero-shot: TAS RN50 đạt oIoU 29,75 / 32,85 / 28,23 và mIoU 39,91 / 42,85 / 35,85 trên RefCOCO ([TAS](https://arxiv.org/pdf/2310.18049)).
- Một đánh giá bên thứ ba của CLIPSeg zero-shot cho PASCAL-5i trung bình 51,7 mIoU và COCO-20i 35,0 ([ClsCLIP](https://arxiv.org/pdf/2304.06212)). Con số này khác xa 72,4 của bài gốc, do khác protocol và trọng số.
- Y tế, CLIPSeg fine-tune (Dice): Kvasir-SEG 89,51; ISIC 92,12. Zero-shot trong khoảng 20–70%, ISIC 67,98% ([MedVLSM](https://arxiv.org/html/2308.07706v3)).

Có ba lưu ý về tính công bằng khi so sánh. (1) PhraseCut là dữ liệu **in-distribution** với CLIPSeg, không phải zero-shot. (2) Checkpoint HF được huấn luyện trên PhraseCut+ đầy đủ, nên chạy nó trên PASCAL-5i không đúng hẳn protocol "lớp held-out". Cần ghi rõ điều này và so sánh với cả hai con số 72,4 và 51,7. (3) Số liệu gốc của PhraseCut đo trên tập test, còn nhóm đo trên val/miniv, nên chỉ so được ở mức "cùng cỡ".

---

## 4. Ba mô hình chạy thật, phần còn lại dùng HF Spaces

| Mô hình | ID / tài nguyên | Độ khó trên T4 | Cách dùng trong đồ án |
|---|---|---|---|
| **CLIPSeg** | `CIDAS/clipseg-rd64-refined` (Apache-2.0, ~0,2B); tùy chọn `CIDAS/clipseg-rd64` | **Dễ** | Mô hình chính. Text: `processor(text=texts, images=[img]*len(texts), padding=True)`. Visual: `conditional_pixel_values` hoặc `visual_prompt=`. Có thể truyền `conditional_embeddings` (batch, 512) để nội suy hoặc ensemble ([HF docs](https://huggingface.co/docs/transformers/model_doc/clipseg); [HF blog](https://huggingface.co/blog/clipseg-zero-shot)) |
| **Grounding DINO** | `IDEA-Research/grounding-dino-tiny` / `-base` | **Dễ** | Text → box; `post_process_grounded_object_detection(threshold=0.4, text_threshold=0.3)` ([HF docs](https://huggingface.co/docs/transformers/model_doc/grounding-dino)) |
| **SAM 2.1** | `facebook/sam2.1-hiera-{tiny,small,base-plus,large}` | **Dễ** | Box/point → mask; tái sử dụng image embedding ([HF docs](https://huggingface.co/docs/transformers/model_doc/sam2)) |
| **SAM 3** (tùy chọn) | `facebook/sam3`, **gated**, cần HF token | **Trung bình** | `Sam3Model` / `Sam3Processor`; mặc định 1008×1008; **ép fp32 trên T4** ([HF docs](https://huggingface.co/docs/transformers/model_doc/sam3); [issue #507](https://github.com/facebookresearch/sam3/issues/507)) |
| SAN / CAT-Seg / OVSeg / SEEM / LSeg | HF Spaces: `Mendel192/SAN-Demo`, `facebook/ov-seg`, CAT-Seg Spaces | Trung bình–khó | Chỉ lấy hình định tính và số liệu công bố; không cài đặt ([SAN](https://github.com/MendelXu/SAN); [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg)) |

Ba mẹo kỹ thuật tiết kiệm nhiều thời gian:
- Output CLIPSeg có shape `[N, 352, 352]`. **Upsample bilinear trên logit về kích thước ảnh gốc rồi mới sigmoid và đặt ngưỡng** ([HF blog](https://huggingface.co/blog/clipseg-zero-shot)).
- Với đa lớp, chồng các bản đồ sigmoid và thêm một kênh "background" hằng bằng ngưỡng, rồi lấy argmax ([HF blog](https://huggingface.co/blog/clipseg-zero-shot)).
- Cache image encoding khi chỉ đổi prompt hoặc ngưỡng. BlabberSeg báo cáo tăng tốc 10–27× nhờ tái sử dụng embedding ([arXiv 2410.12979](https://arxiv.org/pdf/2410.12979)).

Không nguồn nào đo VRAM hay độ trễ của CLIPSeg trên T4. Nhóm nên **tự đo ở tuần 1** bằng `torch.cuda.max_memory_allocated()` và thời gian chạy trên 100 ảnh. Đây đồng thời là một bảng hiệu năng cho báo cáo, kèm số tham số huấn luyện và tổng số tham số theo mẫu bảng hiệu năng của SAN ([SAN](https://arxiv.org/pdf/2302.12242)).

**Demo Gradio.** Lấy mẫu từ Space `taesiri/CLIPSeg` (Blocks, ảnh + textbox + slider ngưỡng/alpha, overlay `jet`), bỏ decorator `@spaces.GPU` khi chạy Colab ([app.py](https://huggingface.co/spaces/taesiri/CLIPSeg/blob/main/app.py)). Nên bổ sung bốn thứ:
1. Ô ảnh visual prompt, kèm checkbox "crop + blur + tối nền".
2. Nhiều prompt cách nhau bằng dấu phẩy, hiển thị bản đồ argmax.
3. Toggle min-max normalize. Demo gốc chuẩn hóa min-max nên **luôn "tìm thấy" vật thể kể cả khi không có**. Đây là một điểm phân tích hay cho báo cáo.
4. Tab so sánh CLIPSeg với GDINO + SAM 2 trên cùng ảnh và prompt.

---

## 5. Bốn thí nghiệm cốt lõi chỉ tốn ~6–12 giờ GPU

*Thời gian GPU trong mục này là ước lượng của kế hoạch. Các ghi chú nguồn không đo trên T4, nên cần đo lại ở tuần 1.*

### Thí nghiệm cốt lõi

| # | Thí nghiệm | Các bước cụ thể | Metric và so với | GPU (ước lượng) | Rủi ro |
|---|---|---|---|---|---|
| **E1** | **Tái lập zero-shot/one-shot trên tập con** | (1) PhraseCut miniv rồi val: xuất PNG nhị phân theo task ID, chạy `evaluate.py` ([repo](https://github.com/ChenyunWu/PhraseCutDataset)). (2) PASCAL-5i: 4 fold × 1.000 episode theo loader HSNet; chạy text prompt (tên lớp) và one-shot visual prompt (ảnh support). (3) Quét ngưỡng 0,1–0,7, báo cáo ở t=0,3 (theo bài gốc), ở ngưỡng tốt nhất trên val, và AP | PhraseCut 43,4 / 54,7 / 76,7; PASCAL-5i text 72,4, visual 59,5 ([paper](https://arxiv.org/pdf/2112.10003)) | ~1–2 giờ | Thấp; lệch số do khác checkpoint và split, cần giải thích |
| **E2** | **Text prompt và visual prompt** | Trên cùng các episode PASCAL-5i, so sánh: (a) text; (b) ảnh support thô; (c) crop / blur / tối nền 10% / kết hợp (tái lập Bảng 2, đo bằng mIoU thay cho ΔP); (d) vòng tròn đỏ ([arXiv 2304.06712](https://arxiv.org/abs/2304.06712)); (e) nội suy text–visual α ∈ {0; 0,25; 0,5; 0,75; 1} qua `conditional_embeddings`; (f) k-shot bằng cách trung bình embedding của 1/3/5 ảnh support | mIoU, FB-IoU, AP; so với Bảng 2 (+23,50) | ~1–2 giờ | Thấp |
| **E3** | **Prompt engineering (text)** | Trên VOC val (semantic, argmax + kênh nền), so sánh: tên lớp trơn, "a photo of a {}", ensemble 5–20 template (trung bình embedding và trung bình logit), từ đồng nghĩa, mô tả do LLM sinh. Lặp lại trên Kvasir-SEG (OOD) | mIoU / Dice. Giả thuyết: ensemble giúp in-domain (CLIP +3,5% ImageNet, [CLIP](https://arxiv.org/pdf/2103.00020)) nhưng không giúp OOD (60 biến thể prompt đều thua baseline, [arXiv 2604.08956](https://arxiv.org/abs/2604.08956)) | ~1–2 giờ | Thấp; hiệu ứng có thể nhỏ, nhưng kết quả âm vẫn đáng báo cáo |
| **E4** | **CLIPSeg, Grounded-SAM và SAM 3** | Trên RefCOCO-M (hoặc RefCOCO testA/testB, 300–500 mẫu) và PhraseCut miniv, cộng ~30 ảnh định tính "stuff/part" (bầu trời, tay cầm cốc). GDINO-tiny → SAM2.1-small; SAM 3 fp32. **Quy ước chuyển sang 1 mask**: referring lấy box hoặc instance điểm cao nhất; PhraseCut lấy hợp các instance vượt ngưỡng. Ghi rõ quy ước trong báo cáo | mIoU, oIoU, Pr@0,5/0,7/0,9, độ trễ, VRAM đỉnh | ~3–6 giờ (SAM 3 ở 1008 px chiếm phần lớn) | Trung bình: quyền truy cập SAM 3 và lỗi bf16 |

Cần ghi nhận trước một giả thuyết cho E4. Detect-then-segment cho mask sắc nét, nhưng **thất bại khi detector bỏ sót hoặc khi prompt mô tả "stuff" hay vùng ảnh** ("sky", "the left part"). CLIPSeg thì ngược lại: mask mờ nhưng xử lý được prompt dạng vùng và affordance. Đây là luận điểm trung tâm của chương thảo luận.

### Mở rộng, xếp hạng theo điểm/công sức

| Hạng | Mở rộng | Bước cụ thể | Metric | Công sức / GPU | Rủi ro |
|---|---|---|---|---|---|
| **1** | **Số liệu CLIPSeg đầu tiên trên RefCOCO** (gần như miễn phí nếu gộp vào E4) | Chạy CLIPSeg trên toàn bộ `lmms-lab/RefCOCO` val/testA/testB, có thể thêm RefCOCO+ và RefCOCOg. Polygon → mask bằng `pycocotools`. Chọn ngưỡng trên val. Tách theo loại biểu thức (có hoặc không có từ chỉ vị trí) | oIoU + mIoU; đặt cạnh baseline training-free (TAS 29,75 oIoU) và CRIS/LAVT | 1–2 ngày người; ~1 giờ GPU | Thấp. "Đầu tiên" chỉ đúng **trong các bài đã kiểm tra** ([CaR](https://arxiv.org/pdf/2312.07661); [SSC](https://arxiv.org/pdf/2310.13479); [B2G](https://arxiv.org/pdf/2509.23098)), nên viết là "chưa tìm thấy số liệu công bố". CLIPSeg không hẳn zero-shot vì đã học trên ảnh VG |
| **2** | **Tinh chỉnh mask CLIPSeg bằng SAM** | Sigmoid → ngưỡng → thành phần liên thông lớn nhất → (a) box, (b) top-k điểm dương và điểm âm → SAM2.1. Ablation box / points / box+points (ClipSAM cho thấy kết hợp tốt nhất, [arXiv 2401.12665](https://ar5iv.labs.arxiv.org/html/2401.12665)). Thêm các hàng không dùng SAM: CRF, guided filter, TTA (lật ảnh, đa tỉ lệ), sliding window | mIoU, boundary F-score, thời gian; tham chiếu Prompt-SAM ([Springer](https://link.springer.com/article/10.1007/s44230-025-00115-4)) | 5–8 ngày người; ~2–4 giờ | Thấp–trung bình; SAM với box làm **mất mask bộ phận** ("handle of the cup"), cần báo cáo như một phát hiện |
| **3** | **Stress-test độ bền** | 100–200 ảnh RefCOCO/COCO, nhóm theo: phủ định ("not the dog"), trái/phải, đếm và nhiều instance, gắn thuộc tính ("red cup"), vật nhỏ (chia bin theo diện tích GT), OOD. Giả thuyết: gần mức ngẫu nhiên với trái/phải, vì embedding "left"/"right" của CLIP có cosine 0,9894 ([arXiv 2311.11477](https://arxiv.org/pdf/2311.11477)), và bỏ qua "not" ([arXiv 2501.10913](https://arxiv.org/abs/2501.10913)) | IoU theo nhóm, biểu đồ radar hoặc cột | 3–5 ngày (chủ yếu tuyển chọn mẫu); <1 giờ | Thấp; tốn công gán nhãn |
| **4** | **Domain adaptation: fine-tune decoder trên Kvasir-SEG / ISIC** | So sánh zero-shot, fine-tune chỉ decoder (`model.clip.requires_grad_(False)`, AdamW 1e-4–1e-3, fp16, batch 16–32, `labels` dạng float {0,1}, có cặp âm), và tùy chọn LoRA/adapter. Vẽ đường label budget 1/5/10/100% | Dice; tham chiếu full fine-tune 89,51 (Kvasir) ([MedVLSM](https://arxiv.org/html/2308.07706v3)), Dense Adapter 89,10 ([VLSM-Adapter](https://ar5iv.labs.arxiv.org/html/2405.06196)) | 6–10 ngày; ~1–3 giờ/run, tổng ~5–10 giờ | Trung bình: lỗi label (loss âm khi mask 0–255, label bị broadcast sai shape) ([HF forum](https://discuss.huggingface.co/t/fine-tune-clipseg-with-image-mask-dataset/36935); [#24494](https://github.com/huggingface/transformers/issues/24494)); split khác MedVLSM (800/100/100 so với 880/120 trên HF) |

**Khuyến nghị:** làm **Hạng 1** mặc định vì gộp được vào E4, cộng **Hạng 2** để có bảng kết quả và hình ảnh ấn tượng. Nếu giảng viên coi trọng phần "có huấn luyện", thay Hạng 2 bằng Hạng 4. Hạng 3 là phần thêm nếu còn 1 tuần dư. Không nên làm cả 4: một khảo sát có 2 mở rộng phân tích kỹ có giá trị hơn 4 mở rộng làm nông.

---

## 6. Lịch 10 tuần: chốt số liệu ở tuần 8, tuần 10 để dự phòng

*Giả định: 10 tuần từ lúc bắt đầu đến khi nộp. Nếu deadline ngắn hơn, gộp tuần 2–3 và 6–7, và chỉ làm 1 mở rộng. Nếu dài hơn, thêm Hạng 3.*

| Tuần | Mốc chính | Sản phẩm cuối tuần |
|---|---|---|
| 1 | Đọc bài CLIPSeg (cả nhóm); dựng môi trường Colab/Kaggle, pin `transformers`; **cả 3 người xin quyền `facebook/sam3` ngay**; đo VRAM và độ trễ của CLIPSeg/GDINO/SAM2; tải PhraseCut miniv, RefCOCO-M, Kvasir | Notebook `00_setup`, bảng hiệu năng v0, dàn ý báo cáo |
| 2 | Viết chương CLIPSeg; E1 trên PhraseCut miniv; dựng loader PASCAL-5i | Bản nháp chương 3; số liệu E1 trên miniv |
| 3 | E1 trên PhraseCut val và PASCAL-5i (text + visual), quét ngưỡng; demo Gradio v0 | Bảng tái lập so với bài gốc; demo chỉ có text |
| 4 | E2 (text so với visual, tái lập Bảng 2) và E3 (prompt engineering); nháp taxonomy | Bảng E2/E3; bảng taxonomy v1 |
| 5 | E4 (CLIPSeg / GDINO+SAM2 / SAM 3) và **Hạng 1** (RefCOCO đầy đủ); **mốc giữa kỳ**: chương 1–5 có nháp | Bảng so sánh chính; số liệu RefCOCO |
| 6 | Mở rộng thứ hai (Hạng 2 hoặc Hạng 4), phần thực thi | Notebook mở rộng chạy được, kết quả sơ bộ |
| 7 | Hoàn thiện mở rộng, ablation; demo v1 (visual prompt, so sánh, đa prompt) | Bảng mở rộng; demo v1 trên HF Space |
| 8 | Phân tích lỗi, hình định tính, **chốt toàn bộ số liệu**; báo cáo bản đầy đủ | Báo cáo v1 hoàn chỉnh |
| 9 | Review chéo, viết slide, quay video demo, tập thuyết trình, chuẩn bị Q&A | Slide v1, video, README repo |
| 10 | **Dự phòng**: sửa theo góp ý, kiểm tra trích dẫn, đóng gói và nộp | Báo cáo, slide, repo và demo bản cuối |

---

## 7. Phân công ba vai có chồng lấn

*Giả định: nhóm 3 người. Nhóm 2 người thì gộp vai A và C. Nhóm 4 người thì tách mở rộng cho người thứ tư.*

| Vai | Phụ trách chính | Hỗ trợ / chồng lấn |
|---|---|---|
| **A: Lý thuyết và báo cáo** | Đọc và tổng hợp ~15 bài; taxonomy và bảng so sánh; chương 1–4; kiểm tra các điểm mâu thuẫn (mục 10); quản lý tài liệu tham khảo; chủ trì E3 (prompt engineering) | Viết phần phân tích kết quả cùng B; chụp hình từ HF Spaces cho các mô hình khó chạy |
| **B: Thực nghiệm** | Loader dữ liệu, script metric (PhraseCut `evaluate.py`, mIoU/FB-IoU, oIoU, Dice); E1, E2, E4, Hạng 1; mở rộng Hạng 4 nếu chọn | Viết chương 5–6; review code của C |
| **C: Demo và kỹ thuật** | Môi trường và pin phiên bản; pipeline GDINO+SAM2 và SAM 3; Hạng 2 (SAM refinement); đo hiệu năng; Gradio + HF Space; hình và video cho slide | Hỗ trợ E4; viết chương demo; dàn slide |

Cả nhóm cùng đọc bài CLIPSeg ở tuần 1. Mỗi người trình bày phần của mình khi bảo vệ. Họp 30 phút mỗi tuần để chốt bảng số liệu chung, lưu trên Google Sheet.

---

## 8. Khung báo cáo và slide

**Báo cáo (đề xuất 9 chương):**
1. Giới thiệu: bài toán phân đoạn theo prompt, động lực, đóng góp của nhóm (tái lập, so sánh, số liệu RefCOCO, mở rộng).
2. Kiến thức nền: phân đoạn ảnh, ViT, CLIP và học contrastive, định nghĩa text prompt, visual prompt, prompt hình học.
3. CLIPSeg chi tiết: kiến trúc, FiLM, decoder, PhraseCut+, visual prompt engineering, các biến thể, kết quả gốc, hạn chế.
4. Khảo sát liên quan: taxonomy 3 trục và 5 nhóm, bảng so sánh, xu hướng 2022→2026, vị trí của SAM 3.
5. Dữ liệu và metric: định nghĩa, quy ước ngưỡng, các vấn đề về công bằng khi so sánh.
6. Thực nghiệm cốt lõi: E1–E4, bảng và hình, thảo luận.
7. Mở rộng: Hạng 1 và Hạng 2 (hoặc 4).
8. Demo và ứng dụng: Gradio, ứng dụng inpainting và tạo nhãn ([HF blog](https://huggingface.co/blog/clipseg-zero-shot)).
9. Kết luận và hướng phát triển; tài liệu tham khảo; phụ lục (cấu hình, prompt template, hướng dẫn chạy lại).

**Slide (15–20 phút):**

| # | Nội dung |
|---|---|
| 1 | Tiêu đề, thành viên |
| 2 | Bài toán: một ảnh, nhiều kiểu prompt, ví dụ trực quan |
| 3 | Đóng góp của nhóm |
| 4–5 | CLIP → CLIPSeg: sơ đồ kiến trúc, FiLM |
| 6 | Visual prompt engineering |
| 7 | Kết quả gốc |
| 8–9 | Taxonomy và bảng so sánh, xu hướng tiến tới SAM 3 |
| 10 | Dữ liệu và metric |
| 11 | E1: tái lập |
| 12 | E2: text so với visual |
| 13 | E3: prompt |
| 14 | E4: so sánh 3 mô hình, kèm hình |
| 15 | RefCOCO |
| 16 | Mở rộng |
| 17 | Lỗi điển hình (trái/phải, phủ định, vật nhỏ) |
| 18 | Demo trực tiếp |
| 19 | Kết luận |
| 20 | Q&A dự phòng (hiệu năng, giới hạn) |

---

## 9. Sáu rủi ro chính đều có phương án dự phòng

| Rủi ro | Biểu hiện | Cách xử lý |
|---|---|---|
| **Chọn ngưỡng** | Bài gốc dùng 0,3 cho PhraseCut/PASCAL và 0,1 cho COCO-20i ([paper](https://arxiv.org/pdf/2112.10003)); "không thể hiệu chỉnh tự động" ([Dataiku](https://www.dataiku.com/blog/no-training-data-no-problem)) | Quét 0,1–0,7 trên val; báo cáo ở t=0,3, ở ngưỡng tốt nhất trên val và AP; **không min-max normalize khi tính metric** |
| **Độ phân giải 352×352 thấp** | Mask mờ, mất vật nhỏ và mảnh ([HF blog](https://huggingface.co/blog/clipseg-zero-shot)) | Dùng bản refined; upsample logit trước khi đặt ngưỡng; sliding window hoặc TTA; SAM refinement (Hạng 2); phân tích IoU theo kích thước vật |
| **SAM 3 gated, lỗi bf16 trên T4** | Chờ duyệt; `RuntimeError ... BFloat16 and Float` ([#507](https://github.com/facebookresearch/sam3/issues/507)); OOM khi cảnh dày đặc ([#301](https://huggingface.co/facebook/sam3/discussions/301)) | Xin quyền từ tuần 1; dùng `transformers` thay vì repo chính thức (repo yêu cầu Python ≥3.12, torch ≥2.7, CUDA ≥12.6, [sam3](https://github.com/facebookresearch/sam3)); `model.to(torch.float32)`; xử lý từng ảnh một; chạy tập con 100–500 ảnh; nếu vẫn lỗi thì chỉ dùng demo và số liệu công bố |
| **Dependency hell với repo cũ** | LSeg cần torch 1.7.1 (đã archive 2024-12); OVSeg archive 2025-07; detectron2 phải build từ source ([lang-seg](https://github.com/isl-org/lang-seg); [ov-seg](https://github.com/facebookresearch/ov-seg)) | Không cài; dùng HF Spaces (kiểm tra trước buổi bảo vệ vì Space hay ngủ hoặc lỗi) và số liệu công bố; mỗi họ mô hình một notebook; pin phiên bản ở cell đầu |
| **Dung lượng tải dữ liệu** | Toàn bộ PhraseCut ~11 GB; COCO 2014 ~37,6 GiB | Chỉ dùng miniv/val, RefCOCO trên HF (~2,3 GB), Kvasir 46 MB; lưu bản đã xử lý vào Kaggle Datasets hoặc Google Drive |
| **Quota Kaggle và Colab** | ~30 giờ/tuần, phiên tối đa 12 giờ; Colab free không cam kết GPU ([gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)) | Ba tài khoản Kaggle chia việc; checkpoint và kết quả ghi ra đĩa sau mỗi fold; cache feature CLIP; chọn "T4 x2" thay vì P100 (P100 sm_60 có thể không còn được PyTorch mới hỗ trợ, **chưa kiểm chứng**, kiểm tra bằng `torch.cuda.get_arch_list()`) |

---

## 10. Checklist để được điểm cao, và các điểm cần kiểm chứng

**Checklist:**
- [ ] Taxonomy **theo loại prompt**, có bảng so sánh đủ các cột: backbone đóng băng hay tune, loại prompt, loại output, dữ liệu, benchmark, khả năng chạy.
- [ ] Giải thích CLIPSeg đến mức tham số: 3 lớp trích, D=64, 1.122.305 tham số, FiLM, PhraseCut+ với 20% mẫu âm.
- [ ] **Bảng tái lập**: số của nhóm đặt cạnh số bài gốc, có giải thích chênh lệch (checkpoint, split, ngưỡng).
- [ ] Mỗi bảng ghi rõ ngưỡng, split, số mẫu, và **cả metric có ngưỡng lẫn AP**. Với RefCOCO, ghi **cả oIoU và mIoU**.
- [ ] Ít nhất một kết quả mới chưa có trong tài liệu: CLIPSeg trên RefCOCO, hoặc hiệu quả của template ensembling.
- [ ] So sánh với mô hình hiện đại (GDINO+SAM 2, SAM 3) và **lập luận khi nào CLIPSeg vẫn đáng dùng**: nhẹ, xử lý được prompt dạng vùng hoặc affordance, nhận visual prompt.
- [ ] Phân tích lỗi có hình: trái/phải, phủ định, vật nhỏ, vật bộ phận, min-max normalize gây dương tính giả.
- [ ] Bảng hiệu năng do nhóm tự đo: tham số, độ trễ, VRAM trên T4.
- [ ] Demo Gradio có cả text prompt và visual prompt, chạy trực tiếp, có video dự phòng.
- [ ] Repo chạy lại được: README, phiên bản đã pin, seed, notebook theo từng thí nghiệm.
- [ ] Trích dẫn đầy đủ; nêu giấy phép (Kvasir chỉ dùng cho nghiên cứu/giáo dục; SAM License; trọng số CLIPSeg gốc không theo MIT).
- [ ] Mục "Hạn chế": không huấn luyện lại ở quy mô bài báo; một số so sánh không cùng protocol.

**Các điểm cần kiểm chứng trước khi viết (ghi chú nguồn mâu thuẫn hoặc chưa chắc chắn):**

| Điểm | Mâu thuẫn hoặc chưa chắc | Cách kiểm |
|---|---|---|
| Lớp CLIP được trích | Bài báo (theo trích xuất): **[3, 7, 9]**; config HF mặc định: **(3, 6, 9)** ([paper](https://arxiv.org/pdf/2112.10003); [HF docs](https://huggingface.co/docs/transformers/model_doc/clipseg)) | Đọc PDF gốc; mở `config.json` của `CIDAS/clipseg-rd64-refined`. HF docs còn ghi image 224 / patch 32, là giá trị mặc định chung chứ không phải của checkpoint |
| HSNet RN50 trên PASCAL-5i 1-shot | Bảng CLIPSeg ghi **64,0** / 76,7; bản trích từ bài HSNet ghi **60,8** / 73,3 ([paper](https://arxiv.org/pdf/2112.10003); [HSNet](https://arxiv.org/pdf/2104.01538)) | Xem Bảng 1 của HSNet |
| CLIPSeg COCO-20i | **33,3** mIoU (COCO+N, t=0,1) xuất hiện ở hai bản trích, nhưng một ghi chú yêu cầu xác nhận lại | Xem Bảng 6 trong PDF |
| PASCAL-5i "zero-shot" | 72,4 (bài gốc, text prompt) so với 51,7 (ClsCLIP) ([ClsCLIP](https://arxiv.org/pdf/2304.06212)) | Đọc kỹ protocol của từng bài; trình bày cả hai |
| HulaNet trên PhraseCut | 50,8 (bảng CLIPSeg) so với 50,2 cum-IoU (bài PhraseCut) | Có thể khác biến thể; trích dẫn nguyên từng nguồn |
| Định nghĩa mIoU trên PhraseCut của CLIPSeg | Trung bình theo lớp hay theo mẫu chưa rõ | Đọc mục 4 của bài báo |
| VRAM và số tham số SAM 3 | 848M (repo) / 0,9B (HF) / ~840M (Roboflow); "vừa 16 GB" ([Roboflow](https://roboflow.com/blog/what-is-sam3)) so với "12 GB đủ" ([Codersera](https://codersera.com/blog/how-to-run-sam-3-locally-2026/amp/)) | Tự đo trên T4 |
| Số liệu Prompt-SAM | Kết quả X-quang mơ hồ (CLIPSeg+SAM DSC 0,93 so với CLIPSeg 0,95) | Đọc bài trước khi trích ([Springer](https://link.springer.com/article/10.1007/s44230-025-00115-4)) |
| Venue của ZegFormer, ZegCLIP, OpenSeg, SAM 2, LISA | Chưa xác nhận từ trang gốc | Kiểm tra trên CVF/ECVA/OpenReview |
| Link trọng số owncloud, PyTorch hỗ trợ P100, trạng thái HF Spaces | Chưa kiểm chứng vào 10/2026 | Thử ở tuần 1 |

---

## Kết luận

Điểm mấu chốt của kế hoạch là phần đóng góp mới không cần đến mô hình mới. Nó nằm ở những ô trống mà tài liệu để lại: CLIPSeg chưa có số liệu công bố trên RefCOCO, template ensembling chưa được định lượng cho CLIPSeg, và chưa ai đo CLIPSeg ở mức mô hình trên các điểm yếu kế thừa từ CLIP (trái/phải, phủ định). Cả ba đều chỉ cần inference trên T4. Vì vậy nhóm nên dồn công sức vào **thiết kế đánh giá cẩn thận** (ngưỡng, protocol, quy ước chuyển instance thành một mask) thay vì vào huấn luyện. Đây cũng là phần giảng viên dễ đánh giá cao nhất.

Về câu chuyện, đồ án nên kết bằng một luận điểm rõ ràng thay vì "SAM 3 tốt hơn". CLIPSeg là bản thiết kế sớm của ý tưởng "text hoặc ảnh mẫu → mask", ý tưởng mà SEEM và SAM 3 về sau mở rộng lên mức instance và video. Thực nghiệm của nhóm cho thấy trong vùng nào thiết kế nhẹ 1,1M tham số ấy vẫn hữu dụng (prompt dạng vùng hoặc affordance, tạo nhãn nhanh, chạy trên CPU) và vùng nào nó đã bị vượt (biên sắc nét, nhiều instance, ngôn ngữ không gian).
