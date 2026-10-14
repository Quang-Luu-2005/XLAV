# Paper cần đọc: CLIPSeg

Timo Lüddecke và Alexander S. Ecker, **Image Segmentation Using Text and Image Prompts**, CVPR 2022.

- [PDF chính thức](https://openaccess.thecvf.com/content/CVPR2022/papers/Luddecke_Image_Segmentation_Using_Text_and_Image_Prompts_CVPR_2022_paper.pdf)
- Bản tải về: `references/CLIPSeg_CVPR2022.pdf`.
- [Bản dịch tiếng Việt](../references/clipseg-vi/CLIPSeg_CVPR2022-vi.pdf) bằng skill VI-Translate; [phạm vi dịch và thuật ngữ](../references/clipseg-vi/README.md). Chữ trong hình, một số nhãn bảng và tài liệu tham khảo giữ tiếng Anh.
- [Code tác giả](https://github.com/timojl/clipseg)
- [Hướng dẫn HF 4.55.4](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/clipseg): tài liệu triển khai, không phải paper thứ hai cần đọc.

Đây là bài chính của mô hình đang dùng. Mục tiêu đọc là tự giải thích được luồng xử lý và thiết kế thực nghiệm. Chưa cần đọc hết các bài trong kế hoạch gốc.

| Lượt | Thời gian | Nội dung | Việc ghi lại |
|---|---:|---|---|
| 1 | 20 phút | Abstract, Introduction, hình kiến trúc | Input/output, bài toán, đóng góp |
| 2 | 60 phút | Mục 3 và kiến trúc | CLIP đóng băng, conditioning, FiLM, decoder |
| 3 | 35 phút | Visual prompt engineering | Ảnh support, crop, tác động nền, protocol |
| 4 | 45 phút | Thực nghiệm và bảng kết quả | Dataset, split, checkpoint, ngưỡng và metric |
| 5 | 20 phút | Kết luận + ghi chú | Hạn chế và liên hệ E1–E3 |

Trả lời 6 câu bằng lời của bạn, mỗi câu khoảng 3–5 dòng:

1. CLIP ban đầu tạo embedding; CLIPSeg thêm gì để tạo mask?
2. Những thành phần nào đóng băng, thành phần nào được huấn luyện?
3. FiLM dùng vector điều kiện như thế nào? Text prompt và ảnh mẫu gặp nhau ở đâu?
4. Vì sao crop hoặc giảm ảnh hưởng nền có thể giúp ảnh mẫu?
5. Paper đánh giá những tác vụ nào, và protocol nào khác thí nghiệm của mình?
6. Hai hạn chế nào cần kiểm tra bằng hình hoặc số liệu của đồ án?

Mẫu ghi chú 1 trang: bài toán → sơ đồ → 3 ý về kiến trúc → 1 bảng kết quả có nguồn → 2 hạn chế → giả thuyết E2/E3. Để trống phần kết quả tự đo cho tới khi chạy thật.

Khi viết báo cáo, đối chiếu cấu hình checkpoint thật: `extract_layers=[3,6,9]`, `reduce_dim=64`, `use_complex_transposed_convolution=true` trong [config HF](https://huggingface.co/CIDAS/clipseg-rd64-refined/blob/main/config.json). Số tham số đo từ code có thể khác biến thể paper; không sao chép sang nhau.
