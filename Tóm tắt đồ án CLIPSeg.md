# Giới thiệu đồ án: Phân đoạn ảnh bằng câu nhắc với CLIPSeg

*Biên soạn ngày 13/10/2026, dựa trên kế hoạch và mã nguồn hiện có trong dự án XLAV.*

## 1. Đồ án đang làm gì?

Đồ án nghiên cứu **phân đoạn ảnh theo yêu cầu của người dùng**: đưa vào một ảnh và một câu mô tả hoặc ảnh mẫu, sau đó xác định những pixel thuộc vùng cần tìm. Mô hình trung tâm là **CLIPSeg**. Hướng triển khai hiện tại gồm tự huấn luyện lại phần decoder, khảo sát ảnh hưởng của cách đưa ra yêu cầu và thử cải tiến cách kết hợp thông tin ảnh với văn bản.

Ví dụ, với ảnh có chiếc thìa và chiếc bát, người dùng nhập `"a spoon"` thì hệ thống cần tạo mặt nạ cho vùng chiếc thìa. Khi đổi thành `"a bowl"`, vùng cần phân đoạn thay đổi theo yêu cầu. Đây là ví dụ minh họa đầu vào và đầu ra; chất lượng thực tế phải được kiểm tra bằng dữ liệu có đáp án.

### 1.1. Đầu vào (Input) và đầu ra (Output)

**Đầu vào của hệ thống khi sử dụng** gồm hai thành phần:

| Thành phần | Ý nghĩa | Ví dụ |
|---|---|---|
| **Ảnh cần phân đoạn (query image)** | Ảnh mà hệ thống cần xác định vùng đối tượng | Ảnh có chiếc thìa và chiếc bát |
| **Yêu cầu phân đoạn (prompt)** | Chỉ dẫn đối tượng hoặc vùng cần tìm, dưới dạng văn bản hoặc ảnh mẫu | Câu `"a spoon"` hoặc ảnh mẫu chứa chiếc thìa |

Người dùng cung cấp ảnh cần phân đoạn và một trong hai dạng prompt. Ảnh mẫu đóng vai trò mô tả trực quan đối tượng cần tìm; mặt nạ kết quả được tạo trên **ảnh query**.

**Đầu ra chính** là mặt nạ phân đoạn ứng với yêu cầu. Hệ thống cũng xuất các kết quả hỗ trợ quan sát:

| Kết quả | Ý nghĩa |
|---|---|
| **Mặt nạ nhị phân (binary mask)** | Đánh dấu từng pixel thuộc vùng được yêu cầu hay phần còn lại; có cùng chiều cao và chiều rộng với ảnh query |
| **Bản đồ xác suất (probability map)** | Giá trị từ 0 đến 1 tại từng pixel, thể hiện dự đoán pixel đó thuộc vùng được yêu cầu |
| **Heatmap và ảnh chồng mặt nạ (overlay)** | Trực quan hóa dự đoán để người dùng xem vùng hệ thống đã chọn |

Ví dụ: **ảnh chiếc thìa và chiếc bát + prompt `"a spoon"` → mặt nạ đánh dấu vùng chiếc thìa trên ảnh đó**. Với nhiều prompt chữ, hệ thống tạo kết quả tương ứng cho từng prompt.

Mặt nạ đáp án (*ground truth*) là dữ liệu phục vụ huấn luyện và đánh giá. Khi người dùng đưa ảnh mới vào hệ thống để phân đoạn, họ chỉ cần cung cấp ảnh và prompt.

### 1.2. Framework chung của hệ thống

Ở mức chức năng, quy trình xử lý một yêu cầu gồm các công đoạn sau:

```text
Ảnh query + prompt
        ↓
Tiếp nhận và tiền xử lý dữ liệu
        ↓
Trích xuất / biểu diễn thông tin ảnh và prompt
        ↓
Kết hợp thông tin ảnh với yêu cầu
        ↓
Dự đoán vùng phân đoạn
        ↓
Hậu xử lý kết quả
        ↓
Xuất mặt nạ, bản đồ xác suất và ảnh trực quan
```

| Công đoạn | Vai trò |
|---|---|
| **1. Tiếp nhận và tiền xử lý** | Kiểm tra đầu vào, đưa ảnh và prompt về định dạng phù hợp để hệ thống xử lý |
| **2. Trích xuất / biểu diễn thông tin** | Chuyển nội dung ảnh và yêu cầu thành các biểu diễn mà mô hình có thể xử lý |
| **3. Kết hợp ảnh với yêu cầu** | Liên hệ thông tin trong ảnh với đối tượng hoặc vùng được mô tả bởi prompt |
| **4. Dự đoán phân đoạn** | Tạo dự đoán theo vị trí ảnh về vùng phù hợp với yêu cầu |
| **5. Hậu xử lý** | Đưa dự đoán về kích thước ảnh gốc và chuyển thành bản đồ xác suất, mặt nạ nhị phân |
| **6. Xuất và trực quan hóa** | Lưu kết quả và tạo heatmap, overlay để quan sát |

Framework này mô tả các chức năng cần có của hệ thống. Cách thực hiện từng khối bằng CLIPSeg được trình bày ở phần mô hình và phương pháp phía dưới.

Để xây dựng hệ thống trước khi sử dụng, đồ án còn có quy trình **chuẩn bị dữ liệu có đáp án → huấn luyện mô hình → đánh giá và phân tích kết quả → lưu mô hình để phân đoạn ảnh mới**. Quy trình xử lý một ảnh mới ở trên sử dụng mô hình đã được huấn luyện.

## 2. Tại sao cần cách tiếp cận này?

Trong nhiều bài toán phân đoạn, mô hình được huấn luyện cho một tập nhãn cố định. Khi yêu cầu thay đổi từ tên lớp sang một mô tả cụ thể, chẳng hạn thuộc tính hoặc bộ phận của vật, hệ thống cần cách biểu diễn yêu cầu linh hoạt hơn. Phân đoạn bằng prompt cho phép người dùng diễn đạt vùng cần tìm bằng chữ hoặc ví dụ trực quan. Đây cũng là động cơ của [bài báo CLIPSeg](https://arxiv.org/abs/2112.10003).

Cách tương tác này có thể phục vụ việc chọn vùng để chỉnh sửa ảnh, hỗ trợ tạo nhãn hoặc tìm đối tượng theo mô tả. Với đồ án này, mục tiêu trước mắt là hiểu cơ chế, tái tạo quá trình học và đánh giá khả năng của mô hình trên dữ liệu có mặt nạ đáp án.

Việc tự huấn luyện decoder giúp kiểm tra vai trò của dữ liệu, prompt và kiến trúc. Từ đó, đồ án có cơ sở thực nghiệm để giải thích mô hình hoạt động tốt hay thất bại trong trường hợp nào, đồng thời đánh giá một thay đổi kiến trúc có thực sự hữu ích hay không.

## 3. CLIPSeg là mô hình gì?

**CLIPSeg là mô hình phân đoạn nhị phân có điều kiện, xây dựng trên nền tảng mô hình thị giác–ngôn ngữ CLIP.** “Có điều kiện” nghĩa là kết quả phụ thuộc vào prompt; “nhị phân” nghĩa là mỗi pixel được xét thuộc vùng được yêu cầu hoặc thuộc phần còn lại. CLIPSeg được giới thiệu trong bài *Image Segmentation Using Text and Image Prompts* của Timo Lüddecke và Alexander Ecker, công bố tại CVPR 2022. [Nguồn tác giả](https://github.com/timojl/clipseg).

CLIP là viết tắt của *Contrastive Language–Image Pretraining*. Mô hình có bộ mã hóa ảnh và bộ mã hóa văn bản, học biểu diễn để ảnh và mô tả tương ứng có độ tương đồng cao. Năng lực này cung cấp nền tảng liên hệ nội dung thị giác với ngôn ngữ. [Mã nguồn và giới thiệu CLIP](https://github.com/openai/CLIP).

CLIPSeg bổ sung một **decoder Transformer** lên CLIP đóng băng để dự đoán theo pixel. Trong bản triển khai của đồ án, luồng xử lý có thể hiểu như sau:

```text
Ảnh cần phân đoạn → CLIP mã hóa ảnh → Đặc trưng ở nhiều lớp ──┐
                                                            ├→ FiLM → Decoder → Mask
Văn bản / ảnh support → CLIP mã hóa → Vector điều kiện ──────┘
```

**FiLM** (*Feature-wise Linear Modulation*) dùng vector điều kiện để điều chỉnh đặc trưng ảnh thông qua hệ số nhân và hệ số cộng. Nhờ đó, cùng một ảnh có thể được xử lý theo những yêu cầu khác nhau. Decoder tiếp tục tổng hợp đặc trưng và tăng độ phân giải để tạo dự đoán. [Tài liệu CLIPSeg của Hugging Face](https://huggingface.co/docs/transformers/model_doc/clipseg).

Dự án sử dụng kiến trúc theo cấu hình `CIDAS/clipseg-rd64-refined`: backbone CLIP **ViT-B/16**, đặc trưng lấy từ các lớp **[3, 6, 9]**, chiều biểu diễn decoder **64**, và đầu tăng độ phân giải kiểu refined. ViT xử lý ảnh thành các patch; `B/16` chỉ biến thể Base với patch 16 × 16 pixel. Các thiết lập của đồ án được mô tả trong [hướng dẫn thực nghiệm](source/README.md) và triển khai ở [script E1](scripts/build_baseline_notebook.py).

Một prompt tạo một mặt nạ cho vùng phù hợp; nếu nhiều vật cùng phù hợp, mặt nạ có thể chứa nhiều vật. Để đánh giá khả năng tách từng cá thể riêng biệt cần một giao thức khác với mặt nạ hợp vùng đang dùng trong đồ án.

## 4. Phương pháp triển khai của đồ án

### Dữ liệu và cách huấn luyện

Đồ án dùng **PhraseCut**, gồm ảnh, cụm từ mô tả và vùng tương ứng. Polygon của các vùng được chuyển thành mặt nạ đáp án (*ground truth*). Nếu một cụm từ trỏ tới nhiều đối tượng, đáp án là hợp các vùng đó.

Quá trình huấn luyện tái tạo công thức **PhraseCut+** từ mã nguồn tác giả. Trong dự án, PhraseCut+ được sinh trong lúc lấy mẫu từ PhraseCut, gồm crop ảnh, thay đổi mẫu câu, lấy ảnh support từ ảnh khác có cùng cụm từ, trộn biểu diễn chữ–ảnh và thêm khoảng 20% mẫu âm có mặt nạ rỗng. Các bước này lần lượt nhằm tăng sự đa dạng, cho mô hình học cả hai dạng prompt và học cách phản hồi khi yêu cầu không xuất hiện trong ảnh. Chi tiết nằm trong [hướng dẫn PhraseCut+](source/README.md).

Ở thí nghiệm chính, **CLIP giữ trọng số đã học và được đóng băng; decoder được khởi tạo ngẫu nhiên rồi huấn luyện**. Cách này tận dụng biểu diễn ảnh–chữ sẵn có, giảm số tham số phải cập nhật và phù hợp hơn với ngân sách tính toán của đồ án. Decoder học bằng loss **BCE** theo pixel, optimizer **AdamW** và lịch giảm learning rate **cosine**. Checkpoint tác giả được đánh giá riêng làm mốc tham chiếu.

### Các thí nghiệm và câu hỏi cần trả lời

| Thí nghiệm | Công việc | Mục đích |
|---|---|---|
| **E1** | Huấn luyện decoder với PhraseCut+, đánh giá cùng giao thức với checkpoint tác giả | Tái tạo quá trình học và xây dựng mô hình nền của đồ án |
| **E1b** | Huấn luyện chỉ với prompt chữ | Kiểm tra ảnh hưởng của việc học thêm ảnh support |
| **E2** | So sánh prompt chữ với các cách xử lý ảnh support | Xác định loại prompt và cách chuẩn bị ảnh mẫu ảnh hưởng ra sao |
| **E3** | So sánh các mẫu câu diễn đạt cùng một yêu cầu | Kiểm tra độ nhạy của mô hình với cách viết prompt |
| **E4** | Thử bổ sung attention theo token chữ vào decoder | Kiểm tra giả thuyết rằng thông tin ngôn ngữ chi tiết hơn có thể cải thiện phân đoạn |

E2 và E3 sử dụng decoder đã huấn luyện ở E1 để kết quả phản ánh cùng một mô hình nền. Các notebook hiện có tại [thư mục source](source/README.md).

### Hướng cải tiến E4 và lý do lựa chọn

CLIPSeg nền sử dụng vector tổng hợp của prompt để điều kiện hóa đặc trưng ảnh. **Giả thuyết của E4** là biểu diễn tổng hợp có thể chưa khai thác đủ các thành phần mô tả như đối tượng, thuộc tính và quan hệ. Vì vậy, phương án chính giữ FiLM và thêm **word-level cross-attention** vào decoder: token ảnh làm query, token chữ làm key/value, giúp từng vị trí ảnh tham chiếu tới các phần của mô tả. “Word-level” ở đây được triển khai trên token của bộ tokenizer, không nhất thiết tương ứng một token cho mỗi từ.

Nhánh mới dự kiến có gate khởi tạo bằng 0 để bắt đầu từ hành vi của mô hình nền rồi học dần đóng góp của attention. Khi chỉ có ảnh support, cần thiết kế đường xử lý phù hợp và đánh giá lại E2 để kiểm tra khả năng dùng visual prompt. Đây là **hướng đề xuất**, chưa được xác nhận bằng kết quả của đồ án. Thiết kế chi tiết nằm trong [nghiên cứu E4](reports/Cải%20tiến%20CLIPSeg%20cho%20E4.md).

Để đánh giá công bằng, mô hình E4 phải được so với mô hình nền **huấn luyện tiếp cùng số bước**, dùng cùng dữ liệu và cấu hình tương ứng. Như vậy mới phân biệt được tác dụng của module mới với tác dụng của việc huấn luyện thêm. Phương án dự phòng là tăng chiều decoder từ 64 lên 128 và huấn luyện lại.

## 5. Đánh giá như thế nào?

Theo pipeline hiện có, logits được nội suy về kích thước ảnh gốc rồi chuyển qua sigmoid thành bản đồ xác suất. Ngưỡng chính **0,3** tạo mặt nạ nhị phân. Ngưỡng bổ sung được chọn trên phần calibration và áp dụng lên phần evaluation riêng.

Chỉ số chính là **IoU**: diện tích giao giữa mặt nạ dự đoán và đáp án chia cho diện tích hợp. **Mean IoU** lấy trung bình theo từng yêu cầu; **cumulative IoU** cộng giao và hợp trên toàn bộ yêu cầu trước khi chia, nên vùng lớn có trọng số cao hơn. Dự án còn tính Dice và AP theo pixel, đồng thời xem ảnh minh họa để phân tích lỗi.

Dữ liệu train và đánh giá được kiểm tra tách biệt theo ảnh. Notebook hỗ trợ `miniv`, `val` và `test`; miniv phục vụ kiểm tra nhanh, còn số liệu cuối cần ghi rõ split thực sự đã chạy. Kết quả trên miniv không thể so trực tiếp với kết quả bài báo trên test. Vì decoder học trên PhraseCut, đánh giá PhraseCut trong đồ án không được mặc nhiên gọi là zero-shot. [Giao thức hiện có](source/README.md).

## 6. Hiện trạng và giá trị dự kiến

Trong workspace đã có mã suy luận bằng prompt chữ hoặc ảnh support, notebook E1/E2/E3 và các kết quả chạy kiểm tra trên CPU. Các tài liệu hiện có chưa ghi nhận benchmark GPU đầy đủ; E4 đang ở mức nghiên cứu và thiết kế. Kết quả chạy kiểm tra code không được dùng thay cho số liệu thực nghiệm chính thức.

Giá trị dự kiến của đồ án gồm một quy trình huấn luyện–đánh giá có thể chạy lại, phân tích vai trò của prompt và dữ liệu PhraseCut+, cùng một thí nghiệm cải tiến có đối chứng. Những hạn chế như biên mask chưa chính xác, bỏ sót vật nhỏ hoặc hiểu sai mô tả cần được phân tích qua kết quả thực tế. Nếu E4 không cải thiện, kết quả đó vẫn giúp đánh giá giả thuyết và chỉ ra hướng nghiên cứu tiếp theo.

## Đoạn giới thiệu có thể dùng khi thuyết trình

> Đồ án của em nghiên cứu phân đoạn ảnh dựa trên câu nhắc văn bản hoặc ảnh mẫu, với mô hình trung tâm là CLIPSeg. Mục tiêu là xác định vùng ảnh mà người dùng yêu cầu và tạo mặt nạ ở mức pixel. CLIPSeg kết hợp khả năng biểu diễn ảnh–chữ của CLIP với một decoder Transformer để tạo dự đoán phân đoạn. Em giữ CLIP đóng băng, tự huấn luyện lại decoder theo công thức PhraseCut+, sau đó khảo sát prompt chữ, ảnh support và cách diễn đạt câu. Hướng cải tiến dự kiến là bổ sung cross-attention theo token chữ để kiểm tra liệu việc kết hợp thông tin ngôn ngữ chi tiết hơn có giúp phân đoạn tốt hơn hay không. Hiệu quả sẽ được đánh giá bằng IoU và so sánh với mô hình nền huấn luyện cùng số bước.
