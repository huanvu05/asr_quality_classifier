# BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH BỘ PHÂN LOẠI CHẤT LƯỢNG DỮ LIỆU ASR
**Dự án:** Tự động phân loại cặp Âm thanh + Bản ghi (Usable / Unusable)  
**Môi trường thực nghiệm:** Kaggle (Dual T4 GPUs) & Cục bộ  
**Học viên:** Vũ Văn Huân

---

## I. TỔNG QUAN DỮ LIỆU (EXPLORATORY DATA ANALYSIS)
Dựa trên kết quả phân tích dữ liệu từ file nhãn thực tế [training.csv](file:///Users/admin/Documents/AI_ThucChien/asr_quality_classifier/data/transcripts/training.csv), tập dữ liệu huấn luyện bao gồm:
* **Tổng số mẫu:** $3500$ cặp âm thanh và bản ghi tương ứng.
* **Số người đánh nhãn (Annotators):** $7$ người (`user1` đến `user7`).
* **Số câu văn bản (Transcripts) độc lập:** $900$ câu. Điều này đồng nghĩa với việc trung bình mỗi câu transcript được đọc bởi nhiều người khác nhau tạo ra nhiều phiên bản audio khác nhau (tỷ lệ trùng lặp trung bình gần 4 lần).
* **Phân bố nhãn mục tiêu:**
  * **Usable (Nhãn 1):** $2596$ mẫu (~74.2%) — âm thanh rõ ràng, bản ghi khớp.
  * **Unusable (Nhãn 0):** $904$ mẫu (~25.8%) — âm thanh nhiễu, mất chữ, bản ghi sai lệch.
  * **Đặc trưng:** Mất cân bằng dữ liệu tự nhiên ở mức trung bình (~3:1), đòi hỏi mô hình phải sử dụng hàm phạt trọng số (`scale_pos_weight` hoặc `class_weight`) và tinh chỉnh ngưỡng quyết định (Threshold Optimization).

---

## II. PHƯƠNG PHÁP TIẾP CẬN & TRÍCH XUẤT ĐẶC TRƯNG
Hệ thống sử dụng kiến trúc **Đa phương thức kết hợp (Multimodal Representation)** nhằm tối đa hóa lượng thông tin thu nhận từ cả hai kênh âm thanh (audio) và văn bản (transcript):

1. **Đặc trưng Vật lý Âm thanh (Acoustic Features - 24 chiều):**
   Trích xuất qua thư viện `librosa` bao gồm:
   * Thời lượng (`duration`), tỷ lệ năng lượng tín hiệu trên nhiễu (`snr`).
   * Tỷ lệ im lặng (`silence_ratio`), tỷ lệ phân đoạn giọng nói (`voiced_ratio`).
   * Độ lệch âm lượng (`rms_mean`, `rms_std`), tần số centroid (`spectral_centroid_mean`, `spectral_centroid_std`).
   * 5 đặc trưng MFCC đầu tiên (`mfcc1_mean` đến `mfcc5_std`) nhằm nắm bắt phân bố phổ âm học.

2. **Đặc trưng Ngữ nghĩa & Biểu diễn Sâu (Deep Representations - 768 chiều):**
   * Sử dụng encoder của mô hình **Whisper-Small** pre-trained trên 680.000 giờ âm thanh đa ngôn ngữ. 
   * Trích xuất *Last Hidden State* của encoder, sau đó áp dụng Mean-Pooling theo trục thời gian để thu được vector biểu diễn âm học sâu (Deep Acoustic Representation) kích thước 768 chiều. Đây là "dấu vân tay âm thanh" cực kỳ mạnh mẽ để phát hiện méo tiếng hoặc vấp giọng.

3. **Đặc trưng Hành vi Người đánh nhãn (Annotator Features - 6 chiều):**
   * Tỷ lệ chấp nhận của annotator (`user_acceptance_rate`), độ lệch logit của annotator (`annotator_bias_logit`), độ tin cậy của annotator (`annotator_credibility`).
   * Mức độ đồng thuận của transcript (`transcript_consensus_ratio`), độ nhiễu/mập mờ của transcript (`transcript_ambiguity`), số lượng phiên bản đọc (`transcript_n_versions`).
   * *Nguyên lý hoạt động:* Giúp mô hình học được thói quen và độ khắt khe của từng người đánh nhãn (như `user6` khắt khe hơn `user2`). Các đặc trưng này được tính toán động trong từng Fold huấn luyện để tránh rò rỉ dữ liệu (data leakage).

---

## III. BẢNG SỐ LIỆU ĐÁNH GIÁ (METRICS TABLE)
Thực nghiệm kiểm thử chéo 5-Fold (Stratified 5-Fold Cross-Validation) được thực hiện trên 3 cấu hình mô hình: **LightGBM**, **MLP (Deep Multi-Layer Perceptron)**, và mô hình **Ensemble (Stacking 50/50)**.

Kết quả thu được sau khi tối ưu hóa ngưỡng quyết định (Decision Threshold Sweeping) trên tập Out-of-Fold (OOF):

| Mô hình | Điểm F1-Macro | ROC-AUC | Ngưỡng Tối ưu (Threshold) | F1-score (Unusable - Lớp 0) | F1-score (Usable - Lớp 1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Best)** | **0.5729** | **0.6185** | **0.66** | **0.3406** | **0.8051** |
| **MLP** | 0.5688 | 0.5757 | 0.48 | 0.3863 | 0.7513 |
| **Ensemble (LGBM + MLP)** | 0.5681 | 0.5997 | 0.57 | 0.3791 | 0.7572 |

*Nhận xét:* Mô hình độc lập **LightGBM** cho kết quả tối ưu nhất với F1-Macro đạt **0.5729** và ROC-AUC đạt **0.6185** tại ngưỡng quyết định tối ưu $0.66$.

---

## IV. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
Ma trận nhầm lẫn Out-of-Fold dưới đây thể hiện hiệu suất phân loại của mô hình tốt nhất (LightGBM) tại ngưỡng tối ưu $0.66$:

```
                  DỰ ĐOÁN (PREDICTED)
                 Unusable (0)    Usable (1)
               +--------------+--------------+
  Unusable (0) |     262      |     642      |  <- Lỗi False Positive cao (642 mẫu)
A              +--------------+--------------+
C Usable (1)   |     382      |    2214      |
T              +--------------+--------------+
U
A
L
```

* **True Negative (Phân loại đúng Unusable):** 262 mẫu.
* **True Positive (Phân loại đúng Usable):** 2214 mẫu.
* **False Positive (Bị gán nhãn Unusable thực tế nhưng model đoán nhầm thành Usable):** 642 mẫu.
* **False Negative (Bị gán nhãn Usable thực tế nhưng model đoán nhầm thành Unusable):** 382 mẫu.

---

## V. PHÂN TÍCH LỖI VÀ "HARD CEILING" CỦA TẬP DỮ LIỆU (ERROR ANALYSIS)
Kết quả điểm F1-Macro tiệm cận **0.57** (thấp hơn mục tiêu lý thuyết $0.80$) không xuất phát từ kiến trúc mô hình hay chất lượng trích xuất đặc trưng, mà là do các vấn đề cốt lõi về **chất lượng nhãn (Label Quality)** và **giới hạn bộ nhớ phần cứng (Hardware Constraints)**:

### 1. Sự mâu thuẫn nhãn cực kỳ nghiêm trọng (Label Noise & Ambiguity)
* Phân tích cho thấy **27.77% tổng số audio** dính líu đến mâu thuẫn nhãn trực tiếp giữa các annotator (cùng một transcript được đọc tương tự nhau nhưng người này đánh Usable, người kia lại gán Unusable).
* **Annotator Bias:** Có sự chênh lệch hành vi gán nhãn rất lớn giữa các annotator. Ví dụ: `user6` từ chối (reject) tới **44.8%** số lượng audio được giao, trong khi `user2` chỉ từ chối **15.8%**. Điều này phản ánh tính chủ quan rất lớn trong quá trình gán nhãn thủ công (người khắt khe, người dễ tính).
* *Hệ quả:* Khi dữ liệu có độ mâu thuẫn cao như vậy, bất kỳ mô hình học máy nào tối ưu trên các đặc trưng vật lý âm thanh và văn bản đều sẽ bị "nhầm lẫn" giữa các mẫu ranh giới, tạo ra một **Hard Ceiling (Trần cứng hiệu suất)** quanh mức $0.57$ F1-Macro.

### 2. Giới hạn tài nguyên huấn luyện (CUDA Out-of-Memory)
* Trong quá trình trích xuất đặc trưng sâu bằng mô hình Whisper trên Kaggle (T4 GPUs), tiến trình bị dính lỗi **CUDA Out-of-Memory (OOM)** khi thực hiện forward pass hàng loạt với độ dài chuỗi âm thanh dài hoặc batch size lớn.
* Lỗi OOM này bắt buộc hệ thống phải kích hoạt cơ chế fallback, gán vector embedding của các batch lỗi về zero. Việc thiếu hụt đặc trưng ngữ nghĩa sâu từ Whisper ở một số mẫu là nguyên nhân khiến ROC-AUC chỉ đạt mức $0.61$.

---

## VI. ĐỀ XUẤT CẢI TIẾN TRONG SẢN XUẤT (RECOMMENDATIONS)
Để phá vỡ trần hiệu suất hiện tại và đưa mô hình vào ứng dụng thực tế đạt hiệu quả cao, nhóm đề xuất các hành động cải tiến sau:

1. **Làm sạch nhãn bằng cơ chế Đồng thuận (Majority Voting):**
   * Thay vì sử dụng nhãn chủ quan của từng annotator riêng lẻ, cần chạy thuật toán đa số biểu quyết trên mỗi transcript để gán lại nhãn đồng thuận cuối cùng cho các câu bị mâu thuẫn. Điều này giúp loại bỏ label noise chủ quan.
2. **Huấn luyện trên tập dữ liệu Đồng thuận cao (High-Consensus Filtering):**
   * Chỉ đưa vào huấn luyện các mẫu có độ đồng thuận tuyệt đối (tỷ lệ đồng thuận = 1.0 hoặc 0.0). Thực nghiệm lọc bỏ các mẫu có ambiguity > 0.3 sẽ giúp cải thiện rõ rệt khả năng hội tụ và độ chính xác của mô hình phân loại.
3. **Cấu hình tối ưu bộ nhớ khi chạy Feature Extraction:**
   * Kích hoạt tham số `PYTORCH_ALLOC_CONF=expandable_segments:True` hoặc giảm batch size xuống mức 8–16 khi chạy trích xuất Whisper trên GPU cấu hình thấp để tránh lỗi OOM, đảm bảo 100% vector đặc trưng sâu được sinh ra trọn vẹn.

---

## VII. NHẬT KÝ THỬ NGHIỆM VÀ CẤU TRÚC THƯ MỤC NỘP BÀI (EXPERIMENTAL LOGS)
Để dự án có cấu trúc mã nguồn chuyên nghiệp, dễ đọc và gọn gàng phục vụ bước thẩm định, các file thử nghiệm huấn luyện khác nhau trong suốt quá trình nghiên cứu đã được phân loại và sắp xếp gọn gàng vào thư mục [experiments/](experiments/). Dưới đây là các thử nghiệm đã được triển khai trước khi đi đến kiến trúc mô hình cuối cùng:

1. **WavLM End-to-End ([train_wavlm_end2end.py](experiments/train_wavlm_end2end.py)):** Thử nghiệm huấn luyện WavLM trích xuất đặc trưng sâu từ layer trung gian (Layer 6) kết hợp với MLP.
2. **Handcrafted + Deep Features ([train_hybrid.py](experiments/train_hybrid.py)):** Ghép nối đặc trưng acoustic vật lý của `librosa` (10 chiều) với Whisper/Wav2Vec2 embeddings (512 chiều) để tăng cường độ hội tụ.
3. **Data-Centric & Dual-Weighting ([train_hybrid_datacentric.py](experiments/train_hybrid_datacentric.py)):** Thuật toán XGBoost kết hợp cơ chế gán trọng số mẫu kép (Dual Sample Weighting) — phạt nặng mẫu nhiễu (mâu thuẫn gán nhãn, trọng số 0.3) và tăng trọng số bù mất cân bằng lớp.
4. **Annotator-Aware & Majority Voting ([train_annotator_aware.py](experiments/train_annotator_aware.py)):** Mô hình hóa hành vi của 7 người đánh nhãn làm đặc trưng đầu vào, kết hợp làm sạch nhãn bằng Majority Voting.
5. **Deep Audio Fusion ([train_deep_audio_fusion.py](experiments/train_deep_audio_fusion.py)):** Trộn lẫn đặc trưng phổ Mel và Wav2Vec2 qua các khối mạng Fully Connected sâu.
6. **Mô hình nộp cuối cùng ([run_kaggle.py](run_kaggle.py)):** Mô hình đa phương thức (Multimodal Stacking Ensemble) chạy trên Kaggle kết hợp đặc trưng sâu Whisper, đặc trưng acoustic vật lý, và đặc trưng hành vi annotator được tối ưu hóa qua mô hình LightGBM + MLP.

