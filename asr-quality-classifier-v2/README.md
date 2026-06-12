# ASR Data Quality Classifier v2 (Task 2)

Dự án phân loại chất lượng các cặp âm thanh - văn bản (Audio-Transcript Quality Classifier) tiếng Việt hỗ trợ lọc dữ liệu phục vụ huấn luyện mô hình ASR.

Mục tiêu là xây dựng bộ phân loại nhị phân dự đoán nhãn: **1 (usable - dùng được)** hoặc **0 (unusable - không dùng được)**.

> [!IMPORTANT]
> **Ràng buộc quan trọng (Hard Constraints):**
> 1. **KHÔNG sử dụng mô hình ASR công khai** để chuyển đổi âm thanh thành văn bản tại thời điểm suy luận (Không gọi `Whisper.generate()`).
> 2. **KHÔNG tính toán WER/CER** (do không chạy ASR giải mã).
> 3. **KHÔNG sử dụng** các đặc trưng rò rỉ thông tin như `annotator_id` hay `label_source`.
> 4. Huấn luyện và phân chia dữ liệu theo cơ chế **GroupKFold** theo `folder` hoặc `transcript` để loại bỏ hoàn toàn hiện tượng rò rỉ dữ liệu (data leakage).

---

## 🛠 Cấu Trúc Thư Mục

```
asr-quality-classifier-v2/
├── data/
│   ├── audio/data2/       # Thư mục chứa các file .wav theo cấu trúc folder/file.wav
│   └── transcripts/       # Chứa file nhãn training.csv
├── src/
│   ├── __init__.py
│   ├── config.py          # Cấu hình tham số và tự động thiết lập CUDA/MPS/CPU
│   ├── data_loader.py     # Đọc dữ liệu, đồng bộ hóa Azure Blob, chia cụm dữ liệu
│   ├── audio_features.py  # Trích xuất 37 đặc trưng âm thanh và 6 đặc trưng chéo kênh
│   ├── audio_encoder.py   # Nhúng âm thanh bằng mô hình đóng băng WavLM
│   ├── text_encoder.py    # Nhúng văn bản bằng mô hình đóng băng PhoBERT
│   ├── cross_attention.py # Căn hàng đặc trưng chéo kênh (audio-text cross-attention)
│   ├── classifier.py      # Phân loại học sâu đa phương thức & Tabular LightGBM
│   ├── trainer.py         # Huấn luyện PyTorch với BCE loss, Pos Weight, Early Stopping
│   └── evaluator.py       # Tính toán F1 Macro, AUC, vẽ ma trận nhầm lẫn
├── experiments/
│   ├── phase1_eda.py      # Phase 1: Phân tích dữ liệu & trực quan hóa phân phối
│   ├── phase2_baseline.py # Phase 2: Huấn luyện LightGBM baseline trên đặc trưng thủ công
│   ├── phase3_deep_audio.py # Phase 3: Huấn luyện bộ mã hóa sâu âm thanh (WavLM)
│   ├── phase4_crossmodal.py # Phase 4: Huấn luyện mô hình Cross-Modal Fusion đầy đủ
│   └── phase5_ensemble.py # Phase 5: Tìm trọng số Ensemble tối ưu giữa các pha
├── tests/                 # Bộ test suite chạy offline hoàn toàn (mocked transformers)
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_audio_features.py
│   ├── test_audio_encoder.py
│   ├── test_text_encoder.py
│   ├── test_cross_attention.py
│   ├── test_classifier.py
│   ├── test_trainer.py
│   └── test_evaluator.py
├── infer.py               # Suy luận độc lập CLI
├── train.py               # Chạy toàn bộ pipeline tự động từ Phase 1 -> 5
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng Dẫn Cài Đặt

1. **Khởi tạo môi trường ảo Python và kích hoạt:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Trên macOS/Linux
   ```

2. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Chuẩn bị dữ liệu:**
   Đảm bảo dữ liệu huấn luyện được đặt đúng thư mục:
   - File nhãn: `data/transcripts/training.csv`
   - File âm thanh: `data/audio/data2/{folder}/{file_name}.wav`

   *Hoặc cấu hình biến môi trường `AZURE_SAS_TOKEN` để tự động đồng bộ hóa và tải xuống dữ liệu từ Azure Storage:*
   ```bash
   export AZURE_SAS_TOKEN="your_azure_sas_token"
   python train.py --sync-azure --phase 1
   ```

---

## 📈 Quy Trình Chạy Pipeline Huấn Luyện (Phases 1-5)

Chạy toàn bộ quy trình thí nghiệm tự động từ phân tích, huấn luyện mô hình Baseline, Deep Audio, Cross-Modal Fusion cho đến Ensemble tối ưu:
```bash
python train.py --all
```

Hoặc chạy đơn lẻ từng pha:
```bash
python train.py --phase 1  # Chỉ chạy EDA phân tích dữ liệu
python train.py --phase 2  # Chạy huấn luyện LightGBM Baseline (trích xuất & cache đặc trưng)
python train.py --phase 3  # Chạy huấn luyện nhánh Deep Audio (WavLM)
python train.py --phase 4  # Chạy huấn luyện mô hình Cross-Modal Fusion đầy đủ
python train.py --phase 5  # Chạy bộ tổng hợp Ensemble tối ưu hóa F1 Macro & xuất bảng so sánh
```

Mọi kết quả, bảng so sánh và biểu đồ ma trận nhầm lẫn sẽ được xuất ra thư mục `outputs/`.

---

## 🔍 Kiểm Tra Suy Luận (Inference)

Sử dụng script suy luận CLI độc lập để phân loại một file âm thanh và văn bản tương ứng:
```bash
python infer.py --audio "data/audio/data2/50000420251102141639_000_ee165aea-b295-453b-9060-689dd51f6abe/clone8.wav" --transcript "a-lô dạ em chào anh đồng ạ"
```

---

## 🧪 Chạy Kiểm Thử Tự Động (Tests)

Bộ kiểm thử được thiết kế chạy ngoại tuyến hoàn toàn (offline) bằng cách mock các mô hình HuggingFace, giúp xác minh nhanh tính toàn vẹn của mã nguồn:
```bash
pytest
```

Chạy kiểm thử có báo cáo độ bao phủ mã nguồn (coverage):
```bash
pytest --cov=src tests/
```
