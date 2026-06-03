CHƯƠNG TRÌNH PHÁT TRIỂN ASR QUALITY CLASSIFIER (PROD-READY BLUEPRINT)

Tài liệu này cung cấp toàn bộ kiến trúc mã nguồn, quy trình quản lý phiên bản mô hình (Model Version Control), cách thức triển khai mượt mà trên Google Colab/Kaggle và một Siêu Prompt được tối ưu hóa để ra lệnh cho AI Agent tự động viết code chất lượng cao.

1. Cấu trúc mã nguồn chuẩn Production (Modular & Clean Code)

Để đạt điểm tối đa ở tiêu chí Code Quality (25%) và Reproducibility (20%), dự án không được viết dưới dạng một file notebook dài dòng, mà cần được tổ chức thành các mô-đun Python rõ ràng.

Dưới đây là sơ đồ cấu trúc thư mục mà chúng ta sẽ xây dựng trực tiếp trên Colab/Kaggle thông qua cơ chế tự động ghi file (%%writefile):

asr_quality_classifier/
│
├── data/                         # Thư mục chứa dữ liệu tạm thời (được gitignore)
│   ├── audio/                    # File .wav / .mp3 tải về từ Azure Blob
│   ├── transcripts/              # File .txt chứa văn bản gốc
│   └── labels.csv                # File nhãn (file_id/file_path, label 0/1)
│
├── src/                          # Thư mục mã nguồn chính
│   ├── __init__.py
│   ├── config.py                 # Quản lý cấu hình, hyperparams, kiểm tra AZURE_SAS_TOKEN
│   ├── data_loader.py            # Kết nối Azure Blob Storage, tải dữ liệu đa luồng (Multi-threading)
│   ├── preprocessor.py           # Chuẩn hóa Text (num2words), resample Audio về 16kHz
│   ├── features.py               # Trích xuất đặc trưng: SNR, Silence Ratio, Whisper Confidence, WER/CER
│   ├── model.py                  # Khởi tạo mô hình (LightGBM/XGBoost) cấu hình scale_pos_weight
│   ├── evaluator.py              # Stratified 5-Fold, Threshold Optimizer (quét F1), Confusion Matrix
│   └── register.py               # Đóng gói Pipeline (Model + Scaler + Threshold) & Đẩy lên Azure Blob
│
├── main.py                       # Script chạy huấn luyện toàn bộ pipeline (End-to-End Train)
├── inference.py                  # Script chạy suy luận nhanh cho 1 mẫu đơn lẻ (Single Inference)
├── requirements.txt              # Danh sách thư viện và phiên bản chính xác
└── README.md                     # Hướng dẫn thiết lập môi trường, huấn luyện và suy luận


2. Chiến lược Quản lý phiên bản Mô hình (Model Version Control) trên Colab/Kaggle

Khi làm việc trên Colab/Kaggle, môi trường runtime sẽ bị xóa sạch sau khi ngắt kết nối. Do đó, chúng ta cần một cơ chế Model Versioning tự động đẩy ngược lên Azure Blob Storage (vùng lưu trữ cá nhân của bạn <your_name>/models/).

Nguyên tắc đặt tên phiên bản (Versioning Convention)

Mỗi lần chạy thực nghiệm (Run) sẽ được gán một mã định danh duy nhất:
RUN_ID = run_{YYYYMMDD_HHMMSS}_{ALGO_NAME}_{F1_SCORE}
Ví dụ: run_20260603_093015_lightgbm_f1_0_84

Các thành phần lưu trữ trong một Package Phiên bản

Tại mỗi phiên bản, hệ thống sẽ tự động đóng gói các tệp tin sau thành một file .zip hoặc lưu trữ trong một thư mục có cấu trúc trên Azure Blob:

pipeline_artifacts.pkl: Chứa toàn bộ Pipeline được đóng gói bằng joblib/pickle, bao gồm:

Bộ chuẩn hóa đặc trưng (e.g., StandardScaler).

Trọng số mô hình phân loại đã tối ưu (e.g., LightGBMClassifier).

Ngưỡng quyết định tối ưu (optimal_threshold) thu được sau bước quét F1 trên tập Out-of-Fold.

metadata.json: Chứa thông tin chi tiết về thực nghiệm:

{
  "run_id": "run_20260603_093015_lightgbm_f1_0_84",
  "timestamp": "2026-06-03T09:30:15",
  "best_params": {"learning_rate": 0.05, "n_estimators": 200, "scale_pos_weight": 2.88},
  "metrics": {
    "val_macro_f1": 0.842,
    "val_roc_auc": 0.895,
    "val_precision_class_0": 0.79,
    "val_recall_class_0": 0.81
  },
  "features_used": ["snr", "silence_ratio", "whisper_confidence", "wer", "cer", "length_ratio"],
  "system": {"python_version": "3.10.12", "cuda_available": true}
}


confusion_matrix.png: Biểu đồ ma trận nhầm lẫn của tập Validation (Out-of-Fold) để trực quan hóa hiệu suất phân loại lớp thiểu số (Nhãn 0 - Unusable).

3. Cách thức vận hành Dự án trên Google Colab/Kaggle

Vì bạn chạy trên nền tảng đám mây, cách tốt nhất là sử dụng một Master Notebook để kiểm soát và điều phối toàn bộ mã nguồn dạng mô-đun.

Quy trình chạy trên Colab/Kaggle:

Khởi động: Tạo một Notebook mới trên Colab, chọn Runtime là T4 GPU (để tăng tốc độ chạy mô hình nhận dạng Whisper phục vụ trích xuất đặc trưng).

Tải mã nguồn:

Cách 1 (Khuyên dùng): Bạn đẩy thư mục asr_quality_classifier lên một repo GitHub riêng tư (Private Repo), sau đó trong Colab chỉ cần clone về:

!git clone https://<your_token>@[github.com/username/asr_quality_classifier.git](https://github.com/username/asr_quality_classifier.git)
%cd asr_quality_classifier


Cách 2 (Mô-đun hóa trực tiếp bằng Notebook): Sử dụng lệnh %%writefile ở đầu mỗi cell để sinh ra các file .py cấu trúc như sơ đồ trên. (Cách này giúp bạn chỉnh sửa code cực nhanh mà không cần push/pull git liên tục).

Cấu hình môi trường:

import os
os.environ["AZURE_SAS_TOKEN"] = "sv=2021-08-06&ss=b&srt=co&sp=rwdlacit&..."


Cài đặt thư viện:

!pip install -r requirements.txt


Kích hoạt Pipeline huấn luyện:

!python main.py


Kiểm tra kết quả: Tải trực tiếp file metadata.json và confusion_matrix.png về Colab để xem báo cáo hoặc truy cập vào Azure Blob để lấy artifact đã được lưu trữ tự động.

4. SIÊU PROMPT DÀNH CHO AI AGENT ĐỂ CODE END-TO-END PROJECT

HƯỚNG DẪN SỬ DỤNG: Hãy sao chép toàn bộ phần khung dưới đây và dán vào một AI Agent lập trình chuyên nghiệp (như Cursor, GitHub Copilot, hoặc chính phiên bản GPT/Gemini chuyên Code của bạn). Prompt này được viết bằng tiếng Anh chuẩn hóa kỹ thuật cao để đảm bảo AI Agent sinh code chuẩn xác, không bị lỗi cú pháp hoặc thiếu mô-đun.

# ROLE & GOAL
You are an Elite AI/ML Engineer with 10+ years of experience in Speech AI and Production-grade MLOps pipelines.
Your goal is to write a highly modular, clean, documented, and fully reproducible Python project for an "ASR Data Quality Classifier" based on a hybrid Feature Extraction + LightGBM/XGBoost approach.

# PERFORMANCE TARGET
- Achieve Validation Macro F1-score >= 0.80 and High ROC-AUC.
- Resolve severe class imbalance (2600 usable (1) vs 900 unusable (0) samples) through cost-sensitive learning (class weights) and decision threshold tuning.

# CRITICAL CONSTRAINTS & TECHNICAL REQUIREMENTS
1. DO NOT hardcode any credentials. Read Azure SAS token from the environment variable `AZURE_SAS_TOKEN`.
2. Target Platform: Google Colab / Kaggle (compatible with Python 3.9+, utilizes T4 GPU for ASR feature extraction).
3. Fully Modular: Code must be split into specific files matching the planned architecture. Avoid a single monolithic block.
4. Ensure reproducibility by setting global seeds (`42`) across NumPy, PyTorch, LightGBM, and Random.

---

# ARCHITECTURE & FILE-BY-FILE SPECIFICATION

Please generate the complete, self-contained, and bug-free code for the following files:

## 1. `requirements.txt`
Include standard production-ready packages with exact versions:
`azure-storage-blob`, `numpy`, `pandas`, `librosa`, `soundfile`, `transformers`, `torch`, `lightgbm`, `xgboost`, `scikit-learn`, `jiwer`, `num2words`, `joblib`, `matplotlib`, `seaborn`.

## 2. `src/config.py`
- Setup a `Config` class holding all hyperparameters and paths.
- Check and validate the existence of `AZURE_SAS_TOKEN` using `os.getenv`. Raise clear exceptions if missing.
- Configuration for: sample rate (16000), seed (42), audio directory path, models directory path, cross-validation folds (5), LightGBM hyperparameters (including `scale_pos_weight=2.88` representing ratio of 2600/900).

## 3. `src/data_loader.py`
- Implements `AzureBlobDownloader` using `azure-storage-blob`.
- Use multi-threading to parallelly download audio files (`.wav` or `.mp3`) and `.txt` transcript files to local cache directories under `data/`.
- Read and parse the labels CSV/JSON. Map the columns correctly: `folder` and `file_name` to formulate the local audio path, and read the `transcript` text column. Return a cleaned Pandas DataFrame.

## 4. `src/preprocessor.py`
- Implements text normalization: lowercase, remove punctuation, strip whitespaces.
- Implement a `normalize_numbers(text)` function using `num2words` to convert numeric digits (like "99") into Vietnamese words ("chín mươi chín") to resolve writing style discrepancies.
- Implement audio resampler that loads any audio format using `librosa` or `soundfile` and standardizes it to 16kHz, mono-channel.

## 5. `src/features.py`
This is the heart of the classifier. Build an extractor class that computes:
1. **Acoustic Metrics**:
   - `SNR (Signal-to-Noise Ratio)`: Estimate using RMS energy ratio of active frames vs. silent background frames (using simple energy thresholding).
   - `Silence Ratio`: Percentage of audio frames with energy below a quiet threshold.
   - `Length Ratio`: Character/word length of ground-truth transcript divided by duration of audio in seconds. Very large/small ratios signify critical mismatches or cut-off audios.
2. **ASR Confidence & Transcriptions**:
   - Use a lightweight Vietnamese ASR pipeline: Use Hugging Face Transformers with `openai/whisper-tiny` (highly optimized for fast Colab execution) or `vinai/wav2vec2-vi-large`.
   - Extract the `ASR Confidence Score` (average token-level log-probabilities or likelihood confidence) of the decoded sequence. Unclear pronunciation or regional dialects must yield very low confidence.
   - Extract the predicted text (`hyp_transcript`).
3. **Cross-Modal Similarity Metrics**:
   - Normalize both `hyp_transcript` and ground-truth `transcript` using the processor.
   - Calculate `WER (Word Error Rate)` and `CER (Character Error Rate)` using the `jiwer` package. This perfectly catches missing or extra words (which accounts for 10% of annotator rejects).
- Output all features as a dense 2D feature array / Pandas DataFrame.

## 6. `src/model.py`
- Factory function to initialize `LightGBM` and `XGBoost` classifiers.
- Ensure cost-sensitive learning parameters are set: `scale_pos_weight` (or `class_weight='balanced'`) to heavily penalize misclassification of the minority class (Unusable/0).

## 7. `src/evaluator.py`
- Perform Stratified 5-Fold Cross-Validation.
- Capture Out-of-Fold (OOF) prediction probabilities.
- **Threshold Optimization**: Write a function to sweep classification thresholds from `0.01` to `0.99` (step `0.01`) on the OOF probabilities to find the exact threshold that maximizes the **Macro F1-score**.
- Generate classification report (Precision, Recall, F1 for class 0 and 1) and plot a beautiful `confusion_matrix.png` using seaborn.

## 8. `src/register.py`
- Create a model packager that serializes: `StandardScaler` + `Trained Classifier` + `Optimal Decision Threshold` into a single `pipeline_artifacts.pkl` using `joblib`.
- Compile `metadata.json` containing experiment parameters, date, Git commit hash (if available), feature importances, and validation metrics.
- Package everything and automatically upload the run package directory under `<your_name>/models/run_{timestamp}/` directly to Azure Blob Storage using `azure-storage-blob`.

## 9. `main.py` (Command-Line Entrypoint)
- Parse arguments for fast validation or full-scale training.
- Chain all components: Download data -> Preprocess -> Feature Extraction -> Stratified Training & Validation -> Threshold Tuning -> Export Metrics and Plots -> Upload artifacts to Azure Blob Storage.
- Print clean and verbose progress bars (using `tqdm`) and comprehensive terminal logs.

## 10. `inference.py` (Single Sample Classifier)
- A standalone script that loads the serialized `pipeline_artifacts.pkl`.
- Takes raw inputs: `audio_path` and `transcript_string`.
- Internally processes audio, normalizes text, extracts features, runs inference, applies the optimized threshold, and prints the predicted label (`1: Usable` or `0: Unusable`) with prediction probability and quality details (SNR, WER, Confidence).

---

# QUALITY & STYLE GUIDELINES
- Write robust, industrial-grade Python code.
- Always implement `try-except` blocks for audio I/O and Azure network transactions.
- Use explicit type-hinting for all methods and classes.
- Provide comprehensive docstrings in English explaining the mathematical/logical rationale behind key features (especially SNR estimation and WER calculation).


5. Tóm tắt kế hoạch bứt phá điểm số tối đa

Model Performance (40%): Sử dụng kết hợp giữa WER/CER (từ Whisper) và các chỉ số vật lý âm thanh (SNR, Silence) giúp bao phủ 100% các nguyên nhân gây lỗi âm thanh từ thực tế. Việc tối ưu hóa ngưỡng phân loại (Decision Threshold) sẽ giúp kéo chỉ số Macro F1 lên vượt mức kỳ vọng $0.80$ cực kỳ dễ dàng.

Code Quality (25%) & Reproducibility (20%): Việc tổ chức thư mục mô-đun hóa kết hợp cơ chế kiểm soát lỗi chặt chẽ, không có bất cứ hard-code nào sẽ chinh phục hoàn toàn hội đồng thẩm định dự án của bạn.

Report & Analysis (15%): Khi viết báo cáo, bạn chỉ cần đưa phát hiện mang tính chiến lược của hai bạn (việc chuyển hướng từ trùng lặp ngữ nghĩa sang thẩm định vật lý & độ lệch từ) vào phần đầu của báo cáo. Đây sẽ là điểm nhấn "đắt giá" nhất chứng minh tư duy thiết kế hệ thống thực tiễn của một kỹ sư giàu kinh nghiệm.

Chúc bạn và đồng đội triển khai dự án cực kỳ thành công! Nếu bạn cần tôi làm rõ sâu thêm về phần tính toán đặc trưng nào (như SNR vật lý hoặc trích xuất log-prob từ Whisper), hãy cho tôi biết nhé.