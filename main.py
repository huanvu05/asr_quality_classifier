import os
import argparse
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import config
from src.data_loader import AzureBlobDownloader, load_labels
from src.features import FeatureExtractor
from src.model import get_model
from src.evaluator import Evaluator
from src.register import ModelRegister, create_metadata

def main(args):
    # 1. Setup Run ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_{args.model}_f1"
    
    print(f"Starting pipeline: {run_id}")

    # 2. Download Data
    if not args.skip_download:
        downloader = AzureBlobDownloader()
        labels_df = load_labels(args.labels_path)
        downloader.download_dataset(labels_df)
    else:
        labels_df = load_labels(args.labels_path)

    # 3. Feature Extraction
    extractor = FeatureExtractor()
    features_df = extractor.extract_features(labels_df)
    features_df.to_csv(os.path.join(config.DATA_DIR, f"{run_id}_features.csv"), index=False)

    # 4. Prepare Data
    X = features_df.drop(columns=["file_name", "label"])
    y = features_df["label"]
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 5. Training & Cross-Validation
    evaluator = Evaluator(n_folds=config.N_FOLDS)
    oof_probs = evaluator.run_cv(X_scaled, y, lambda: get_model(args.model))

    # 6. Threshold Optimization
    optimal_threshold = evaluator.optimize_threshold(y, oof_probs)
    evaluator.evaluate_oof(y, oof_probs, optimal_threshold)

    # 7. Final Model Training (on full data)
    final_model = get_model(args.model)
    final_model.fit(X_scaled, y)

    # 8. Register & Upload
    register = ModelRegister()
    pipeline_path = register.save_pipeline(final_model, scaler, optimal_threshold, run_id)
    importance_path = register.plot_feature_importance(final_model, X.columns.tolist(), run_id)
    
    # Calculate detailed metrics for metadata
    metrics = {
        "oof_macro_f1": evaluator.optimize_threshold(y, oof_probs),
        "optimal_threshold": optimal_threshold
    }
    meta_path = create_metadata(run_id, metrics, config.LGBM_PARAMS)
    
    artifacts_to_upload = [pipeline_path, meta_path, "confusion_matrix.png"]
    if importance_path:
        artifacts_to_upload.append(importance_path)
        
    if args.upload:
        register.upload_artifacts(run_id, artifacts_to_upload)

    print(f"Pipeline finished successfully. Run ID: {run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR Quality Classifier Training Pipeline")
    parser.add_argument("--model", type=str, default="lightgbm", help="Model type: lightgbm or xgboost")
    parser.add_argument("--labels-path", type=str, default="data/transcripts/training.csv", help="Path to labels CSV")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading files from Azure")
    parser.add_argument("--upload", action="store_true", help="Upload artifacts to Azure Blob")
    
    args = parser.parse_args()
    main(args)
