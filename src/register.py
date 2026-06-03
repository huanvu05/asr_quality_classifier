import os
import joblib
import json
import datetime
from azure.storage.blob import BlobServiceClient
from src.config import config

class ModelRegister:
    """
    Packages artifacts and uploads to Azure Blob Storage.
    """
    def __init__(self):
        self.blob_service_client = BlobServiceClient(
            account_url=config.AZURE_STORAGE_URL, 
            credential=config.AZURE_SAS_TOKEN
        )
        self.container_name = "models"

    def save_pipeline(self, model, scaler, threshold: float, run_id: str):
        """
        Serializes the pipeline to a local file.
        """
        artifacts = {
            "model": model,
            "scaler": scaler,
            "threshold": threshold
        }
        save_path = os.path.join(config.MODELS_DIR, f"{run_id}_pipeline.pkl")
        joblib.dump(artifacts, save_path)
        return save_path

    def upload_artifacts(self, run_id: str, local_files: list):
        """
        Uploads artifacts to Azure Blob under a unique run directory.
        """
        try:
            for file_path in local_files:
                file_name = os.path.basename(file_path)
                blob_path = f"{run_id}/{file_name}"
                
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name, 
                    blob=blob_path
                )
                
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
            print(f"Successfully uploaded artifacts for {run_id}")
        except Exception as e:
            print(f"Upload failed: {e}")

def create_metadata(run_id: str, metrics: dict, params: dict):
    """
    Generates a metadata.json file.
    """
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": metrics,
        "params": params,
        "system": {
            "platform": "Darwin/Linux",
            "python_version": "3.x"
        }
    }
    meta_path = os.path.join(config.MODELS_DIR, f"{run_id}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    return meta_path
