import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm
from src.config import config

class AzureBlobDownloader:
    """
    Handles downloading audio and transcript files from Azure Blob Storage.
    """
    def __init__(self):
        if not config.AZURE_SAS_TOKEN:
            raise ValueError("AZURE_SAS_TOKEN is missing in environment.")
        
        self.blob_service_client = BlobServiceClient(
            account_url=config.AZURE_STORAGE_URL, 
            credential=config.AZURE_SAS_TOKEN
        )
        self.container_name = "data" # Adjust container name as per actual Azure setup

    def download_file(self, blob_path: str, local_path: str):
        """Downloads a single blob to a local path."""
        try:
            if os.path.exists(local_path):
                return
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_path
            )
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
        except Exception as e:
            print(f"Error downloading {blob_path}: {e}")

    def download_dataset(self, df: pd.DataFrame, max_workers: int = 10):
        """Downloads all files in the dataframe using multi-threading."""
        tasks = []
        for _, row in df.iterrows():
            # formulate paths based on logic: folder/file_name
            audio_blob = f"{row['folder']}/{row['file_name']}"
            audio_local = os.path.join(config.AUDIO_DIR, row['file_name'])
            tasks.append((audio_blob, audio_local))
            
            # If there's a transcript path logic, add here
        
        print(f"Starting download of {len(tasks)} files...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(lambda x: self.download_file(*x), tasks), total=len(tasks)))

def load_labels(csv_path: str) -> pd.DataFrame:
    """Reads and cleans labels file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labels file not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Map columns if they don't match the expected names (optional, for robustness)
    # The user's CSV has: file_path, transcript, label
    required_cols = ['file_path', 'transcript', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")
            
    # Clean up transcripts and labels
    df['transcript'] = df['transcript'].fillna("")
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    
    return df
