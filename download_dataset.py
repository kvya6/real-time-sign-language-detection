# download_dataset.py
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Set your Kaggle API credentials (expects kaggle.json in ~/.kaggle)
api = KaggleApi()
api.authenticate()

# Download and unzip the dataset
dataset = 'grassknoted/asl-alphabet'
api.dataset_download_files(dataset, path='dataset', unzip=True)
print("✅ Download complete.")
