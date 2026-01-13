import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import time
import numpy as np
import boto3

from src.data_loader import load_bearing_data, compute_melspec

# --- KONFIGURACJA ---
s3 = boto3.client('s3', endpoint_url='http://localhost:4566',
                  aws_access_key_id='test', aws_secret_access_key='test', region_name='us-east-1')
BUCKET_NAME = 'echoguard-data'
DATA_DIR = 'data/raw/2nd_test'


def run_simulation(interval=0.5):
    """
    Simulates machine behavior throughout its entire lifecycle.
    interval: Time in seconds between sending consecutive files.
    """
    files = sorted(os.listdir(DATA_DIR))
    files = [f for f in files if not f.endswith(
        '.pdf') and not f.endswith('.doc')]

    print(f"🚀 Rozpoczynam symulację. Do przetworzenia: {len(files)} plików.")
    print(f"⏱️ Interwał wysyłania: {interval}s")

    for i, filename in enumerate(files):
        file_path = os.path.join(DATA_DIR, filename)

        try:
            # 1. Edge Processing
            df = load_bearing_data(filename, DATA_DIR)
            melspec = compute_melspec(df)

            # Normalization
            NORM_MIN, NORM_MAX = -80.0, 0.0
            norm_mel = (melspec - NORM_MIN) / (NORM_MAX - NORM_MIN)
            norm_mel = np.clip(norm_mel, 0, 1)

            # 2. Save to .npy and upload
            npy_filename = f"{filename}.npy"
            np.save(npy_filename, norm_mel)

            # Upload to S3 (triggers Lambdę)
            s3.upload_file(npy_filename, BUCKET_NAME, npy_filename)

            os.remove(npy_filename)

            print(f"[{i+1}/{len(files)}] 📡 Wysłano: {filename} -> S3")

        except Exception as e:
            print(f"❌ Błąd przy pliku {filename}: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    try:
        s3.create_bucket(Bucket=BUCKET_NAME)
    except:
        pass 

    run_simulation(interval=3) 
