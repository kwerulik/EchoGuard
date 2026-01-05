import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

import time
import numpy as np
import boto3
from data_loader import load_bearing_data, compute_melspec
from botocore.exceptions import NoCredentialsError

sys.path.append(os.path.abspath(os.path.join('..', 'src')))


s3 = boto3.client('s3',
                  endpoint_url='http://localhost:4566',
                  aws_access_key_id='test',
                  aws_secret_access_key='test',
                  region_name='us-east-1')

BUCKET_NAME = 'echoguard-data'


def process_and_upload(file_path):
    print(f"\n[EDGE] 📡 Przetwarzanie pliku: {file_path}")

    try:
        df = load_bearing_data(file_path, '')

        melspec = compute_melspec(df)

        NORM_MIN = -80.0
        NORM_MAX = 0.0
        norm_mel = (melspec - NORM_MIN) / (NORM_MAX - NORM_MIN)

        norm_mel = np.clip(norm_mel, 0, 1)

        filename = os.path.basename(file_path)
        npy_filename = f"{filename}.npy"
        np.save(npy_filename, norm_mel)

        print(f"[EDGE] ☁️ Wysyłanie do S3: {npy_filename}...")
        s3.upload_file(npy_filename, BUCKET_NAME, npy_filename)
        print("[EDGE] ✅ Sukces! Dane w chmurze.")

        os.remove(npy_filename)

    except Exception as e:
        print(f"[EDGE] ❌ Błąd: {e}")


if __name__ == "__main__":
    TEST_FILE = 'data/raw/2nd_test/2004.02.12.18.32.39'

    if os.path.exists(TEST_FILE):
        process_and_upload(TEST_FILE)
    else:
        print(f"Nie znaleziono pliku testowego: {TEST_FILE}")
        print("Sprawdź ścieżkę w zmiennej TEST_FILE w skrypcie.")
