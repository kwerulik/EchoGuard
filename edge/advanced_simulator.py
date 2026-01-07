import boto3
import numpy as np
import os
import time
import random
from datetime import datetime

BUCKET_NAME = 'echoguard-data'
SOURCE_FILE = 'data/raw/2nd_test/2004.02.12.18.32.39.npy'

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:4566',
    aws_access_key_id='test',
    aws_secret_access_key='test',  
    region_name='us-east-1'
)

def load_base_data():
    """Wczytuje 'zdrowy' plik jako wzorzec."""
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ Nie znaleziono pliku wzorcowego: {SOURCE_FILE}")
        return np.random.normal(0, 0.1, (128, 64)).astype(np.float32)
    return np.load(SOURCE_FILE)


def simulate_vibration():
    base_data = load_base_data()

    print("\n--- ROZPOCZYNAM DELIKATNĄ SYMULACJĘ ---")

    try:
        while True:
            is_anomaly = random.random() < 0.2
            current_data = base_data.copy()

            if is_anomaly:
                print("⚠️  AWARIA! Generowanie wibracji...")
                # Zmieniamy charakterystykę drastycznie
                current_data = current_data * 2.0
                # Szum anomalii
                noise = np.random.normal(0, 0.1, current_data.shape)
                current_data += noise
                file_prefix = "ANOMALY_"
            else:
                print("✅  Praca normalna.")
                # --- TUTAJ BYŁ PROBLEM ---
                # Zmniejszamy szum z 0.02 na 0.001 (było za głośno dla modelu)
                noise = np.random.normal(0, 0.001, current_data.shape)
                current_data += noise
                file_prefix = "NORMAL_"

            # Generowanie nazwy i zapisu
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") 
            filename = f"{file_prefix}{timestamp}.npy"
            temp_path = f"temp_{filename}"

            np.save(temp_path, current_data)

            try:
                s3.upload_file(temp_path, BUCKET_NAME, filename)
                print(f"📤 Wysłano: {filename}")
            except Exception as e:
                print(f"❌ Błąd wysyłania: {e}")

            os.remove(temp_path)
            time.sleep(3)  

    except KeyboardInterrupt:
        print("\n🛑 Zatrzymano.")

if __name__ == "__main__":
    simulate_vibration()
