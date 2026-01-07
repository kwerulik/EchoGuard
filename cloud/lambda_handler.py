import os
import json
import logging
import sys
import boto3
import numpy as np
import onnxruntime as ort
from datetime import datetime

# Konfiguracja loggera
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- INICJALIZACJA ZASOBÓW AWS ---
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
TABLE_NAME = "EchoGuardResults"
table = dynamodb.Table(TABLE_NAME)

session = None
threshold = None


def create_windows(data, window_width=64, stride=32):
    """
    Tnie spektrogram (128, Time) na okna (Batch, 128, 64, 1).
    Używamy stride (przesunięcia), żeby okna na siebie nachodziły (lepsze pokrycie).
    """
    n_mels, time_steps = data.shape
    windows = []

    # Jeśli plik jest krótszy niż okno, padujemy zerami
    if time_steps < window_width:
        pad_width = window_width - time_steps
        data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
        time_steps = window_width

    # Sliding window
    for start in range(0, time_steps - window_width + 1, stride):
        end = start + window_width
        window = data[:, start:end]
        windows.append(window)

    # Konwersja na tensor: (Batch_Size, Height, Width)
    windows = np.array(windows)

    # Dodanie kanału (Batch, H, W, 1) - format dla CNN/Autoenkodera
    windows = np.expand_dims(windows, axis=-1)

    return windows.astype(np.float32)


def lambda_handler(event, context):
    global session, threshold

    logger.info("--- START LAMBDA (Deep Scan Mode) ---")

    # --- 1. ŁADOWANIE MODELU I KONFIGURACJI ---
    try:
        MODEL_PATH = 'bearing_model.onnx'
        CONFIG_PATH = 'model_config.json'

        if session is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Brak pliku modelu: {MODEL_PATH}")
            # Ładowanie sesji ONNX raz (cache'owanie w pamięci Lambdy)
            session = ort.InferenceSession(MODEL_PATH)
            logger.info("✅ Model ONNX załadowany.")

        if threshold is None:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    threshold = config.get('threshold', 0.002)
            else:
                threshold = 0.002
            logger.info(f"⚙️ Próg (Threshold): {threshold}")

    except Exception as e:
        logger.error(f"❌ Błąd inicjalizacji: {e}")
        return {'statusCode': 500, 'body': f"Init Error: {e}"}

    # --- 2. PRZETWARZANIE DANYCH ---
    try:
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        download_path = f"/tmp/{os.path.basename(key)}"

        logger.info(f"📥 Pobieranie: {key}")
        s3.download_file(bucket, key, download_path)

        # Wczytanie pełnego spektrogramu (np. 128 x 500)
        full_spectrogram = np.load(download_path).astype(np.float32)

        # Zabezpieczenie: Jeśli wejście ma zły wymiar (np. zapomniane (128, T))
        if full_spectrogram.ndim > 2:
            # Spłaszcz do 2D (Freq, Time)
            full_spectrogram = np.squeeze(full_spectrogram)

        # KLUCZOWA ZMIANA: Tworzenie Batcha okien zamiast ucinania
        # Tniemy sygnał na wiele kawałków po 64 kroki czasowe
        batch_data = create_windows(
            full_spectrogram, window_width=64, stride=32)

        logger.info(
            f"🧩 Utworzono batch o wymiarach: {batch_data.shape} (Liczba okien: {batch_data.shape[0]})")

        # --- 3. INFERENCJA (BATCH PROCESSING) ---
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # ONNX Runtime obsłuży cały batch na raz (dużo wydajniej!)
        reconstructions = session.run(
            [output_name], {input_name: batch_data})[0]

        # Obliczamy MSE dla każdego okna osobno
        # axis=(1,2,3) uśrednia po wymiarach obrazka, zostawiając wymiar batcha
        mse_per_window = np.mean(
            np.power(batch_data - reconstructions, 2), axis=(1, 2, 3))

        # Ostateczny wynik to średnia MSE ze wszystkich okien
        # (Można też użyć np.max(mse_per_window) jeśli chcemy być bardzo czuli na piki)
        final_mse = float(np.mean(mse_per_window))

        logger.info(
            f"📊 ŚREDNI MSE: {final_mse:.6f} (Max w oknie: {np.max(mse_per_window):.6f})")

        status = "HEALTHY" if final_mse <= threshold else "ANOMALY_DETECTED"
        if status == "ANOMALY_DETECTED":
            logger.warning(
                f"🚨 ANOMALIA! MSE ({final_mse:.5f}) > Próg ({threshold})")

        # --- 4. ZAPIS DO DYNAMODB ---
        try:
            filename = os.path.basename(key)
            timestamp_str = filename.replace('.npy', '').replace('.', '-')

            item = {
                'device_id': 'test_rig_1',
                'timestamp': timestamp_str,
                # DynamoDB wymaga string dla float
                'mse_value': str(final_mse),
                'status': status,
                'threshold': str(threshold),
                'source_file': key,
                'windows_processed': str(batch_data.shape[0]),
                'processed_at': datetime.now().isoformat()
            }
            table.put_item(Item=item)

        except Exception as db_error:
            logger.error(f"⚠️ Błąd DynamoDB: {db_error}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'file': key,
                'status': status,
                'mse': final_mse,
                'windows_count': batch_data.shape[0]
            })
        }

    except Exception as e:
        logger.error(f"❌ Critical Error: {e}", exc_info=True)
        return {'statusCode': 500, 'body': str(e)}
