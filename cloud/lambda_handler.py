import os
import json
import logging
import sys
import boto3
from datetime import datetime

# Konfiguracja loggera
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- INICJALIZACJA ZASOBÓW AWS (Global Scope) ---
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
TABLE_NAME = "EchoGuardResults"
table = dynamodb.Table(TABLE_NAME)

session = None
threshold = None


def lambda_handler(event, context):
    global session, threshold

    logger.info("--- START LAMBDA ---")
    logger.info(f"Python version: {sys.version}")

    try:
        import numpy as np
        import onnxruntime as ort
        logger.info("✅ Biblioteki załadowane (numpy, onnxruntime).")
    except ImportError as e:
        logger.error(f"❌ BŁĄD IMPORTU: {e}")
        return {'statusCode': 500, 'body': f"Library Error: {e}"}

    # --- 1. ŁADOWANIE MODELU I KONFIGURACJI ---
    try:
        MODEL_PATH = 'bearing_model.onnx'
        CONFIG_PATH = 'model_config.json'

        if session is None:
            logger.info(f"Ładowanie modelu z {MODEL_PATH}...")
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Brak pliku modelu: {MODEL_PATH}")
            session = ort.InferenceSession(MODEL_PATH)

        if threshold is None:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    threshold = config.get('threshold', 0.002)
            else:
                threshold = 0.002
            logger.info(f"Ustawiono próg (threshold): {threshold}")

    except Exception as e:
        logger.error(f"❌ Błąd ładowania modelu: {e}")
        return {'statusCode': 500, 'body': f"Model Load Error: {e}"}

    # --- 2. PRZETWARZANIE DANYCH ---
    try:
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        download_path = f"/tmp/{os.path.basename(key)}"

        logger.info(f"Pobieranie pliku: {bucket}/{key}")
        s3.download_file(bucket, key, download_path)

        data = np.load(download_path).astype(np.float32)

        TARGET_HEIGHT = 128
        TARGET_WIDTH = 64

        if data.shape[1] > TARGET_WIDTH:
            data = data[:, :TARGET_WIDTH]

        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=-1)

        # --- 3. INFERENCJA (URUCHOMIENIE MODELU) ---
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        reconstructions = session.run([output_name], {input_name: data})[0]

        # Obliczenie błędu MSE
        mse = np.mean(np.power(data - reconstructions, 2))
        mse_float = float(mse) 

        logger.info(f"📊 WYNIK MSE: {mse_float:.6f} (Próg: {threshold})")

        status = "HEALTHY" if mse_float <= threshold else "ANOMALY_DETECTED"
        if status == "ANOMALY_DETECTED":
            logger.warning("🚨 !!! WYKRYTO ANOMALIĘ !!! 🚨")

        # --- 4. ZAPIS DO DYNAMODB (NOWOŚĆ) ---
        try:
            filename = os.path.basename(key)
            timestamp_str = filename.replace('.npy', '').replace('.', '-')

            item = {
                'device_id': 'test_rig_1',
                'timestamp': timestamp_str,
                'mse_value': str(mse_float),
                'status': status,
                'threshold': str(threshold),
                'source_file': key,
                'processed_at': datetime.now().isoformat()
            }

            table.put_item(Item=item)
            logger.info(f"✅ Wynik zapisany w DynamoDB: {TABLE_NAME}")

        except Exception as db_error:
            logger.error(f"⚠️ Nie udało się zapisać do DynamoDB: {db_error}")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'file': key,
                'status': status,
                'mse': mse_float,
                'threshold': threshold
            })
        }

    except Exception as e:
        logger.error(f"❌ Błąd przetwarzania: {e}", exc_info=True)
        return {'statusCode': 500, 'body': str(e)}
