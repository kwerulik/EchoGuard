import boto3
import pytest
import numpy as np
import time
import os
import tempfile

ENDPOINT_URL = 'http://localhost:4566'
AWS_CONFIG = {
    'aws_access_key_id': 'test',
    'aws_secret_access_key': 'test',
    'region_name': 'us-east-1',
    'endpoint_url': ENDPOINT_URL
}

@pytest.fixture(scope="module")
def aws_clients():
    """Tworzy klientów połączonych z działającym LocalStackiem"""
    try:
        s3 = boto3.client('s3', **AWS_CONFIG)
        ddb = boto3.resource('dynamodb', **AWS_CONFIG)
        s3.list_buckets()
        return s3, ddb
    except Exception as e:
        pytest.fail(
            f"Nie można połączyć się z LocalStackiem {ENDPOINT_URL}. Błąd: {e}")
        
def create_temp_npy(shape=(128, 100), content=None):
    '''Funkcja pomocnicza do tworzenia plików'''
    fd, path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    data = content if content is not None else np.zeros(
        shape, dtype=np.float32)
    np.save(path, data)
    return path

#*--- Test 1 ---


def test_e2e_processing_zeros(aws_clients):
    """Sprawdza pełny cykl przetwarzania dla pliku z samymi zerami (oczekiwana anomalia)."""
    s3, ddb = aws_clients
    bucket_name = 'echoguard-data'
    file_key = f'e2e_01_{int(time.time())}.npy'
    local_path = create_temp_npy()

    try:
        s3.upload_file(local_path, bucket_name, file_key)
        time.sleep(10)

        table = ddb.Table('EchoGuardResults')
        items = table.scan().get('Items', [])

        found_item = None
        for item in items:
            if item.get('source_file') == file_key:
                found_item = item
                break

        assert found_item is not None, f"Nie znaleziono pliku {file_key} w bazie"
        assert found_item['status'] == 'ANOMALY_DETECTED'
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
#*--- Test 2 ---
def test_e2e_processing_noise(aws_clients):
    """Sprawdza przepływ dla losowego szumu"""
    s3, ddb = aws_clients
    bucket_name = 'echoguard-data'

    test_data = np.random.rand(128, 100).astype(np.float32)
    file_key = f'e2e_noise_{int(time.time())}.npy'

    fd, local_path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(local_path, test_data)

    try:
        s3.upload_file(local_path, bucket_name, file_key)
        time.sleep(15)

        table = ddb.Table('EchoGuardResults')
        response = table.scan()
        items = response.get('Items', [])
        found_item = next((i for i in items if i.get('source_file') == file_key), None)

        assert found_item is not None
        assert found_item['status'] == 'ANOMALY_DETECTED'

    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

#*--- Test 3 ---
def test_e2e_ignore_txt_files(aws_clients):
    """Lambda ma filtr na .npy, więc plik .txt nie powinien trafić do bazy"""
    s3, ddb = aws_clients
    bucket_name = 'echoguard-data'

    fd, local_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    file_key = f'readme_{int(time.time())}.txt'

    with open(local_path, 'w') as f:
        f.write("To nie jest plik npy")

    try:

        print(f"\n📤 [E2E] Wysyłanie pliku .txt: {file_key}")
        s3.upload_file(local_path, bucket_name, file_key)

        print("[E2E] Czekanie 10s (Lambda NIE powinna zadziałać)...")
        time.sleep(10)

        table = ddb.Table('EchoGuardResults')
        items = table.scan()['Items']

        # Szukamy czy nie pojawił się wpis dla tego pliku
        found_item = next(
            (i for i in items if i.get('source_file') == file_key), None)

        # Oczekujemy, że wpisu NIE MA (Lambda zignorowała plik)
        if found_item:
            pytest.fail(
                f"Błąd! Lambda przetworzyła plik .txt, a nie powinna! Status: {found_item}")

        print("[E2E] Sukces: Plik .txt został zignorowany przez trigger.")

    finally:
        # Sprzątanie
        if os.path.exists(local_path):
            os.remove(local_path)
