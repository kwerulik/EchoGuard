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
            f"Nie można połączyć się z LocalStackiem pod {ENDPOINT_URL}. Czy kontenery działają? Błąd: {e}")
        

#*--- Test 1 ---
def test_e2e_full_processing_chain(aws_clients):
    """
    Testuje pełny przepływ w działającym środowisku Docker/LocalStack:
    1. Upload pliku do S3
    2. Czekanie na przetworzenie przez Lambdę
    3. Weryfikacja wpisu w DynamoDB
    """
    s3, ddb = aws_clients
    bucket_name = 'echoguard-data'
    table_name = 'EchoGuardResults'

    # 1. Przygotowanie danych testowych
    test_data = np.zeros((128, 100), dtype=np.float32)
    file_key = f'e2e_test_{int(time.time())}.npy'

    # Tworzenie pliku tymczasowego
    fd, local_path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(local_path, test_data)

    try:
        # 2. Upload do LocalStack S3
        print(f"\n[E2E] Wysyłanie pliku {file_key} do S3...")
        s3.upload_file(local_path, bucket_name, file_key)

        # 3. Czekanie na Lambdę
        wait_time = 15
        print(f"[E2E] Czekanie {wait_time}s na przetworzenie...")
        time.sleep(wait_time)

        # 4. Weryfikacja w DynamoDB
        print("[E2E] Skanowanie tabeli DynamoDB...")
        table = ddb.Table(table_name)
        response = table.scan()
        items = response.get('Items', [])

        found_item = next(
            (item for item in items if item.get('source_file') == file_key), None)

        if found_item is None:
            print(f"Nie znaleziono wpisu dla {file_key}. Zawartość tabeli:")
            for i in items:
                print(f"   - {i.get('source_file')} -> {i.get('status')}")
            pytest.fail("Lambda nie zapisała wyniku w DynamoDB.")

        print(
            f"[E2E] Znaleziono wynik: {found_item['status']} (MSE: {found_item['mse_value']})")

        assert found_item['status'] == 'ANOMALY_DETECTED'
        assert float(found_item['mse_value']) > 0.002
        assert found_item['device_id'] == 'test_rig_1'

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

        print("⏳ [E2E] Czekanie 10s (Lambda NIE powinna zadziałać)...")
        time.sleep(10)

        table = ddb.Table('EchoGuardResults')
        items = table.scan()['Items']

        # Szukamy czy przypadkiem nie pojawił się wpis dla tego pliku
        found_item = next(
            (i for i in items if i.get('source_file') == file_key), None)

        # Oczekujemy, że wpisu NIE MA (Lambda zignorowała plik)
        if found_item:
            pytest.fail(
                f"❌ Błąd! Lambda przetworzyła plik .txt, a nie powinna! Status: {found_item}")

        print("✅ [E2E] Sukces: Plik .txt został zignorowany przez trigger.")

    finally:
        # Sprzątanie
        if os.path.exists(local_path):
            os.remove(local_path)
