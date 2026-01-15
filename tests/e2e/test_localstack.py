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
        dynamodb = boto3.resource('dynamodb', **AWS_CONFIG)
        s3.list_buckets()
        return s3, dynamodb
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
    s3, dynamodb = aws_clients
    bucket_name = 'echoguard-data'
    file_key = f'e2e_01_{int(time.time())}.npy'
    local_path = create_temp_npy()

    try:
        s3.upload_file(local_path, bucket_name, file_key)
        time.sleep(10)

        table = dynamodb.Table('EchoGuardResults')
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
    """Sprawdza czy system poprawnie przetwarza i zapisuje losowy szum."""
    s3, dynamodb = aws_clients
    file_key = f'e2e_02_{int(time.time())}.npy'
    local_path = create_temp_npy(content=np.random.rand(128, 100).astype(np.float32))

    try:
        s3.upload_file(local_path, 'echoguard-data', file_key)
        time.sleep(10)

        items = dynamodb.Table('EchoGuardResults').scan().get('Items', [])

        found_item = None
        for item in items:
            if item.get('source_file') == file_key:
                found_item = item
                break

        assert found_item is not None
        assert 'mse_value' in found_item
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


#*--- Test 3 ---
def test_e2e_ignore_txt(aws_clients):
    """Weryfikuje, czy filtr S3 ignoruje pliki inne niż .npy (np. .txt)."""
    s3, dynamodb = aws_clients
    file_key = f'e2e_03_{int(time.time())}.txt'

    fd, local_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)

    try:
        s3.upload_file(local_path, 'echoguard-data', file_key)
        time.sleep(5)

        items = dynamodb.Table('EchoGuardResults').scan().get('Items', [])

        found_item = None
        for item in items:
            if item.get('source_file') == file_key:
                found_item = item
                break

        assert found_item is None  # Nie powinno być wpisu
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


#*--- Test 4 ---
def test_e2e_corrupted_file(aws_clients):
    """Sprawdza uszkodzony plik nie powinien trafić do bazy."""
    s3, dynamodb = aws_clients
    file_key = f'e2e_04_corrupted_{int(time.time())}.npy'

    fd, local_path = tempfile.mkstemp(suffix=".npy")
    os.write(fd, b"To nie jest poprawny numpy array")
    os.close(fd)

    try:
        s3.upload_file(local_path, 'echoguard-data', file_key)
        time.sleep(10)

        items = dynamodb.Table('EchoGuardResults').scan().get('Items', [])

        found_item = None
        for item in items:
            if item.get('source_file') == file_key:
                found_item = item
                break

        assert found_item is None
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


#*--- Test 5 ---
def test_e2e_subfolder_file(aws_clients):
    """Sprawdza czy Trigger S3 działa poprawnie dla plików w podfolderach."""
    s3, dynamodb = aws_clients
    file_key = f'sensors/zone_a/e2e_05_{int(time.time())}.npy'
    local_path = create_temp_npy()

    try:
        s3.upload_file(local_path, 'echoguard-data', file_key)
        time.sleep(10)

        items = dynamodb.Table('EchoGuardResults').scan().get('Items', [])

        found_item = None
        for item in items:
            if item.get('source_file') == file_key:
                found_item = item
                break

        assert found_item is not None
        assert found_item['device_id'] == 'test_rig_1'
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


#*--- Test 6 ---
def test_e2e_3d_input(aws_clients):
    """Testuje czy dane 3D są poprawnie spłaszczane."""
    s3, dynamodb = aws_clients
    file_key = f'e2e_06_{int(time.time())}.npy'
    local_path = create_temp_npy(shape=(1, 128, 100))

    try:
        s3.upload_file(local_path, 'echoguard-data', file_key)
        time.sleep(10)

        items = dynamodb.Table('EchoGuardResults').scan().get('Items', [])

        found_item = None
        for item in items:
            if item.get('source_file') == file_key:
                found_item = item
                break

        assert found_item is not None
        assert int(found_item['windows_processed']) > 0
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


#*--- Test 7 ---
def test_e2e_burst_upload(aws_clients):
    """Weryfikuje czy system obsłuży nagły napływ kilku plików."""
    s3, dynamodb = aws_clients
    base_name = f'e2e_07_{int(time.time())}'
    files = [f"{base_name}_{i}.npy" for i in range(3)]
    local_path = create_temp_npy()

    try:
        for f in files:
            s3.upload_file(local_path, 'echoguard-data', f)

        time.sleep(15)

        items = dynamodb.Table('EchoGuardResults').scan().get('Items', [])

        found_count = 0
        for item in items:
            if item.get('source_file') in files:
                found_count += 1

        assert found_count == 3, f"Oczekiwano 3 wpisów, znaleziono {found_count}"
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


#*--- Test 8 ---
def test_e2e_overwrite_file(aws_clients):
    """Sprawdza czy ponowne wgranie pliku aktualizuje wpis zamiast tworzyć duplikat."""
    s3, dynamodb = aws_clients
    file_key = f'e2e_09_overwrite.npy'
    local_path = create_temp_npy()

    try:
        s3.upload_file(local_path, 'echoguard-data', file_key)
        time.sleep(8)

        s3.upload_file(local_path, 'echoguard-data', file_key)
        time.sleep(8)

        items = dynamodb.Table('EchoGuardResults').scan().get('Items', [])

        matching_items = []
        for item in items:
            if item.get('source_file') == file_key:
                matching_items.append(item)

        assert len(matching_items) == 1
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)