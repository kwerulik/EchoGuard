import pytest
import subprocess
import time
import boto3
import os
import requests
from playwright.sync_api import Page, expect

ENDPOINT_URL = 'http://localhost:4566'
AWS_CONFIG = {
    'aws_access_key_id': 'test',
    'aws_secret_access_key': 'test',
    'region_name': 'us-east-1',
    'endpoint_url': ENDPOINT_URL
}

@pytest.fixture(scope="module")
def run_streamlit():
    """Uruchamia dashboard/app.py w osobnym procesie na czas testów"""
    print("Startowanie Streamlit...")

    app_path = os.path.join("dashboard", "app.py")

    # Uruchomienie procesu
    process = subprocess.Popen(
        ["streamlit", "run", app_path, "--server.port=8501", "--server.headless=true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    # Czekanie aż serwer wstanie
    for _ in range(10):
        try:
            response = requests.get("http://localhost:8501/_stcore/health")
            if response.status_code == 200:
                print("Streamlit działa!")
                break
        except requests.ConnectionError:
            time.sleep(1)
    else:
        process.kill()
        pytest.fail("Streamlit nie wystartował w ciągu 10 sekund")

    yield
    # Sprzątanie po testach
    print("Zamykanie Streamlit...")
    process.kill()

@pytest.fixture
def db_setup():
    """Czyści tabelę i daje klienta do wstawiania danych testowych"""
    dynamodb = boto3.resource('dynamodb', **AWS_CONFIG)
    table = dynamodb.Table("EchoGuardResults")

    scan = table.scan()
    with table.batch_writer() as batch:
        for each in scan.get('Items', []):
            batch.delete_item(
                Key={'device_id': each['device_id'], 'timestamp': each['timestamp']})

    return table


#*--- Test 1 ---
def test_dashboard_empty_state(run_streamlit, db_setup, page: Page):
    """
    Scenariusz:
    1. Baza jest pusta.
    2. Wchodzimy na dashboard.
    3. Oczekujemy komunikatu ostrzegawczego (Warning).
    """
    # Otwórz stronę dashboardu
    page.goto("http://localhost:8501")

    expect(page.get_by_text("EchoGuard: Predictive Maintenance Dashboard")).to_be_visible(timeout=10000)

    expect(page.get_by_text("LocalStack")).to_be_visible()

    expect(page.get_by_text("Oczekiwanie na dane w DynamoDB...")).to_be_visible()


#*--- Test 2 ---
def test_dashboard_displays_metrics(run_streamlit, db_setup, page: Page):
    """
    Scenariusz:
    1. Wrzucamy do bazy rekord z Anomalią.
    2. Odświeżamy dashboard (lub czekamy na auto-refresh).
    3. Sprawdzamy czy widać "ANOMALY_DETECTED" i czerwony alarm.
    """
    db_setup.put_item(Item={
        'device_id': 'test_rig_1',
        'timestamp': '2024-01-01-12-00-00',
        'mse_value': '0.0055',
        'threshold': '0.002',
        'status': 'ANOMALY_DETECTED',
        'source_file': 'test_dashboard.npy'
    })

    page.goto("http://localhost:8501")
    expect(page.get_by_text("ANOMALY_DETECTED")).to_be_visible(timeout=15000)
    expect(page.get_by_text("0.005500")).to_be_visible()
    expect(page.get_by_text("Krzywa Życia Łożyska")).to_be_visible()
