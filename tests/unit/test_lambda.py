import pytest
import json
import numpy as np
import os
from unittest.mock import MagicMock, patch
import cloud.lambda_handler as lh
from cloud.lambda_handler import lambda_handler

#* ---Test 1: HEALTY ---

@patch('cloud.lambda_handler.create_windows')
@patch('cloud.lambda_handler.s3')
@patch('cloud.lambda_handler.table')
@patch('cloud.lambda_handler.ort')
@patch('cloud.lambda_handler.os.path.exists')
def test_lambda_healthy_flow(mock_exists, mock_ort, mock_table, mock_s3, mock_create_windows):
    '''Sprawdza czy Lambda zwróci HEALTHY'''
    def check_file_exists(path):
        if 'bearing_model.onnx' in str(path):
                return True
        return False


    mock_exists.side_effect = check_file_exists
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session

    fake_input = np.zeros((128, 100), dtype=np.float32)
    processed_batch = np.zeros((5, 128, 64, 1), dtype=np.float32)

    mock_create_windows.return_value = processed_batch
    mock_session.run.return_value = [processed_batch]

    mock_session.get_inputs.return_value = [MagicMock(name='input_node')]
    mock_session.get_outputs.return_value = [MagicMock(name='output_node')]
    
    lh.session = None
    lh.threshold = None


    with patch('numpy.load', return_value=fake_input):
        event = {
            'Records': [{'s3': {'bucket': {'name': 'test-bucket'}, 'object': {'key': 'bearing_ok.npy'}}}]
        }

        response = lambda_handler(event, None)

    body = json.loads(response['body'])

    assert response['statusCode'] == 200
    assert body['status'] == 'HEALTHY'
    assert body['mse'] < 0.002


    mock_create_windows.assert_called_once()
    mock_table.put_item.assert_called_once()


#* --- Test 2: Anomaly ---

@patch('cloud.lambda_handler.create_windows')
@patch('cloud.lambda_handler.s3')
@patch('cloud.lambda_handler.table')
@patch('cloud.lambda_handler.ort')
@patch('cloud.lambda_handler.os.path.exists')
def test_lambda_anomaly_flow(mock_exists, mock_ort, mock_table, mock_s3, mock_create_windows):
    '''Sprawdza czy Lambda poprawnie wykrywa anomalię (duże MSE)'''
    def check_file_exists(path):
        if 'bearing_model.onnx' in str(path):
            return True
        return False

    mock_exists.side_effect = check_file_exists
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session

    fake_input = np.zeros((128, 100), dtype=np.float32)
    processed_batch = np.zeros((5, 128, 64, 1), dtype=np.float32)
    reconstruction = np.ones((5, 128, 64, 1), dtype=np.float32)

    mock_create_windows.return_value = processed_batch
    mock_session.run.return_value = [reconstruction]

    mock_session.get_inputs.return_value = [MagicMock(name='input_node')]
    mock_session.get_outputs.return_value = [MagicMock(name='output_node')]

    lh.session = None
    lh.threshold = None


    with patch('numpy.load', return_value=fake_input):
        event = {
            'Records': [{'s3': {'bucket': {'name': 'test-bucket'}, 'object': {'key': 'bearing_fault.npy'}}}]
        }
        response = lambda_handler(event, None)

    body = json.loads(response['body'])

    # 5. Asercje
    assert response['statusCode'] == 200
    assert body['status'] == 'ANOMALY_DETECTED' 
    assert body['mse'] > 0.002
    mock_table.put_item.assert_called_once() 


# * --- Test 3: Inny Threshold ---

@patch('cloud.lambda_handler.create_windows')
@patch('cloud.lambda_handler.s3')
@patch('cloud.lambda_handler.table')
@patch('cloud.lambda_handler.ort')
@patch('cloud.lambda_handler.os.path.exists')
@patch('builtins.open', new_callable=MagicMock)
@patch('json.load')
def test_lambda_custom_config(mock_json_load, mock_open, mock_exists, mock_ort, mock_table, mock_s3, mock_create_windows):
    '''Sprawdza czy Lambda ładuje próg z pliku config'''
    mock_exists.return_value = True
    mock_json_load.return_value = {'threshold': 0.5}

    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session

    processed_batch = np.zeros((1, 128, 64, 1), dtype=np.float32)
    reconstruction = np.full((1, 128, 64, 1), 0.1, dtype=np.float32)  

    mock_create_windows.return_value = processed_batch
    mock_session.run.return_value = [reconstruction]

    mock_session.get_inputs.return_value = [MagicMock(name='input')]
    mock_session.get_outputs.return_value = [MagicMock(name='output')]

    lh.session = None
    lh.threshold = None

    with patch('numpy.load', return_value=np.zeros((128, 100))):
        event = {'Records': [
            {'s3': {'bucket': {'name': 'b'}, 'object': {'key': 'k'}}}]}
        response = lambda_handler(event, None)

    body = json.loads(response['body'])

    assert body['status'] == 'HEALTHY'
    assert lh.threshold == 0.5


# * --- Test 4: Błąd Inicjalizacji (Brak Modelu) ---
@patch('cloud.lambda_handler.os.path.exists')
def test_lambda_missing_model(mock_exists):
    '''Sprawdza zachowanie gdy brak pliku modelu .onnx'''
    mock_exists.return_value = False

    lh.session = None

    event = {'Records': [{'s3': {'bucket': {'name': 'b'}, 'object': {'key': 'k'}}}]}

    response = lambda_handler(event, None)

    assert response['statusCode'] == 500
    assert "Init Error" in response['body']
    assert "Brak pliku modelu" in response['body']


# * --- Test 5: Błąd S3 (Brak pliku danych) ---
@patch('cloud.lambda_handler.ort')
@patch('cloud.lambda_handler.os.path.exists')
@patch('cloud.lambda_handler.s3')
def test_lambda_s3_error(mock_s3, mock_exists, mock_ort):
    '''Sprawdza zachowanie gdy S3 rzuci błąd przy pobieraniu'''
    mock_exists.return_value = True
    mock_ort.InferenceSession.return_value = MagicMock()

    mock_s3.download_file.side_effect = Exception("Access Denied")

    lh.session = None

    event = {'Records': [
        {'s3': {'bucket': {'name': 'b'}, 'object': {'key': 'k'}}}]}

    response = lambda_handler(event, None)
    
    assert response['statusCode'] == 500
    assert "Access Denied" in response['body']


# * --- Test 6: Błąd DynamoDB (Zapis nieudany) ---
@patch('cloud.lambda_handler.create_windows')
@patch('cloud.lambda_handler.s3')
@patch('cloud.lambda_handler.table')
@patch('cloud.lambda_handler.ort')
@patch('cloud.lambda_handler.os.path.exists')
def test_lambda_db_failure(mock_exists, mock_ort, mock_table, mock_s3, mock_create_windows):
    '''Sprawdza czy Lambda zwraca 200 nawet jeśli zapis do bazy się nie uda'''
    def check_file_exists(path):
        if 'bearing_model.onnx' in str(path):
            return True
        return False

    mock_exists.side_effect = check_file_exists
    mock_session = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session

    fake_input = np.zeros((128, 100), dtype=np.float32)
    processed_batch = np.zeros((5, 128, 64, 1), dtype=np.float32)
    mock_create_windows.return_value = processed_batch
    mock_session.run.return_value = [processed_batch]
    mock_session.get_inputs.return_value = [MagicMock()]
    mock_session.get_outputs.return_value = [MagicMock()]

    mock_table.put_item.side_effect = Exception("DynamoDB Timeout")

    lh.session = None
    lh.threshold = None

    with patch('numpy.load', return_value=fake_input):
        event = {'Records': [
            {'s3': {'bucket': {'name': 'b'}, 'object': {'key': 'k'}}}]}
        response = lambda_handler(event, None)

    assert response['statusCode'] == 200
    body = json.loads(response['body'])
    assert body['status'] == 'HEALTHY'

    mock_table.put_item.assert_called_once()
