import pytest
import json
import numpy as np
import os
from unittest.mock import MagicMock, patch
import cloud.lambda_handler as lh
from cloud.lambda_handler import lambda_handler
#* ---Test 1: Zdrowe łożysko ---

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

    print(f"DEBUG BODY: {response['body']}")
    body = json.loads(response['body'])

    assert response['statusCode'] == 200
    assert body['status'] == 'HEALTHY'
    assert body['mse'] < 0.002


    mock_create_windows.assert_called_once()
    mock_table.put_item.assert_called_once()