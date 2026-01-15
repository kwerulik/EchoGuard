import pytest
import pandas as pd
import numpy as np
import os 
from unittest.mock import patch, MagicMock
from src.data_loader import load_bearing_data, compute_melspec

#*--- Test 1 ---
@patch('src.data_loader.pd.read_csv')
def test_load_bearing_data_success(mock_read_csv):
    '''Sprawdza czy funkcja poprawnie wczytuje pliki i nadaje nazwy kolumn'''

    mock_df = pd.DataFrame(np.random.rand(5, 4))
    mock_read_csv.return_value = mock_df

    filename = 'test_file.csv'
    data_dir = '/tmp/data'

    result = load_bearing_data(filename, data_dir=data_dir)

    expectede_path = os.path.join(data_dir, filename)
    mock_read_csv.assert_called_once()
    args, kwargs = mock_read_csv.call_args
    assert args[0] == expectede_path

    expected_cols = ['Bearing_1', 'Bearing_2', 'Bearing_3', 'Bearing_4']
    assert list(result.columns) == expected_cols
    assert len(result) == 5


#*--- Test 2 ---
@patch('src.data_loader.pd.read_csv')
def test_load_bearing_data_file_not_found(mock_read_csv):
    '''Sprawdza zachowanie funkcji gdy nie ma pliku'''
    mock_read_csv.side_effect = FileNotFoundError('Brak Pliku')

    with pytest.raises(FileNotFoundError):
        load_bearing_data('brakpliku.csv')


#*---Test 3 ---
@patch('src.data_loader.librosa.power_to_db')
@patch('src.data_loader.librosa.feature.melspectrogram')
def test_compute_mel_calls(mock_melspec, mock_power_to_db):
    """Sprawdza czy librosa została wywołana z prawidłowymi parametrami (n_mels=128 itp.)"""
    df = pd.DataFrame({'Bearing_1': np.random.rand(1000)})

    mock_melspec_result = np.zeros((128, 10))
    mock_melspec.return_value = mock_melspec_result
    mock_power_to_db.return_value = mock_melspec_result

    compute_melspec(df)

    mock_melspec.assert_called_once()
    _, kwargs = mock_melspec.call_args

    assert kwargs['sr'] == 20000
    assert kwargs['n_mels'] == 128
    assert kwargs['fmax'] == 10000
    assert kwargs['hop_length'] == 128

    mock_power_to_db.assert_called_once()
