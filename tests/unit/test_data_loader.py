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