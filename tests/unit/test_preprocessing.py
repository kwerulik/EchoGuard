import numpy as np
import pytest
from src.preprocessing import create_windows

#*--- Test 1 ---
def test_create_windows_shape():
    """Sprawdza, czy funkcja tworzy tensor o poprawnych wymiarach (Batch, H, W, C)"""
    input_signal = np.zeros((128, 200))
    result = create_windows(input_signal)

    assert result.ndim == 4, "Wynik musi być 4-wymiarowy"
    # ((200 - 64) // 32) + 1 = 5 okien
    assert result.shape[0] == 5, f"Oczekiwano 5 okien, otrzymano {result.shape[0]}"
    assert result.shape[1] == 128, "Wysokość (H) musi wynosić 128"
    assert result.shape[2] == 64, "Szerokość (W) musi wynosić 64"
    assert result.shape[3] == 1, "Liczba kanałów (C) musi wynosić 1"

#*--- Test 2 ---
def test_create_windows_padding():
    """Sprawdza, czy krótki sygnał jest poprawnie dopełniany zerami"""
    short_signal = np.ones((128, 10))
    result = create_windows(short_signal, window_width=64)

    assert result.shape[0] == 1, "Powinno powstać dokładnie jedno okno"
    assert np.all(result[0, :, :10, 0] == 1), "Początek okna powinien zawierać sygnał (jedynki)"
    assert np.all(result[0, :, 10:, 0] == 0), "Reszta okna powinna być wypełniona zerami"

#*--- Test 3 ---
def test_data_type_is_float32():
    """Sprawdza, czy typ danych wyjściowych to float32 (wymagany przez ONNX Runtime)."""
    input_signal = np.random.rand(128, 50).astype(np.float64)
    result = create_windows(input_signal)

    assert result.dtype == np.float32, "Typ danych musi być przekonwertowany na float32"

#*--- Test 4 ---
def test_empty_input():
    """Sprawdza zachowanie funkcji dla pustego sygnału wejściowego."""
    empty_signal = np.zeros((128, 0))
    result = create_windows(empty_signal, window_width=64)
    assert result.shape == (1, 128, 64, 1)
    assert np.sum(result) == 0, "Wynik dla pustego wejścia powinien składać się z samych zer"

#*--- Test 5 ---
def test_stride_logic():
    """Sprawdza poprawność przesuwania okna (stride)"""
    input_signal = np.zeros((128, 128))
    input_signal[:, :64] = 1
    input_signal[:, 64:] = 2
    result = create_windows(input_signal, window_width=64, stride=64)

    assert result.shape[0] == 2, "Powinny powstać dwa niezależne okna"
    assert np.mean(result[0]) == 1.0, "Pierwsze okno powinno zawierać same 1"
    assert np.mean(result[1]) == 2.0, "Drugie okno powinno zawierać same 2"

#*--- Test 6 ---
def test_exact_window_fit():
    '''Sprawdza zachowanie funkcji, gdy sygnał idealnie mieści się w okno(128, 64)'''
    input_signal = np.ones((128, 64))
    result = create_windows(input_signal)

    assert result.shape == (1, 128, 64, 1)
    assert np.all(result[0, :, :, 0]) == 1

#*--- Test 7 ---
def test_3d_input_handling():
    '''Sprawdza zachowanie funkcji gdy input ma wymiar 3d'''
    input_signal = np.zeros((1, 128, 64))
    result = create_windows(input_signal)

    assert result.ndim == 4
    assert result.shape == (1, 128, 64, 1)

#*--- Test 8 ---
def test_1d_input_handling():
    '''Sprawdza zachowanie funkcji gdy input ma wymiar 1d'''
    input_signal = np.zeros((128))
    with pytest.raises(ValueError):
        create_windows(input_signal)

# *--- Test 9 ---
def test_stride_larger_than_window():
    '''Sprawdza przypadek, gdy przesunięcie jest większe niż szerokość okna (dziury w analizie)'''
    input_signal = np.zeros((128, 200))

    result = create_windows(input_signal, window_width=64, stride=100)
    assert result.shape[0] == 2

#*--- Test 10 ---
# *--- Test 10: Niestandardowe wymiary okna ---


def test_custom_window_size():
    '''Sprawdza działanie funkcji z niestandardową szerokością okna'''
    input_signal = np.zeros((128, 100))
    custom_width = 32

    result = create_windows(input_signal, window_width=custom_width)

    assert result.shape[2] == custom_width, "Szerokość wyjściowa musi zgadzać się z parametrem"
