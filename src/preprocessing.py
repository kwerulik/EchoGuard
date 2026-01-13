import numpy as np


def create_windows(data, window_width=64, stride=32):
    """
    Tworzy okna z danych spektrogramu.
    Obsługuje padding dla krótkich sygnałów.
    """
    # Upewnij się, że dane są 2D (n_mels, time_steps)
    if data.ndim > 2:
        data = np.squeeze(data)

    n_mels, time_steps = data.shape
    windows = []

    # Jeśli sygnał jest krótszy niż okno, dodaj padding (zera)
    if time_steps < window_width:
        pad_width = window_width - time_steps
        data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
        time_steps = window_width

    # Sliding window
    for start in range(0, time_steps - window_width + 1, stride):
        end = start + window_width
        window = data[:, start:end]
        windows.append(window)

    # (Batch_Size, Height, Width)
    windows = np.array(windows)

    # Dodaj kanał (Batch, H, W, 1) - wymagane przez CNN/ONNX
    if windows.ndim == 3:
        windows = np.expand_dims(windows, axis=-1)

    return windows.astype(np.float32)
