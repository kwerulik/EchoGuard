import librosa.display
from scipy.fft import fft, fftfreq
import os
import pandas as pd
import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .preprocessing import create_windows
DEFAULT_DATA_DIR = '../data/raw/2nd_test'


def load_bearing_data(filemane, data_dir=DEFAULT_DATA_DIR):
    """
    Downloading a single file with vibration
    """
    file_path = os.path.join(data_dir, filemane)
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns = ['Bearing_1', 'Bearing_2', 'Bearing_3', 'Bearing_4']
    return df


def compute_melspec(df, colum_name='Bearing_1', sr=20000):
    signal = df['Bearing_1'].values
    melspec = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=128, fmax=10000, hop_length=128)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    return melspec_db


# def creadte_dataset_windows(melspec, window_size=64, stride=32):
#     n_mels, time_steps = melspec.shape
#     windows = []
#     for i in range(0, time_steps-window_size, stride):
#         w = melspec[:, i:i+window_size]
#         windows.append(w)

#     X = np.array(windows)
#     X = X[..., np.newaxis]
#     return X
