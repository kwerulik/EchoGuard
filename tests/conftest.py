import sys
import os
import pytest
import numpy as np

os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
os.environ['AWS_SECURITY_TOKEN'] = 'testing'
os.environ['AWS_SESSION_TOKEN'] = 'testing'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_dir)
sys.path.insert(0, main_dir)

@pytest.fixture
def sample_spectogram():
    '''Przykładowy spectogram (mock danych) do testów'''
    return np.random.rand(128, 100).astype(np.float32)