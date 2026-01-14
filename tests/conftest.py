import sys
import os
import pytest
import numpy as np
import boto3

os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
os.environ['AWS_SECURITY_TOKEN'] = 'testing'
os.environ['AWS_SESSION_TOKEN'] = 'testing'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_dir)
sys.path.insert(0, main_dir)
sys.path.insert(0, os.path.join(main_dir, 'cloud'))


@pytest.fixture
def sample_spectogram():
    '''Przykładowy spectogram (mock danych) do testów'''
    return np.random.rand(128, 100).astype(np.float32)

@pytest.fixture(scope='session')
def s3_client():
    '''S3 łączy się z LocalStack'''
    return boto3.client('s3', endpoint_url='http://localhost:4566')

@pytest.fixture(scope='session')
def dynamodb_resource():
    '''DynamoDB łączy się z LocalStackiem'''
    return boto3.resource("dynamodb", endpoint_url="http://localhost:4566")

@pytest.fixture(scope='session')
def dynamodb_table(dynamodb_resource):
    '''Zwraca gotowy obiekt z tabeli'''
    return dynamodb_resource.Table('EchoGuardResults')
