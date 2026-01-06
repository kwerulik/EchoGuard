import os
import subprocess
import shutil
import boto3
import sys
import time
import uuid

FUNCTION_NAME = 'EchoGuardAnalyzer'
BUCKET_NAME = 'echoguard-data'
ZIP_NAME = 'lambda_package.zip'
BUILD_DIR = 'dist'

lambda_client = boto3.client('lambda', endpoint_url='http://localhost:4566',
                             aws_access_key_id='test', aws_secret_access_key='test', region_name='us-east-1')
s3_client = boto3.client('s3', endpoint_url='http://localhost:4566',
                         aws_access_key_id='test', aws_secret_access_key='test', region_name='us-east-1')


def ensure_infrastructure():
    print("🏗️ KROK 0: Sprawdzanie infrastruktury (Bucket S3)...")
    try:
        s3_client.create_bucket(Bucket=BUCKET_NAME)
        print(f"   ✅ Utworzono bucket: {BUCKET_NAME}")
    except Exception as e:
        print(f"   ℹ️ Bucket {BUCKET_NAME} już istnieje.")


def build_package():
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)

    if os.path.exists(ZIP_NAME):
        os.remove(ZIP_NAME)

    print("🏭 KROK 1: Budowanie paczki zgodnej z Amazon Linux 2...")

    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{os.getcwd()}/{BUILD_DIR}:/install',
        'python:3.9-slim',
        'pip', 'install',
        'numpy==1.23.5',
        'onnxruntime==1.14.1',
        'protobuf==3.20.3',
        '--platform', 'manylinux2014_x86_64',
        '--only-binary=:all:',
        '--target', '/install',
        '--implementation', 'cp',
        '--python-version', '3.9',
        '--abi', 'cp39'
    ]

    print("   ⏳ Pobieranie bibliotek (Fix GLIBC 2.26)...")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print("❌ Błąd Dockera. Upewnij się, że Docker Desktop działa.")
        sys.exit(1)

    for root, dirs, files in os.walk(BUILD_DIR):
        for d in dirs:
            if d.endswith('.dist-info') or d == '__pycache__':
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

    print("   📂 Kopiowanie plików projektu...")
    shutil.copy('cloud/lambda_handler.py', f'{BUILD_DIR}/lambda_handler.py')
    shutil.copy('models/bearing_model.onnx', f'{BUILD_DIR}/bearing_model.onnx')
    shutil.copy('config/model_config.json', f'{BUILD_DIR}/model_config.json')

    print("   📦 Pakowanie ZIP...")
    shutil.make_archive('lambda_package', 'zip', BUILD_DIR)
    print(f"   ✅ Gotowe: {ZIP_NAME}")


def deploy():
    print(f"🚀 KROK 2: Wdrażanie {FUNCTION_NAME}...")
    with open(f'{ZIP_NAME}', 'rb') as f:
        zip_content = f.read()

    try:
        lambda_client.delete_function(FunctionName=FUNCTION_NAME)
        print("   (Usunięto starą wersję funkcji)")
    except:
        pass

    response = lambda_client.create_function(
        FunctionName=FUNCTION_NAME,
        Runtime='python3.9',
        Role='arn:aws:iam::000000000000:role/lambda-role',
        Handler='lambda_handler.lambda_handler',
        Code={'ZipFile': zip_content},
        Timeout=60,
        MemorySize=512,
        Environment={'Variables': {'LOG_LEVEL': 'INFO'}}
    )
    print("   ⏳ Czekam 2s na stabilizację...")
    time.sleep(2)
    return response['FunctionArn']


def configure_trigger(function_arn):
    print("🔗 KROK 3: Podpinanie S3 Trigger...")
    statement_id = f's3-trigger-{uuid.uuid4()}'

    try:
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId=statement_id,
            Action='lambda:InvokeFunction',
            Principal='s3.amazonaws.com',
            SourceArn=f'arn:aws:s3:::{BUCKET_NAME}'
        )
        time.sleep(1)
    except Exception:
        pass

    try:
        try:
            try:
                lambda_client.remove_permission(
                    FunctionName='EchoGuardAnalyzer',
                    StatementId='s3-trigger-permission'
                )
            except:
                pass
            
            lambda_client.add_permission(
                FunctionName='EchoGuardAnalyzer',
                StatementId='s3-trigger-permission',
                Action='lambda:InvokeFunction',
                Principal='s3.amazonaws.com',
                SourceArn=f"arn:aws:s3:::echoguard-data"
            )
            print("✅ Nadano uprawnienia dla S3 do wywoływania Lambdy.")
        except lambda_client.exceptions.ResourceConflictException:
            print("ℹ️ Uprawnienia już istnieją.")
        except Exception as e:
            print(f"⚠️ Ostrzeżenie przy nadawaniu uprawnień: {e}")

        s3_client.put_bucket_notification_configuration(
            Bucket=BUCKET_NAME,
            NotificationConfiguration={
                'LambdaFunctionConfigurations': [{
                    'LambdaFunctionArn': function_arn,
                    'Events': ['s3:ObjectCreated:*'],
                    'Filter': {'Key': {'FilterRules': [{'Name': 'suffix', 'Value': '.npy'}]}}
                }]
            }
        )
        print("   ✅ Trigger skonfigurowany.")
    except Exception as e:
        print(f"❌ BŁĄD triggera: {e}")


if __name__ == "__main__":
    ensure_infrastructure()
    build_package()
    arn = deploy()
    configure_trigger(arn)
    print("\n🎉 SUKCES!")
