#!/bin/bash
echo "--- INICJALIZACJA ECHOGUARD ---"

awslocal s3 mb s3://echoguard-data

echo "--- GOTOWE! Bucket 'echoguard-data' utworzony. ---"

echo "Tworzenie tabeli DynamoDB..."
awslocal dynamodb create-table \
    --table-name EchoGuardResults \
    --attribute-definitions \
        AttributeName=device_id,AttributeType=S \
        AttributeName=timestamp,AttributeType=S \
    --key-schema \
        AttributeName=device_id,KeyType=HASH \
        AttributeName=timestamp,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST