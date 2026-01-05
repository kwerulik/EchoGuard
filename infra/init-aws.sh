#!/bin/bash
echo "--- INICJALIZACJA ECHOGUARD ---"

# 1. Tworzenie Bucketa S3 na dane z czujników
awslocal s3 mb s3://echoguard-data

echo "--- GOTOWE! Bucket 'echoguard-data' utworzony. ---"