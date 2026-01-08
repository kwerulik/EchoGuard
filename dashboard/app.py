# dashboard/app.py
import streamlit as st
import boto3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from botocore.exceptions import NoCredentialsError

# --- KONFIGURACJA ---
st.set_page_config(page_title="EchoGuard Dashboard", layout="wide")

# Podłączenie do LocalStack
dynamodb = boto3.resource('dynamodb',
                          endpoint_url='http://localhost:4566',
                          region_name='us-east-1',
                          aws_access_key_id='test',
                          aws_secret_access_key='test')

TABLE_NAME = "EchoGuardResults"
table = dynamodb.Table(TABLE_NAME)


def get_data():
    """Pobiera wszystkie wyniki z DynamoDB i konwertuje na DataFrame"""
    try:
        response = table.scan()
        data = response['Items']

        # Paginacja (gdyby danych było bardzo dużo)
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                ExclusiveStartKey=response['LastEvaluatedKey'])
            data.extend(response['Items'])

        df = pd.DataFrame(data)
        if not df.empty:
            # Konwersja typów
            df['mse_value'] = df['mse_value'].astype(float)
            df['threshold'] = df['threshold'].astype(float)
            # Parsowanie daty (timestamp w nazwie pliku to nasza oś czasu)
            # Format: 2004.02.12.10.32.39 -> datetime
            df['datetime'] = pd.to_datetime(
                df['timestamp'], format='%Y-%m-%d-%H-%M-%S')
            df = df.sort_values(by='datetime')
        return df
    except Exception as e:
        st.error(f"Błąd połączenia z DynamoDB: {e}")
        return pd.DataFrame()


# --- UI ---
st.title("🛡️ EchoGuard: Predictive Maintenance Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Status Systemu", "ONLINE", "LocalStack")

# Placeholders na dane
chart_placeholder = st.empty()
metrics_placeholder = st.empty()

# Pętla odświeżania (symulacja Real-Time Dashboard)
# W Streamlit można to zrobić przyciskiem "Odśwież" lub pętlą z rerun
if st.checkbox("🔴 Włącz Live Monitoring", value=True):
    while True:
        df = get_data()

        if not df.empty:
            last_record = df.iloc[-1]
            current_mse = last_record['mse_value']
            threshold = last_record['threshold']
            status = last_record['status']

            # 1. KPI Metrics
            with metrics_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Ostatni plik", last_record['timestamp'])

                delta_color = "normal" if current_mse < threshold else "inverse"
                c2.metric("Aktualne MSE (Błąd)",
                          f"{current_mse:.6f}", delta_color=delta_color)

                status_color = "🟢" if status == "HEALTHY" else "🚨"
                c3.markdown(f"### Stan: {status_color} {status}")

            # 2. Wykres główny (Health Index)
            fig = go.Figure()

            # Linia MSE
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df['mse_value'],
                mode='lines',
                name='MSE (Reconstruction Error)',
                line=dict(color='#00CC96', width=2)
            ))

            # Linia Progu (Threshold)
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=[threshold] * len(df),
                mode='lines',
                name='Próg Awarii',
                line=dict(color='#EF553B', width=2, dash='dash')
            ))

            fig.update_layout(
                title='Krzywa Życia Łożyska (Run-to-Failure)',
                xaxis_title='Czas',
                yaxis_title='MSE Loss (Anomalia)',
                template='plotly_dark',
                height=500
            )

            chart_placeholder.plotly_chart(
                fig, 
                width="stretch",
                key=f"live_chart_{time.time()}"
            )
        else:
            chart_placeholder.warning("Oczekiwanie na dane w DynamoDB...")

        time.sleep(2)
