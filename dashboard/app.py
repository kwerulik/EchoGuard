import streamlit as st
import boto3
import pandas as pd
import time
from botocore.config import Config
from decimal import Decimal

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="EchoGuard | IoT Monitoring",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def get_dynamodb_table():
    try:
        config = Config(
            connect_timeout=2,
            read_timeout=2,
            retries={'max_attempts': 1}
        )

        dynamodb = boto3.resource(
            'dynamodb',
            endpoint_url='http://localhost:4566',
            region_name='us-east-1',
            config=config,
            aws_access_key_id='test',
            aws_secret_access_key='test'
        )
        table = dynamodb.Table('EchoGuardResults')
        table.load()
        return table
    except Exception as e:
        st.error(f"⚠️ Problem z połączeniem do LocalStack: {e}")
        return None

table = get_dynamodb_table()


def fetch_data():
    if not table:
        return pd.DataFrame()

    try:
        response = table.scan()
        items = response.get('Items', [])
        if not items:
            return pd.DataFrame()

        df = pd.DataFrame(items)

        # 1. Konwersja liczb
        df['mse_value'] = pd.to_numeric(df['mse_value'], errors='coerce')
        df['threshold'] = pd.to_numeric(df['threshold'], errors='coerce')

        # 2. CZYSZCZENIE DATY (To naprawi 'None' w tabeli)
        # Usuwamy prefiksy z nazwy timestampu
        df['clean_timestamp'] = df['timestamp'].astype(
            str).str.replace('NORMAL_', '').str.replace('ANOMALY_', '')

        # Zamieniamy kropki na myślniki (jeśli symulator używał kropek)
        df['clean_timestamp'] = df['clean_timestamp'].str.replace('.', '-')

        # Parsujemy datę
        df['timestamp_dt'] = pd.to_datetime(
            df['clean_timestamp'], format='%Y-%m-%d-%H-%M-%S', errors='coerce')

        # Usuwamy wiersze, gdzie data się nie udała
        df = df.dropna(subset=['timestamp_dt'])

        return df.sort_values('timestamp_dt')
    except Exception as e:
        st.error(f"Błąd danych: {e}")
        return pd.DataFrame()

st.title("🛡️ EchoGuard: Monitor Anomalii Łożysk")
st.markdown(f"**Status połączenia:** {'✅ Online' if table else '❌ Offline'}")

col_btn, col_info = st.columns([1, 5])
with col_btn:
    if st.button('🔄 Odśwież teraz'):
        st.rerun()

df = fetch_data()

if not df.empty:
    latest_record = df.iloc[-1]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Ostatni Błąd Rekonstrukcji (MSE)",
            value=f"{latest_record['mse_value']:.6f}",
            delta=f"Limit: {latest_record['threshold']:.6f}",
            delta_color="off"
        )

    with col2:
        is_healthy = latest_record['status'] == "HEALTHY"
        status_text = "🟢 OK (Zdrowe)" if is_healthy else "🔴 AWARIA (Anomalia)"
        st.metric(label="Status Maszyny", value=status_text)

    with col3:
        st.metric(label="Ostatnia Aktualizacja",
                  value=latest_record['timestamp'])

    st.divider()
    st.subheader("📉 Wykres Wibracji w Czasie")

    recent_df = df.tail(50)
    chart_df = df[['timestamp_dt', 'mse_value', 'threshold']].copy()
    chart_df = chart_df.set_index('timestamp_dt')

    st.line_chart(chart_df, color=["#0000FF", "#FF0000"])


    with st.expander("🔍 Pokaż szczegółowe logi"):
        st.dataframe(df.sort_values('timestamp_dt',
                     ascending=False).style.format({"mse_value": "{:.6f}"}))

else:
    st.info("Baza danych jest pusta lub nie udało się pobrać danych. Uruchom symulator!")

time.sleep(5)
st.rerun()
