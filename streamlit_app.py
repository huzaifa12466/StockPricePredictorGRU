import streamlit as st
import pandas as pd
import yfinance as yf
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.graph_objects as go
from models.model import GRUStock
from models.predict import predict_next_days

st.set_page_config(page_title="Interactive Multi-Ticker Stock Prediction", layout="wide")
st.title("📈 Multi-Ticker Stock Price Prediction (GRU)")

# ------------------ User Inputs ------------------
tickers = ["AMZN", "AAPL", "MSFT", "GOOGL", "META"]
selected_ticker = st.selectbox("Select Ticker", tickers)
compare_ticker = st.selectbox("Compare With (Optional)", ["None"] + tickers)
prediction_days = st.slider("Days to Predict", 1, 30, 10)
seq_length = 30

# ------------------ Load Data ------------------
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2015-01-01", end="2025-01-01")
    df = df[['Close']].dropna().reset_index(drop=True)
    return df

data_main = load_data(selected_ticker)
if compare_ticker != "None":
    data_compare = load_data(compare_ticker)

# ------------------ Scale Data ------------------
scaler = MinMaxScaler()
scaled_main = scaler.fit_transform(data_main[['Close']].values)
if compare_ticker != "None":
    scaler_compare = MinMaxScaler()
    scaled_compare = scaler_compare.fit_transform(data_compare[['Close']].values)

# ------------------ Load Model ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUStock().to(device)
model.load_state_dict(torch.load("models/best_multi_ticker_model.pth", map_location=device))
model.eval()

# ------------------ Predictions ------------------
predicted_main = predict_next_days(model, scaler, scaled_main, seq_length, prediction_days, device)
recent_main = scaler.inverse_transform(scaled_main[-seq_length:].reshape(-1,1)).flatten()
all_main = np.concatenate([recent_main, predicted_main])
x_axis_main = np.arange(1, len(all_main)+1)

if compare_ticker != "None":
    predicted_compare = predict_next_days(model, scaler_compare, scaled_compare, seq_length, prediction_days, device)
    recent_compare = scaler_compare.inverse_transform(scaled_compare[-seq_length:].reshape(-1,1)).flatten()
    all_compare = np.concatenate([recent_compare, predicted_compare])
    x_axis_compare = np.arange(1, len(all_compare)+1)

# ------------------ Interactive Line Plot ------------------
st.subheader("📊 Interactive Line Plot: Recent + Predicted Prices")
fig = go.Figure()

# Selected ticker
fig.add_trace(go.Scatter(x=x_axis_main[:seq_length], y=recent_main,
                         mode='lines+markers', name=f'{selected_ticker} Recent'))
fig.add_trace(go.Scatter(x=x_axis_main[seq_length:], y=predicted_main,
                         mode='lines+markers', name=f'{selected_ticker} Predicted'))

# Compare ticker
if compare_ticker != "None":
    fig.add_trace(go.Scatter(x=x_axis_compare[:seq_length], y=recent_compare,
                             mode='lines+markers', name=f'{compare_ticker} Recent'))
    fig.add_trace(go.Scatter(x=x_axis_compare[seq_length:], y=predicted_compare,
                             mode='lines+markers', name=f'{compare_ticker} Predicted'))

# Prediction start line
fig.add_vline(x=seq_length, line=dict(color='gray', dash='dash'), annotation_text="Prediction Start", annotation_position="top right")

fig.update_layout(xaxis_title="Days",
                  yaxis_title="Close Price (USD)",
                  hovermode="x unified",
                  width=1000, height=500)

st.plotly_chart(fig)

# ------------------ Predicted Prices Table ------------------
st.subheader("📋 Predicted Prices Table")
pred_table = pd.DataFrame({f"{selected_ticker}": [round(p,2) for p in predicted_main]})
if compare_ticker != "None":
    pred_table[compare_ticker] = [round(p,2) for p in predicted_compare]
st.table(pred_table)
