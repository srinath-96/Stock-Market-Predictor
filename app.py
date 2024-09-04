import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("StockPredictPy")


stocks = ("AAPL", "GOOG", "MSFT", "GME", "AMZN", "TSLA", "BTC-USD", "ETH-USD", "DOGE-USD", "ADA-USD", "XRP-USD", "A", "AA", "AACG", "AACI", "AAGR", "AAL", "AAMC", "AAME", "AAN", "AAOI", "AAON", "AAP", "AAPL", "AAT", "AAU", "AAWW", "AB", "ABB", "ABBV", "ABC", "ABCB", "ABCL", "ABCM", "ABEO", "ABEV", "ABG", "ABIO", "ABM", "ABMD", "ABNB", "ABOS", "ABR", "ABST", "ABT", "ABTX", "ABUS", "AC", "ACA", "ACAC", "ACAD", "ACAH", "ACB", "ACBI", "ACC", "ACCD", "ACCO", "ACEL", "ACER", "ACET", "ACEV", "ACGL", "ACH", "ACHC", "ACHL", "ACHV", "ACI", "ACIA", "ACIC", "ACII", "ACIU", "ACIW", "ACKIT", "ACLS", "ACM", "ACMR", "ACN", "ACNB", "ACND", "ACOR", "ACP", "ACR", "ACRE", "ACRS", "ACRX", "ACST", "ACTC", "ACTG", "ACU", "ACV", "ACVA", "ACVF", "ACVX", "ACWI", "ACWX", "ACXM", "ACY", "ADAG", "ADAP", "ADBE", "ADC", "ADCT", "ADES", "ADEX", "ADF", "ADFI", "ADI", "ADIL", "ADM", "ADMA", "ADME", "ADMP", "ADMS", "ADN", "ADNT", "ADOC", "ADP", "ADPT", "ADRA", "ADRO", "ADS", "ADSK", "ADT", "ADTN", "ADTX", "ADUS", "ADV", "ADVM", "ADX", "ADXS", "AE", "AEE", "AEG", "AEGN", "AEHL", "AEHR", "AEI", "AEIS", "AEL", "AEM", "AEMD", "AEO", "AEP", "AER","ZYME", "ZYNE", "ZYXI", "ZZ", "ZZZ","ZETA", "ZEO", "ZEUS")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")


st.markdown("""
    <style>
    .css-1l06vq2 {
        width: 2000px;
    }
    </style>
    """, unsafe_allow_html=True)

st.subheader('Details')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text="Time Series data with Rangeslider", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('Forecast Data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)
