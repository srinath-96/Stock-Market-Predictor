import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go

# Configuration
START = "2014-01-01"
TODAY = pd.to_datetime("today").strftime("%Y-%m-%d")
STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "JPM", "V"]

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to load stock data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Function to fetch news
def get_news(stock_symbol):
    api_key = "a3d435ee70484c19b4fdc4b3e537d9fd"  # Replace with your NewsAPI key
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles

# Function to perform sentiment analysis
def analyze_sentiment(articles):
    sentiments = [analyzer.polarity_scores(article['description'])['compound'] for article in articles if article['description']]
    return sum(sentiments) / len(sentiments) if sentiments else 0

# Streamlit App
st.title("Stock Market Predictor and Sentiment Analyzer")

selected_stock = st.selectbox("Select a stock for prediction", STOCKS)

data = load_data(selected_stock)
st.write(data.tail())

# Plot raw data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open"))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
fig.update_layout(title_text="Stock Price Over Time", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Forecasting with Prophet
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())
st.write('Forecast Plot')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Forecasting with ARIMA
st.subheader("ARIMA Forecast")
model = ARIMA(df_train['y'], order=(5,1,0))
model_fit = model.fit()
forecast_arima = model_fit.forecast(steps=period)
forecast_arima_df = pd.DataFrame({
    'Date': pd.date_range(start=data['Date'].max(), periods=period+1, closed='right'),
    'Forecast': forecast_arima
})
st.line_chart(forecast_arima_df.set_index('Date'))

# Sentiment Analysis
st.subheader("Sentiment Analysis")
articles = get_news(selected_stock)
avg_sentiment = analyze_sentiment(articles)
st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")

if articles:
    for article in articles[:5]:
        st.write(f"**{article['title']}**")
        st.write(article['description'])
        st.write(f"Sentiment Score: {analyzer.polarity_scores(article['description'])['compound']:.2f}")

