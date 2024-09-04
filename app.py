import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from google.cloud import language_v1
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go
import plotly.express as px

# Configuration
START = "2014-01-01"
TODAY = pd.to_datetime("today").strftime("%Y-%m-%d")
STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "JPM", "V"]

# Initialize Sentiment Analyzers
vader_analyzer = SentimentIntensityAnalyzer()
client = language_v1.LanguageServiceClient()

# Function to load stock data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Function to fetch historical news data
def get_news(stock_symbol, start_date, end_date):
    api_key = "a3d435ee70484c19b4fdc4b3e537d9fd"  # Replace with your NewsAPI key
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&from={start_date}&to={end_date}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles

# Sentiment Analysis Functions
def analyze_sentiment_vader(text):
    return vader_analyzer.polarity_scores(text)['compound']

def analyze_sentiment_textblob(text):
    return TextBlob(text).sentiment.polarity

def analyze_sentiment_google(text):
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    return sentiment.score

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

st.subheader('Prophet Forecast')
st.write(forecast.tail())
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

# Historical Sentiment Analysis
st.subheader("Historical Sentiment Trends")
historical_data = get_news(selected_stock, START, TODAY)
sentiments = {
    'Date': [],
    'VADER Sentiment': [],
    'TextBlob Sentiment': [],
    'Google Sentiment': []
}

for article in historical_data:
    date = pd.to_datetime(article['publishedAt']).date()
    description = article['description'] or ''
    
    sentiments['Date'].append(date)
    sentiments['VADER Sentiment'].append(analyze_sentiment_vader(description))
    sentiments['TextBlob Sentiment'].append(analyze_sentiment_textblob(description))
    sentiments['Google Sentiment'].append(analyze_sentiment_google(description))

sentiments_df = pd.DataFrame(sentiments)

fig2 = px.line(sentiments_df, x='Date', y=['VADER Sentiment', 'TextBlob Sentiment', 'Google Sentiment'], title='Sentiment Trends')
st.plotly_chart(fig2)

# Display news and sentiment scores
st.subheader("Recent News and Sentiment Analysis")
for article in historical_data[:5]:
    st.write(f"**{article['title']}**")
    st.write(article['description'])
    st.write(f"VADER Sentiment Score: {analyze_sentiment_vader(article['description']):.2f}")
    st.write(f"TextBlob Sentiment Score: {analyze_sentiment_textblob(article['description']):.2f}")
    st.write(f"Google Sentiment Score: {analyze_sentiment_google(article['description']):.2f}")
    st.write("---")
