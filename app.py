import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

# Constants
START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
NEWS_API_KEY = 'a3d435ee70484c19b4fdc4b3e537d9fd'  # Replace with your NewsAPI key

# Setup
st.title("StockPredictPy")

# Stock options
stocks = ["AAPL", "GOOG", "MSFT", "GME", "AMZN", "TSLA", "NVDA", "META", "MS", "GS"]

# User selection
selected_stock = st.selectbox("Select dataset for prediction", stocks)

@st.cache
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_data(selected_stock)

if not data.empty:
    st.subheader('Details')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
        fig.update_layout(title_text="Time Series Data with Rangeslider", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Prediction
    n_years = st.slider("Years of prediction:", 1, 4)
    period = n_years * 365

    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    try:
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
    except Exception as e:
        st.error(f"Error in forecasting: {e}")

    # Sentiment Analysis with NewsAPI
    def fetch_news(ticker, start_date, end_date):
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': ticker,
            'from': start_date,
            'to': end_date,
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'relevancy'
        }
        response = requests.get(url, params=params)
        news_data = response.json()
        return news_data

    def analyze_sentiment(news_articles):
        sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = []
        dates = pd.date_range(start=historical_start_date, end=TODAY)

        for date in dates:
            daily_articles = [article['description'] for article in news_articles if article['publishedAt'].startswith(date.strftime("%Y-%m-%d"))]
            if daily_articles:
                combined_text = ' '.join(daily_articles)
                sentiment = sentiment_analyzer.polarity_scores(combined_text)
                sentiment_scores.append({'date': date, 'sentiment': sentiment['compound']})

        return pd.DataFrame(sentiment_scores)

    # Historical period setup
    historical_start_date = (date.today() - timedelta(days=180)).strftime("%Y-%m-%d")
    news_data = fetch_news(selected_stock, historical_start_date, TODAY)
    articles = news_data.get('articles', [])
    sentiment_df = analyze_sentiment(articles)

    if not sentiment_df.empty:
        st.subheader('Historical Sentiment Trends')

        # Visualize sentiment trends
        fig = px.line(sentiment_df, x='date', y='sentiment', title=f'Sentiment Trends for {selected_stock}')
        st.plotly_chart(fig)
else:
    st.error("No data available to display.")
