
import numpy as np
np.float_ = np.float64
import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

# Streamlit app title
st.title('Stock Prediction App for Major Companies')

# Dropdown menu for selecting the stock
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
selected_stock = st.selectbox('Select Stock', stocks)

# Dropdown menu for selecting the time interval
intervals = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y"
}
selected_interval = st.selectbox('Select Time Interval', list(intervals.keys()))

# Input for the forecast period
forecast_period = st.number_input('Enter number of days to predict', min_value=1, max_value=365, value=30)

# Fetch the historical data when the user inputs a ticker
if selected_stock and selected_interval:
    # Display a loading spinner while fetching data
    with st.spinner(f'Fetching data for {selected_stock}...'):
        data = yf.download(selected_stock, period=intervals[selected_interval], interval="1d")
        
        if not data.empty:
            # Prepare data for Prophet
            df = data.reset_index()[['Date', 'Close']]
            df.columns = ['ds', 'y']
            
            # Display raw data
            st.subheader(f'{selected_stock} - Last {selected_interval} of Data')
            st.write(df.tail())
            
            # Train the Prophet model
            model = Prophet()
            model.fit(df)
            
            # Create a dataframe for future predictions
            future = model.make_future_dataframe(periods=forecast_period)
            forecast = model.predict(future)
            
            # Display forecasted data
            st.subheader(f'Forecast for the next {forecast_period} days')
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            
            # Plot the forecast
            st.subheader('Forecast Plot')
            fig1 = plot_plotly(model, forecast)
            st.plotly_chart(fig1)
            
            # Plot the forecast components
            st.subheader('Forecast Components')
            fig2 = model.plot_components(forecast)
            st.write(fig2)
        else:
            st.error(f"No data found for ticker {selected_stock}. Please try a different ticker.")
