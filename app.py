import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time

# Page configuration
st.set_page_config(
    page_title="⛽ Ethereum Gas Surge Predictor",
    page_icon="⛽",
    layout="wide"
)

st.title("⛽ Ethereum Gas Surge Predictor")
st.markdown("Get real-time Ethereum gas prices and AI-powered predictions for the next hour")

# API Configuration
try:
    ETHERSCAN_API_KEY = st.secrets["ETHERSCAN_API_KEY"]
except (FileNotFoundError, KeyError):
    ETHERSCAN_API_KEY = "YourApiKeyToken"
ETHERSCAN_GAS_ORACLE_URL = "https://api.etherscan.io/api?module=gastracker&action=gasoracle"

# Function to fetch current gas prices
@st.cache_data(ttl=60)
def fetch_current_gas_prices(demo_mode=False):
    """Fetch current gas prices from Etherscan Gas Oracle"""
    if demo_mode:
        # Return demo data for testing
        base_price = 30 + np.random.uniform(-5, 5)
        return {
            'low': max(10, base_price - 5),
            'normal': base_price,
            'high': base_price + 10,
            'timestamp': datetime.now()
        }
    
    try:
        url = f"{ETHERSCAN_GAS_ORACLE_URL}&apikey={ETHERSCAN_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == '1' and data['message'] == 'OK':
            result = data['result']
            return {
                'low': float(result['SafeGasPrice']),
                'normal': float(result['ProposeGasPrice']),
                'high': float(result['FastGasPrice']),
                'timestamp': datetime.now()
            }
        else:
            st.error(f"API Error: {data.get('message', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch gas prices: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

# Function to generate synthetic historical data (for demonstration)
def generate_historical_data(current_price, hours=24):
    """Generate synthetic historical gas price data for ML model training"""
    timestamps = []
    prices = []
    
    now = datetime.now()
    base_price = current_price
    
    for i in range(hours * 12):  # 12 data points per hour (every 5 minutes)
        time_offset = timedelta(minutes=5 * i)
        timestamp = now - timedelta(hours=hours) + time_offset
        
        # Create realistic-looking gas price fluctuations
        hour_of_day = timestamp.hour
        day_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24)  # Daily pattern
        noise = np.random.normal(0, 0.1)  # Random noise
        trend = 0.002 * i  # Slight upward trend
        
        price = base_price * (day_factor + noise + trend)
        price = max(10, price)  # Ensure price doesn't go below 10 Gwei
        
        timestamps.append(timestamp)
        prices.append(price)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'gas_price': prices
    })
    
    return df

# Function to train ML model and make predictions
def predict_next_hour_gas(historical_df, current_price):
    """Train a simple linear regression model to predict average gas for next hour"""
    # Prepare features
    df = historical_df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['time_index'] = range(len(df))
    
    # Use recent data (last 6 hours) for training
    recent_df = df.tail(72)  # 72 = 6 hours * 12 points/hour
    
    # Features: time_index, hour of day, minute
    X = recent_df[['time_index', 'hour', 'minute']].values
    y = recent_df['gas_price'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next hour (12 points)
    future_predictions = []
    future_timestamps = []
    
    last_index = df['time_index'].max()
    last_timestamp = df['timestamp'].max()
    
    for i in range(1, 13):  # Next 12 points (1 hour)
        future_timestamp = last_timestamp + timedelta(minutes=5 * i)
        future_hour = future_timestamp.hour
        future_minute = future_timestamp.minute
        future_index = last_index + i
        
        X_future = np.array([[future_index, future_hour, future_minute]])
        prediction = model.predict(X_future)[0]
        
        future_predictions.append(max(10, prediction))  # Ensure non-negative
        future_timestamps.append(future_timestamp)
    
    avg_predicted = np.mean(future_predictions)
    
    # Create prediction dataframe
    prediction_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'gas_price': future_predictions
    })
    
    return avg_predicted, prediction_df

# Function to generate recommendation
def get_recommendation(current_price, predicted_price):
    """Generate recommendation based on current and predicted prices"""
    difference_pct = ((predicted_price - current_price) / current_price) * 100
    
    if difference_pct > 10:
        return "⚠️ Might want to wait", "Gas prices are predicted to increase by {:.1f}%. Consider waiting for a better rate.".format(difference_pct), "warning"
    elif difference_pct < -5:
        return "✅ Send now", "Gas prices are predicted to decrease by {:.1f}%. Now is a good time to transact!".format(abs(difference_pct)), "success"
    else:
        return "✅ Send now", "Gas prices are relatively stable (predicted change: {:.1f}%). Good time to transact.".format(difference_pct), "info"

# Main application
def main():
    # Check for demo mode
    demo_mode = st.sidebar.checkbox("Demo Mode (No API Required)", value=False)
    
    if demo_mode:
        st.sidebar.info("📍 Demo mode is enabled. Using simulated data.")
    
    # Fetch current gas prices
    with st.spinner("Fetching current gas prices..."):
        gas_data = fetch_current_gas_prices(demo_mode=demo_mode)
    
    if gas_data is None:
        st.warning("⚠️ Unable to fetch current gas prices. Please check your API key or try again later.")
        st.info("💡 **Tip**: To use this app, you need an Etherscan API key. Get one for free at https://etherscan.io/apis")
        st.info("💡 **Alternative**: Enable 'Demo Mode' in the sidebar to test the app with simulated data.")
        return
    
    # Display current gas prices
    st.header("📊 Current Gas Prices")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🐢 Low (Safe)",
            value=f"{gas_data['low']:.0f} Gwei"
        )
    
    with col2:
        st.metric(
            label="⚡ Normal (Standard)",
            value=f"{gas_data['normal']:.0f} Gwei"
        )
    
    with col3:
        st.metric(
            label="🚀 High (Fast)",
            value=f"{gas_data['high']:.0f} Gwei"
        )
    
    st.caption(f"Last updated: {gas_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate historical data and predictions
    st.header("🔮 ML-Based Prediction")
    
    with st.spinner("Training ML model and generating predictions..."):
        # Use normal gas price as reference
        current_price = gas_data['normal']
        
        # Generate synthetic historical data
        historical_df = generate_historical_data(current_price, hours=24)
        
        # Make predictions
        predicted_avg, prediction_df = predict_next_hour_gas(historical_df, current_price)
    
    # Display prediction
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Current Average Gas",
            value=f"{current_price:.1f} Gwei"
        )
    
    with col2:
        delta = predicted_avg - current_price
        st.metric(
            label="Predicted Next Hour Average",
            value=f"{predicted_avg:.1f} Gwei",
            delta=f"{delta:+.1f} Gwei"
        )
    
    # Generate and display recommendation
    st.header("💡 Recommendation")
    recommendation, explanation, status = get_recommendation(current_price, predicted_avg)
    
    if status == "success":
        st.success(f"### {recommendation}")
    elif status == "warning":
        st.warning(f"### {recommendation}")
    else:
        st.info(f"### {recommendation}")
    
    st.write(explanation)
    
    # Plot chart
    st.header("📈 Gas Price Trend & Prediction")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data (last 6 hours)
    recent_historical = historical_df.tail(72)
    ax.plot(recent_historical['timestamp'], recent_historical['gas_price'], 
            label='Recent Historical', color='#1f77b4', linewidth=2)
    
    # Plot prediction
    ax.plot(prediction_df['timestamp'], prediction_df['gas_price'], 
            label='Predicted (Next Hour)', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Mark current time
    ax.axvline(x=gas_data['timestamp'], color='red', linestyle=':', 
               linewidth=2, label='Current Time', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Gas Price (Gwei)', fontsize=12)
    ax.set_title('Ethereum Gas Price: Historical vs Predicted', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Additional information
    with st.expander("ℹ️ About this predictor"):
        st.markdown("""
        This tool provides:
        - **Real-time gas prices** from Etherscan Gas Oracle
        - **ML-based predictions** using Linear Regression on recent gas price trends
        - **Clear recommendations** to help you decide when to transact
        
        **How it works:**
        1. Fetches current gas prices (low/normal/high) from Etherscan
        2. Analyzes historical patterns in gas prices
        3. Uses machine learning to predict average gas price for the next hour
        4. Provides actionable recommendations based on predicted trends
        
        **Note:** This is a demonstration tool. For production use, integrate with a real-time 
        gas price data feed and retrain the model regularly with actual historical data.
        """)
    
    # Auto-refresh option
    if not demo_mode:
        st.sidebar.header("⚙️ Settings")
        auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)
        
        if auto_refresh:
            time.sleep(60)
            st.rerun()

if __name__ == "__main__":
    main()
