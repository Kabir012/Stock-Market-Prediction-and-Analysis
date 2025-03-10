# Stock-Market-Prediction-and-Analysis

## Overview
This project is a **Stock Trend Prediction** web application built using **Python, Streamlit, and Keras**. The application fetches stock market data from **Yahoo Finance**, processes it using **LSTM (Long Short-Term Memory) Neural Networks**, and visualizes past trends along with future predictions.

## Features
- Fetches real-time stock data from **Yahoo Finance**.
- Displays **descriptive statistics** of the stock price.
- Visualizes **Closing Price vs Time**.
- Uses a pre-trained **LSTM model** to predict future stock trends.
- Compares **actual vs predicted** stock prices using a line graph.

## Technologies Used
- **Python**
- **Streamlit** (for web interface)
- **Yahoo Finance API (yfinance)** (for fetching stock data)
- **Pandas** (for data handling)
- **Matplotlib** (for data visualization)
- **NumPy** (for numerical computations)
- **Keras & TensorFlow** (for LSTM model)
- **MinMaxScaler** (for data normalization)

## Installation & Setup
1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/stock-trend-prediction.git
   cd stock-trend-prediction
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Enter a stock ticker symbol (e.g., **AAPL**, **GOOGL**, **TSLA**).
2. The application will fetch and display historical data.
3. View **Closing Price vs Time** chart.
4. The trained LSTM model will predict future trends.
5. View **Predictions vs Actual Prices** graph.

## Model Training
- The **LSTM model** was trained using historical stock price data.
- The dataset is split into **80% training** and **20% testing**.
- The data is normalized using **MinMaxScaler**.
- The model uses **60 previous time steps** to predict the next value.
- The trained model is saved as **keras_model.h5**.


## Future Improvements
- Implement additional **technical indicators** for better accuracy.
- Allow users to **adjust model parameters** dynamically.
- Extend the model to predict stock **price movement trends**.
- Deploy the application using **Docker or cloud platforms**.

## Contributing
Feel free to submit **issues, feature requests, or pull requests** to improve this project!


