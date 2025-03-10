# Importing required libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import math

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model
import streamlit as st



# Define the start and end date of the analysis period
start_date = datetime(2012, 1, 1)
end_date = datetime(2022, 1, 1)

st.title('Stock Trend Prediction')
# Define the stock ticker symbol you want to analyze
user_input = st.text_input('Enter Stock Ticker' , 'AAPL')

# Fetching data from Yahoo Finance
df = yf.download(user_input, start=start_date, end=end_date)

# Descrribing Data
st.subheader('Data from 2012 - 2022')
st.write(df.describe())

# Visualisations
st.subheader('Closing Price vs Time Chart')
# Plot the stock price over time
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.title("Stock Price over Time")
plt.xlabel("Date")
plt.ylabel("Price")
st.pyplot(fig)


# Create a new dataframe with only the close column
data = df.filter(["Close"])
# Convert the dataframe to a numpy array
dataset = data.values
training_data_len = math.ceil(len(dataset)*.8)
# training_data_len

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# scaled_data

# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:training_data_len , :]


# Loading my model
model = load_model('keras_model.h5')

# Create the testing dataset
# Create a new array containing scaled values
test_data = scaled_data[training_data_len - 60: , : ]
# Create the datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i,0])

# Convert the data into a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))

# Get the models prediced prices values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt( np.mean(predictions - y_test )**2 ) 
# rmse

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualise the data
st.subheader('Predicitions vs Original')
fig2 = plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot(fig2)
# Show the valid and predicted prices
st.subheader('The actual and predicted prices')
valid