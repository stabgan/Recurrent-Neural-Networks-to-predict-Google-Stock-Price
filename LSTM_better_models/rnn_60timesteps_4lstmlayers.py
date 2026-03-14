# Recurrent Neural Network
# Variant: 60 timesteps, 4 LSTM layers

# Part 1 - Data Preprocessing

# Importing the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Use relative paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'Recurrent_Neural_Networks')

# Importing the training set
dataset_train = pd.read_csv(os.path.join(DATA_DIR, 'Google_Stock_Price_Train.csv'))
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling (fit on training data only)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and t+1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units=3, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# Adding a second LSTM layer
regressor.add(LSTM(units=3, return_sequences=True))

# Adding a third LSTM layer
regressor.add(LSTM(units=3, return_sequences=True))

# Adding a fourth LSTM layer
regressor.add(LSTM(units=3))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='rmsprop', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv(os.path.join(DATA_DIR, 'Google_Stock_Price_Test.csv'))
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# Use sc.transform() (NOT fit_transform) to avoid data leakage
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction (60 timesteps, 4 LSTM layers)')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
