# Recurrent Neural Network
# Variant: 60 timesteps, 4 LSTM layers

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Use relative paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'Recurrent_Neural_Networks')
TIMESTEPS = 60


def load_data(filename):
    """Load CSV data with error handling."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    return pd.read_csv(filepath)


def build_model(timesteps):
    """Build a 4-layer LSTM model."""
    model = Sequential()
    model.add(LSTM(units=3, return_sequences=True, input_shape=(timesteps, 1)))
    model.add(LSTM(units=3, return_sequences=True))
    model.add(LSTM(units=3, return_sequences=True))
    model.add(LSTM(units=3))
    model.add(Dense(units=1))
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model


def main():
    # Part 1 - Data Preprocessing
    dataset_train = load_data('Google_Stock_Price_Train.csv')
    training_set = dataset_train.iloc[:, 1:2].values

    # Feature Scaling (fit on training data only)
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Creating a data structure with 60 timesteps and t+1 output
    num_samples = len(training_set_scaled)
    X_train = []
    y_train = []
    for i in range(TIMESTEPS, num_samples):
        X_train.append(training_set_scaled[i - TIMESTEPS:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping to [samples, timesteps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Part 2 - Building the RNN
    regressor = build_model(TIMESTEPS)
    regressor.summary()
    regressor.fit(X_train, y_train, epochs=100, batch_size=32)

    # Part 3 - Making the predictions and visualising the results
    dataset_test = load_data('Google_Stock_Price_Test.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values

    # Use sc.transform() (NOT fit_transform) to avoid data leakage
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - TIMESTEPS:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(TIMESTEPS, TIMESTEPS + len(dataset_test)):
        X_test.append(inputs[i - TIMESTEPS:i, 0])
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


if __name__ == "__main__":
    main()
