# 📈 Google Stock Price Prediction with LSTM

Predicting Google stock prices using Long Short-Term Memory (LSTM) recurrent neural networks. Trains on historical stock data (2012–2016) and predicts January 2017 prices.

## What It Does

Uses Keras LSTM networks to learn temporal patterns in Google's opening stock price, then forecasts future prices. The project includes multiple model configurations to compare how timestep window size and network depth affect prediction quality.

## Project Structure

```
├── Recurrent_Neural_Networks/
│   ├── rnn.py                          # Main model — 4 LSTM layers, 60 timesteps, dropout
│   ├── Google_Stock_Price_Train.csv    # Training data (Jan 2012 – Dec 2016)
│   └── Google_Stock_Price_Test.csv     # Test data (Jan 2017)
├── LSTM_better_models/
│   ├── rnn_20timesteps_1lstmlayers.py  # 20 timesteps, 1 LSTM layer
│   ├── rnn_20timesteps_4lstmlayers.py  # 20 timesteps, 4 LSTM layers
│   ├── rnn_60timesteps_1lstmlayers.py  # 60 timesteps, 1 LSTM layer
│   └── rnn_60timesteps_4lstmlayers.py  # 60 timesteps, 4 LSTM layers
└── LICENSE
```

## Tech Stack

- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn (MinMaxScaler)

## Getting Started

```bash
# Install dependencies
pip install tensorflow numpy pandas matplotlib scikit-learn

# Run the main model
cd Recurrent_Neural_Networks
python rnn.py

# Or try the model variants
cd LSTM_better_models
python rnn_60timesteps_4lstmlayers.py
```

> Training runs for 100 epochs with batch size 32. The main model uses Adam optimizer with 50 LSTM units per layer and 20% dropout.

## Results

Comparison of different model configurations (red = actual price, blue = predicted):

| 20 Timesteps, 1 LSTM Layer | 20 Timesteps, 4 LSTM Layers |
|:--:|:--:|
| ![](LSTM_better_models/rnn_20timesteps_1lstmlayers.png) | ![](LSTM_better_models/rnn_20timesteps_4lstmlayers.png) |

| 60 Timesteps, 1 LSTM Layer | 60 Timesteps, 4 LSTM Layers |
|:--:|:--:|
| ![](LSTM_better_models/rnn_60timesteps_1lstmlayers.png) | ![](LSTM_better_models/rnn_60timesteps_4lstmlayers.png) |

Longer timestep windows (60 vs 20) generally produce smoother, more accurate predictions. Adding more LSTM layers helps capture complex temporal dependencies.

## Known Issues

- The model variant scripts in `LSTM_better_models/` use `sc.fit_transform()` on combined train+test data during prediction, which introduces minor data leakage. The main `rnn.py` handles this correctly by only calling `sc.transform()` on test inputs.
- Keras imports use the standalone `keras` package syntax. For TensorFlow 2.x, update imports to `from tensorflow.keras.models import Sequential`, etc.

## Author

[Kaustabh Ganguly](https://github.com/stabgan)

## License

[MIT](LICENSE) © 2018 Kaustabh Ganguly
