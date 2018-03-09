# Recurrent Neural Networks to predict Google Stock Price

I tried to predict google stock price using LSTMs

Long short-term memory (LSTM) units (or blocks) are a building unit for layers of a recurrent neural network (RNN). A RNN composed of LSTM units is often called an LSTM network. A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell is responsible for "remembering" values over arbitrary time intervals; hence the word "memory" in LSTM. Each of the three gates can be thought of as a "conventional" artificial neuron, as in a multi-layer (or feedforward) neural network: that is, they compute an activation (using an activation function) of a weighted sum. Intuitively, they can be thought as regulators of the flow of values that goes through the connections of the LSTM; hence the denotation "gate". There are connections between these gates and the cell.

The results of the better models in prediction are :

1.


![](https://image.ibb.co/cD04dS/rnn_20timesteps_1lstmlayers.png)

2.


![](https://image.ibb.co/mKJqJS/rnn_20timesteps_4lstmlayers.png)

3.


![](https://image.ibb.co/ikZmsn/rnn_60timesteps_1lstmlayers.png)

4.


![](https://image.ibb.co/cO2vk7/rnn_60timesteps_4lstmlayers.png)
