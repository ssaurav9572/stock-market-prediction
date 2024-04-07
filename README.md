# Stock Price Prediction using LSTM

The provided Python code demonstrates how to use Long Short-Term Memory (LSTM) neural networks to predict stock prices based on historical data. Here's an overview of the key components and functionalities of the code:

# Key Features:

# Loading Historical Stock Price Data:

Historical stock price data is loaded from a CSV file ('TATAMOTORS.csv' in this case). The 'Close' column from the dataset is extracted, representing the closing prices of the stock.

# Data Preprocessing:

The closing prices are normalized using Min-Max scaling to bring them within the range of 0 to 1, facilitating better convergence during training.

# Creating Sequences for LSTM Training:

The data is divided into sequences of a specified length (sequence_length) along with corresponding target values. This prepares the data for training the LSTM model, where each input sequence is associated with a target value.

# Splitting Data into Train and Test Sets:

The dataset is split into training and testing sets, with 80% of the data used for training and the remaining 20% for testing.

# Building the LSTM Model:

The LSTM model architecture is defined using the Sequential API from Keras. The model consists of multiple LSTM layers followed by a dense output layer for regression. The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss function.

# Training the Model:

The LSTM model is trained on the training data using the fit() method. The number of epochs and batch size are specified for training.

# Evaluating the Model:

The trained model is evaluated on the test data to assess its performance. The Mean Squared Error (MSE) on the test data is calculated and printed to evaluate the model's accuracy.

# Making Predictions:

The trained LSTM model is used to make predictions on the test data.

# Visualizing Results:

The predicted stock prices are inverse transformed to the original scale using Min-Max scaling. Both the predicted and actual stock prices are visualized using Matplotlib to compare the model's predictions with the ground truth.
