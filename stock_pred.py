import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load and preprocess the dataset
dataframe = pd.read_csv("NSE-TATA.csv")
dataframe["Date"] = pd.to_datetime(dataframe["Date"], format="%Y-%m-%d")
dataframe.set_index("Date", inplace=True)

# Visualize closing price
plt.figure(figsize=(16, 8))
plt.plot(dataframe["Close"], label="Close Price History")
plt.legend()
plt.show()

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(dataframe[["Close"]])

data_split = int(len(data_scaled) * 0.8)  # 80% training, 20% validation
train_data, valid_data = data_scaled[:data_split], data_scaled[data_split:]

X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=50),
    Dense(1)
])

model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=2)

# Prepare test data
inputs = dataframe["Close"].values[-(len(valid_data) + 60):]
inputs = scaler.transform(inputs.reshape(-1, 1))
X_test = [inputs[i-60:i, 0] for i in range(60, len(inputs))]
X_test = np.array(X_test).reshape(len(X_test), 60, 1)

# Make predictions
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Save model
model.save("lstm_stock_model.h5")

# Plot results
train_df, valid_df = dataframe[:data_split], dataframe[data_split:]
valid_df["Predictions"] = predicted_prices
plt.figure(figsize=(16, 8))
plt.plot(train_df["Close"], label="Training Data")
plt.plot(valid_df[["Close", "Predictions"]], label=["Actual Price", "Predicted Price"])
plt.legend()
plt.show()
