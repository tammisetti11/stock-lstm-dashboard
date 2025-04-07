from flask import Flask, jsonify
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and data once at startup
model = load_model("lstm_stock_model.h5")

# Load and preprocess dataset
df = pd.read_csv("NSE-TATA.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[["Close"]])

# Prepare test input (last 20% + 60 time steps)
test_window = int(len(scaled_data) * 0.2)
inputs = df["Close"].values[-(test_window + 60):]
inputs = scaler.transform(inputs.reshape(-1, 1))

X_test = [inputs[i-60:i, 0] for i in range(60, len(inputs))]
X_test = np.array(X_test).reshape(len(X_test), 60, 1)

@app.route("/")
def home():
    return "Flask API is running. Visit /predict for LSTM output."

@app.route("/predict", methods=["GET"])
def predict():
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Trim to last 100 predictions
    predictions = predictions[-100:]
    dates = df.index[-len(predictions):].strftime("%Y-%m-%d").tolist()
    prices = predictions.flatten().tolist()

    return jsonify({
        "dates": dates,
        "prices": prices
    })

if __name__ == "__main__":
    app.run()
