from flask import Flask, request, jsonify
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

app = Flask(__name__)
model = load_model("lstm_stock_model.h5")

# Load and scale the dataset
df = pd.read_csv("NSE-TATA.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[["Close"]])
input_data = df["Close"].values[-(60 + int(len(scaled_data) * 0.2)):]
input_data = scaler.transform(input_data.reshape(-1, 1))

@app.route("/predict", methods=["GET"])
def predict():
    X_test = [input_data[i-60:i, 0] for i in range(60, len(input_data))]
    X_test = np.array(X_test).reshape(len(X_test), 60, 1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    dates = df.index[-len(predictions):].strftime('%Y-%m-%d').tolist()
    prices = predictions.flatten().tolist()

    return jsonify({"dates": dates, "prices": prices})

if __name__ == "__main__":
    app.run(debug=True)
