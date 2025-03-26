import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Initialize Dash app
app = dash.Dash()
server = app.server

# Load and preprocess dataset
df_nse = pd.read_csv("NSE-TATA.csv").tail(600)
df_nse["Date"] = pd.to_datetime(df_nse["Date"], format="%Y-%m-%d")
df_nse.set_index("Date", inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df_nse[["Close"]])

data_split = int(len(data_scaled) * 0.8)  # 80% training, 20% validation
train, valid = data_scaled[:data_split], data_scaled[data_split:]

X_train, y_train = [], []
for i in range(60, len(train)):
    X_train.append(train[i-60:i, 0])
    y_train.append(train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Load trained model
model = load_model("lstm_stock_model.h5")

# Prepare test data
inputs = df_nse["Close"].values[-(len(valid) + 60):]
inputs = scaler.transform(inputs.reshape(-1, 1))
X_test = np.array([inputs[i-60:i, 0] for i in range(60, len(inputs))])
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Make predictions
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train_df, valid_df = df_nse[:data_split], df_nse[data_split:]
valid_df["Predictions"] = closing_price

# Define app layout
app.layout = html.Div([
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='NSE-TATAGLOBAL Stock Data', children=[
            html.Div([
                html.H2("Actual Closing Price", style={"textAlign": "center"}),
                dcc.Graph(id="Actual Data", figure={
                    "data": [go.Scatter(x=train_df.index, y=valid_df["Close"], mode='markers')],
                    "layout": go.Layout(title='Scatter Plot', xaxis={'title': 'Date'}, yaxis={'title': 'Closing Rate'})
                }),
                html.H2("LSTM Predicted Closing Price", style={"textAlign": "center"}),
                dcc.Graph(id="Predicted Data", figure={
                    "data": [go.Scatter(x=valid_df.index, y=valid_df["Predictions"], mode='markers')],
                    "layout": go.Layout(title='Scatter Plot', xaxis={'title': 'Date'}, yaxis={'title': 'Closing Rate'})
                })
            ])
        ]),
    ])
])

if __name__ == '__main__':
    app.run(debug=True)

