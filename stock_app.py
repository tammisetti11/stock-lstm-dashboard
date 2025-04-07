import dash
from dash import dcc, html
import requests
import plotly.graph_objs as go
import pandas as pd

app = dash.Dash()
server = app.server

# Load local CSV just to display historical data
df = pd.read_csv("NSE-TATA.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df.sort_index()

# Call the API to fetch predictions
def fetch_predictions():
    try:
        res = requests.get("https://your-flask-api-url/predict")
        data = res.json()
        dates = pd.to_datetime(data["dates"])
        prices = data["prices"]
        return dates, prices
    except:
        return [], []

pred_dates, pred_prices = fetch_predictions()

# Layout
app.layout = html.Div([
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Actual Stock Data', children=[
            html.Div([
                dcc.Graph(
                    id="actual-graph",
                    figure={
                        "data": [go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Actual")],
                        "layout": go.Layout(title="Actual Closing Prices", xaxis={"title": "Date"}, yaxis={"title": "Price"})
                    }
                )
            ])
        ]),
        dcc.Tab(label='Predicted Stock Data', children=[
            html.Div([
                dcc.Graph(
                    id="predicted-graph",
                    figure={
                        "data": [go.Scatter(x=pred_dates, y=pred_prices, mode="lines", name="Predicted")],
                        "layout": go.Layout(title="Predicted Prices", xaxis={"title": "Date"}, yaxis={"title": "Price"})
                    }
                )
            ])
        ])
    ])
])

if __name__ == "__main__":
    app.run(debug=True)
