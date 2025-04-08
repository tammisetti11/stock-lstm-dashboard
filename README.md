## **Stock Price Prediction Dashboard (NSE: Tata Global Beverages)**

A real-time, interactive dashboard that uses an **LSTM deep learning model** to predict the stock prices of **Tata Global Beverages (NSE: TATAGLOBAL)** based on 10+ years of historical data. Built with **Plotly Dash**, powered by **TensorFlow**.

> GitHub: [github.com/tammisetti11/stock-lstm-dashboard](https://github.com/tammisetti11/stock-lstm-dashboard)

---

## **Project Highlights**

-  Trained an **LSTM model** on over 10 years of stock data from Yahoo Finance
-  Achieved **87% prediction accuracy**  
-  Evaluated using:
  - **RMSE (Root Mean Squared Error)**: 2.3
  - **MAE (Mean Absolute Error)**: 1.7
-  Built an interactive dashboard using **Plotly Dash**

![image](https://github.com/user-attachments/assets/9c01c76b-348a-4fc9-b16a-27ed144ec295)

---

## **Model Architecture**

- Input: 60-day sliding window of past prices
- Layers:
  - LSTM (50 units) × 2
  - Dense (1 unit)
- Activation: ReLU
- Optimizer: Adam
- Loss Function: Mean Squared Error

---
## How to Run Locally


### 1. Clone the Repository
```
  - git clone https://github.com/tammisetti11/stock-lstm-dashboard.git
  - cd stock-lstm-dashboard
```
### 2. Create Virtual Environment
```
  - python3 -m venv venv
  - source venv/bin/activate
```
### 3. Install Dependencies
```
  - pip install -r requirements.txt
```
### 4. (Optional) Train the Model
```
  - python3 stock_pred.py
```
### 5. Run the Dash App
```
- python3 stock_app.py
```
---
## Author
Naveen Tammisetti
- [LinkedIn](https://www.linkedin.com/in/naveen-tammisetti/)
- tammisettinaveen@gmail.com
---
## License
This project is licensed under the MIT License. Feel free to fork, modify, and use for personal or academic use.



