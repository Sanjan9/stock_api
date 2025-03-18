import joblib
import numpy as np
import yfinance as yf

# Load Model & Scaler
model = joblib.load("stock_model.pkl")
scaler = joblib.load("scaler.pkl")

stock_symbols = ["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT"]

def predict_stock_price(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="5d")

    if hist.empty:
        print(f"⚠️ No data found for {stock_symbol}")
        return

    latest_data = hist.iloc[-1]
    
    features = np.array([
        latest_data["Open"], latest_data["High"], latest_data["Low"],
        latest_data["Close"], latest_data["Volume"], latest_data["Close"] - latest_data["Open"],
        hist["Close"].rolling(10).mean().iloc[-1]
    ]).reshape(1, -1)

    features_scaled = scaler.transform(features)
    predicted_price = round(float(model.predict(features_scaled)[0]), 2)

    print(f"{stock_symbol} - Real: ${latest_data['Close']} | Predicted: ${predicted_price}")

for stock in stock_symbols:
    predict_stock_price(stock)
