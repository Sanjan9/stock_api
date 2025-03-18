import yfinance as yf
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#  Fetch Historical Stock Data
def get_stock_data(symbol, period="1y"):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)

    if hist.empty:
        print(f"⚠️ No data found for {symbol}")
        return None

    hist["Next_Day_Close"] = hist["Close"].shift(-1)  # Target Variable
    hist["Daily_Change"] = hist["Close"] - hist["Open"]
    hist["MA_10"] = hist["Close"].rolling(10).mean()

    hist = hist.dropna()
    return hist[["Open", "High", "Low", "Close", "Volume", "Daily_Change", "MA_10", "Next_Day_Close"]]

#  List of Stocks for Training (Only 50)
tickers = [
    "AAPL", "TSLA", "GOOGL", "AMZN", "MSFT", "NVDA", "META", "NFLX", "AMD", "IBM",
    "INTC", "ORCL", "CSCO", "PYPL", "QCOM", "TXN", "ADBE", "AVGO", "CRM", "BA",
    "NKE", "MCD", "V", "MA", "DIS", "JPM", "GS", "MS", "BAC", "C", "WMT",
    "HD", "LOW", "TGT", "COST", "XOM", "CVX", "PEP", "KO", "ABBV", "JNJ",
    "PFE", "UNH", "TMO", "LIN", "DHR", "BMY", "RTX", "LMT"
]

all_data = [get_stock_data(symbol) for symbol in tickers if get_stock_data(symbol) is not None]

if len(all_data) == 0:
    print("❌ No valid stock data found! Exiting.")
    exit()

df = pd.concat(all_data).dropna()

#  Features & Target
features = df.drop(columns=["Next_Day_Close"])
target = df["Next_Day_Close"]

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

#  Save Model & Scaler
joblib.dump(model, "stock_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model Trained with Random Forest & Saved!")
