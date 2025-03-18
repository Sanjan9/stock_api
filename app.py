from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import yfinance as yf
import os

app = Flask(__name__)

#  Load Model & Scaler
model_path = "stock_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("ERROR: Model or Scaler file is missing!")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("Random Forest Model and Scaler Loaded Successfully!")

#  Serve index.html Directly (Without `templates/`)
@app.route("/")
def home():
    return send_file("index.html")

# Predict API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        stock_symbol = data.get("symbol", "").upper()

        if not stock_symbol:
            return jsonify({"error": "Stock symbol is required!"}), 400

        # Fetch Latest Stock Data
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="5d")

        if hist.empty:
            return jsonify({"error": f"No stock history data for {stock_symbol}!"})

        latest_data = hist.iloc[-1]

        #  Use Consistent Features
        features = np.array([
            latest_data["Open"], latest_data["High"], latest_data["Low"],
            latest_data["Close"], latest_data["Volume"], latest_data["Close"] - latest_data["Open"],
            hist["Close"].rolling(10).mean().iloc[-1]
        ]).reshape(1, -1)

        # Scale Features
        features_scaled = scaler.transform(features)
        predicted_price = round(float(model.predict(features_scaled)[0]), 2)

        return jsonify({
            "symbol": stock_symbol,
            "today_price": round(float(latest_data["Close"]), 2),
            "predicted_price": predicted_price
        })

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": f"Internal error: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
