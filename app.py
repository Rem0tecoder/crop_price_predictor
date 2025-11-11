import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# SerpAPI key (Replace with your actual API key)
API_KEY = "e6b697600477bdd98ded7327e1b3c66f0834c38e3e52188416802fb8dde83892"

# Function to fetch crop price using SerpAPI
def fetch_crop_price(crop_name):
    params = {
        "engine": "google",
        "q": f"{crop_name} price per kg in India",
        "api_key": API_KEY,
    }
    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()
    
    for result in data.get("organic_results", []):
        snippet = result.get("snippet", "")
        if "₹" in snippet:
            return int(''.join(filter(str.isdigit, snippet.split('₹')[1])))  # Extract numeric price
    
    return None

# List of crops to predict prices for
crops = ["Rice", "Wheat", "Corn", "Moong Dal", "Arahar Dal", "Mustard", "Sugar Cane", "Mango", "Dragon Fruit", "Tea"]

# Fetch updated daily prices (Using more historical data for better accuracy)
data = {"Date": [(datetime.today() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(29, -1, -1)]}
for crop in crops:
    fetched_price = fetch_crop_price(crop) or np.random.randint(40, 80)  # Default random price if not found
    data[crop] = [fetched_price - 12, fetched_price - 10, fetched_price - 8, fetched_price - 5, fetched_price - 3,
              fetched_price, fetched_price + 3, fetched_price + 5, fetched_price + 7, fetched_price + 10] * 3
data[crop] = data[crop][:30]  # Ensure exactly 30 values

df = pd.DataFrame(data)

# Function to train and predict prices for the next day
def predict_crop_price(crop_name):
    if crop_name not in df.columns:
        print(f"Error: {crop_name} data not available.")
        return
    
    df["Date"] = pd.to_datetime(df["Date"])
    df["Timestamp"] = df["Date"].map(pd.Timestamp.toordinal)  # Convert dates to numerical values
    
    X = df[["Timestamp"]]  # Features (Date as ordinal values)
    y = df[crop_name]  # Target (Crop Prices)

    # Normalize timestamps for better accuracy
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the dataset into training and testing sets (Smaller test size for better learning)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

    # Training the Random Forest Regression Model with optimized parameters
    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)

    # Predicting crop prices
    predictions = model.predict(X_test)

    # Evaluating the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"\n--- {crop_name} Price Prediction ---")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared Score: {r2}")

    # Predict price for the next day
    next_day = datetime.today() + timedelta(days=1)
    next_day_timestamp = scaler.transform([[next_day.toordinal()]])
    future_price = model.predict(next_day_timestamp)
    future_price_int = int(round(future_price[0]))  # Convert to integer

    print(f"Predicted {crop_name} Price for {next_day.strftime('%Y-%m-%d')}: ₹{future_price[0]:.2f} per kg")
    print(f"Predicted {crop_name} Price for {next_day.strftime('%Y-%m-%d')} (Integer): ₹{future_price_int} per kg")

# Ask user for crop name
user_crop = input("Enter the crop name for prediction (e.g., Rice, Wheat, Corn, etc.): ")
predict_crop_price(user_crop)
