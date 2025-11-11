🌾 Crop Price Prediction using Machine Learning

This project predicts daily crop prices in India using machine learning (Random Forest Regression) and real-time data fetched from Google Search via SerpAPI.
It helps farmers, traders, and researchers estimate the next day's crop prices for better planning and decision-making.

🧠 Features

✅ Fetches real-time crop prices from Google using SerpAPI
✅ Predicts next-day crop price using a trained Random Forest Regressor
✅ Uses time-based features (date as timestamp) for prediction
✅ Calculates key model performance metrics (MAE, MSE, RMSE, R²)
✅ Works for multiple crops (Rice, Wheat, Corn, Pulses, etc.)

⚙️ Tech Stack

Python 3.8+

NumPy, Pandas – Data handling

Scikit-learn – Machine learning model

Requests – API integration

SerpAPI – Real-time price data from Google

🗂️ Project Structure
crop-price-prediction/
│
├── crop_price_prediction.py    # Main script
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies (optional)

🔑 Setup Instructions
1. Clone the Repository
git clone https://github.com/Rem0tecoder/crop-price-prediction.git
cd crop-price-prediction

2. Install Dependencies
pip install numpy pandas scikit-learn requests

3. Add Your SerpAPI Key

Replace the placeholder API key in the script:

API_KEY = "your_serpapi_key_here"


👉 You can get your free API key from SerpAPI
.

🚀 How to Run
python crop_price_prediction.py


When prompted, enter the name of the crop you want to predict, e.g.:

Enter the crop name for prediction (e.g., Rice, Wheat, Corn, etc.): Rice


Output example:

--- Rice Price Prediction ---
Mean Absolute Error: 1.25
Mean Squared Error: 2.58
Root Mean Squared Error: 1.60
R-squared Score: 0.94
Predicted Rice Price for 2025-11-12: ₹57.23 per kg
Predicted Rice Price for 2025-11-12 (Integer): ₹57 per kg

📊 How It Works

Data Collection:
Uses SerpAPI to fetch recent crop prices from Google search.

Data Preparation:
Builds a 30-day dataset with daily prices for selected crops.

Feature Engineering:
Converts date to numerical timestamp (ordinal).

Model Training:
Trains a Random Forest Regressor to learn price trends.

Prediction:
Predicts the next day’s price for the chosen crop.

🌾 Supported Crops

Rice

Wheat

Corn

Moong Dal

Arahar Dal

Mustard

Sugar Cane

Mango

Dragon Fruit

Tea

📈 Evaluation Metrics

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

R² (Coefficient of Determination)

These metrics help assess model accuracy and performance.

🧩 Example Use Cases

Farmers predicting tomorrow’s price to decide when to sell crops.

Market analysts tracking commodity trends.

Educational demos of machine learning in agriculture.

🛡️ Disclaimer

This tool provides approximate predictions based on limited data and trends.
For actual market prices, please refer to government or mandi sources.

📬 Author

Saurabh Yadav
🔗 https://github.com/Rem0tecoder

📧 eryadav001@gmail.com
