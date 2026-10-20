# Bangalore House Price Predictor

A machine learning project that predicts Bangalore house prices using a Random Forest model, served via a Flask REST API.

## Project Structure
```
├── housing.csv          # Dataset
├── main_old.py          # Model training script
├── main.py              # Flask inference API
├── housing_model.pkl    # Trained Random Forest model (not tracked)
├── housing_pipeline.pkl # Fitted preprocessing pipeline (not tracked)
├── input.csv            # Test set (generated after training)
└── predictions.csv      # Model predictions on test set
```

## ML Pipeline
- Stratified train/test split by income category
- Numerical features: median imputation + standard scaling
- Categorical features: one-hot encoding
- Model: `RandomForestRegressor`

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python main_old.py
```

### 3. Start the API server
```bash
python main.py
```

### 4. Make a prediction
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'
```

### Response
```json
{
  "predicted_price": 452600.0
}
```
