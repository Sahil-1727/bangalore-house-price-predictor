import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import Pipeline
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import os
import joblib

# MODEL_FILE = "housing_model.pkl"
# PIPELINE_FILE = "housing_pipeline.pkl"
#
# def build_pipeline(num_attributes, cat_attributes):
#
#     #5. Create a pipeline
#     #  for numerical attributes
#     num_pipeline = Pipeline([
#         ('imputer', SimpleImputer(strategy="median")),
#         ('std_scaler', StandardScaler()),
#     ])
#     # for categorical attributes
#     cat_pipeline = Pipeline([
#         ('one_hot', OneHotEncoder(handle_unknown='ignore'))
#     ])
#
#     #Combine the pipelines using ColumnTransformer
#     full_pipeline = ColumnTransformer([
#         ("num", num_pipeline, num_attributes),
#         ("cat", cat_pipeline, cat_attributes)
#     ])
#
#     return full_pipeline
#
# if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
#     #1.  Load data
#     housing = pd.read_csv("housing.csv")
#
#     #2. Create a stratified test set
#     housing["income_cat"] = pd.cut(housing["median_income"],
#                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
#
#     split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#
#     for train_index, test_index in split.split(housing, housing["income_cat"]):
#         housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index=False)  # Save test set for inference
#         housing = housing.loc[train_index].drop("income_cat", axis=1)
#
#
#
#     #3. Seprate the Lable and Features
#     housing_labels = housing["median_house_value"].copy()
#     housing_features = housing.drop("median_house_value", axis=1)
#
#         #4. list the numerical and categorical columns
#     num_attributes = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
#     cat_attributes = ["ocean_proximity"]
#
#     pipeline = build_pipeline(num_attributes, cat_attributes)
#
#     #6. Apply the pipeline to the housing data
#     housing_prepared = pipeline.fit_transform(housing_features)
#
#     #7. Train the model
#     model = RandomForestRegressor(random_state=42)
#     model.fit(housing_prepared, housing_labels)
#
#     #8. Save the model and pipeline
#     joblib.dump(model, MODEL_FILE)
#     joblib.dump(pipeline, PIPELINE_FILE)
#     print("Model and pipeline saved successfully.")
# else:
#     # Load the model and pipeline [Inference mode]
#     model = joblib.load(MODEL_FILE)
#     pipeline = joblib.load(PIPELINE_FILE)
#     print("Model and pipeline loaded successfully.")
#
#     input_data = pd.read_csv("input.csv")
#     transformed_input = pipeline.transform(input_data)
#     predictions = model.predict(transformed_input)
#     input_data["median_house_value"] = predictions
#
#     input_data.to_csv("predictions.csv", index=False)
#     print("Predictions saved to predictions.csv.")
#     print("results saved successfully.")



app = Flask(__name__)

# Load BOTH the pipeline (for formatting) and the model (for predicting) using joblib
pipeline = joblib.load("housing_pipeline.pkl")
model_rf = joblib.load("housing_model.pkl")


@app.route("/predict", methods=["POST"])
def predict_price():
    try:
        data = request.get_json()

        # Build a pandas DataFrame using the exact columns
        input_data = pd.DataFrame([{
            "longitude": data["longitude"],
            "latitude": data["latitude"],
            "housing_median_age": data["housing_median_age"],
            "total_rooms": data["total_rooms"],
            "total_bedrooms": data["total_bedrooms"],
            "population": data["population"],
            "households": data["households"],
            "median_income": data["median_income"],
            "ocean_proximity": data["ocean_proximity"]
        }])

        # 1. Transform the input data using the pipeline
        transformed_input = pipeline.transform(input_data)

        # 2. Predict the price using the Random Forest model
        prediction = model_rf.predict(transformed_input)

        return jsonify({
            "predicted_price": float(prediction[0])
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Start the Flask server
    app.run(debug=True)