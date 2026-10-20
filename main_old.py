import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer    
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


#1.  Load data
housing = pd.read_csv("housing.csv")

#2. Create a stratified test set
housing["income_cat"] = pd.cut(housing["median_income"], 
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

#copy the train data to a new variable
housing = strat_train_set.copy()

#3. Seprate the Lable and Features
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

print(housing , housing_labels)

#4. list the numerical and categorical columns
num_attributes = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attributes = ["ocean_proximity"]

#5. Create a pipeline
#  for numerical attributes
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
# for categorical attributes
cat_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

#Combine the pipelines using ColumnTransformer
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes)
])

#6. Apply the pipeline to the housing data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)

#7. Train the models
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
# print("Linear Regression RMSE:", lin_rmse)
lin_rmse = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# print("Linear Regression RMSE:", lin_rmses)
print(pd.Series(lin_rmse).describe())

#Decision Tree Regressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
tree_preds = tree_reg.predict(housing_prepared)
# tree_rmse = root_mean_squared_error(housing_labels, tree_preds)
tree_rmse = -cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# print("Decision Tree RMSE:", tree_rmses)
print(pd.Series(tree_rmse).describe())

#Random Forest Regressor
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)                
forest_preds = forest_reg.predict(housing_prepared)
# forest_rmse = root_mean_squared_error(housing_labels, forest_preds)
# print("Random Forest RMSE:", forest_rmse)
forest_rmse = -cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# print("Random Forest RMSE:", forest_rmses)
print(pd.Series(forest_rmse).describe())