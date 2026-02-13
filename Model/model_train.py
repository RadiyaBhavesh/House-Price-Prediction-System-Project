import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# ================= LOAD DATA =================
data = pd.read_csv("../dataset/gujarat_house_price_.csv")
print(" Dataset Loaded :", data.shape)

# ================= CLEANING =================
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

data.rename(columns={
    "location": "Location",
    "area_sqft": "Area",
    "bhk": "BHK",
    "bath": "Bathroom",
    "price": "Price"
}, inplace=True)

# ================= OUTLIER REMOVAL =================
Q1 = data["Price"].quantile(0.25)
Q3 = data["Price"].quantile(0.75)
IQR = Q3 - Q1

data = data[
    (data["Price"] >= Q1 - 1.5 * IQR) &
    (data["Price"] <= Q3 + 1.5 * IQR)
]

print(" After Cleaning :", data.shape)

# ================= ENCODING =================
le = LabelEncoder()
data["Location"] = le.fit_transform(data["Location"])

# Log Transform Target
data["Price"] = np.log1p(data["Price"])

# ================= FEATURES & TARGET =================
X = data[["Area", "BHK", "Bathroom", "Location"]]
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODELS =================

# Linear Regression
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])
lr_pipeline.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=22,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeRegressor(
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
dt_model.fit(X_train, y_train)

# Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
gb_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# ================= PREDICTIONS =================
lr_pred = lr_pipeline.predict(X_test)
rf_pred = rf_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# Hybrid Weighted Prediction
hybrid_pred = (
    0.2 * lr_pred +
    0.25 * rf_pred +
    0.15 * dt_pred +
    0.2 * gb_pred +
    0.2 * xgb_pred
)

# Reverse Log Transform
y_test_real = np.expm1(y_test)
hybrid_real = np.expm1(hybrid_pred)

# ================= EVALUATION =================
r2 = r2_score(y_test, hybrid_pred)
mae = mean_absolute_error(y_test_real, hybrid_real)
rmse = np.sqrt(mean_squared_error(y_test_real, hybrid_real))

print("\n HYBRID MODEL PERFORMANCE")
print("R2 Score        :", round(r2, 3))
print("MAE (₹)         :", round(mae, 2))
print("RMSE (₹)        :", round(rmse, 2))

# Cross Validation
cv_scores = cross_val_score(lr_pipeline, X, y, cv=5, scoring="r2")
print("CV R2 Avg       :", round(cv_scores.mean(), 3))

# ================= SAVE MODELS =================
pickle.dump(lr_pipeline, open("linear_model.pkl", "wb"))
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(dt_model, open("dt_model.pkl", "wb"))
pickle.dump(gb_model, open("gb_model.pkl", "wb"))
pickle.dump(xgb_model, open("xgb_model.pkl", "wb"))
pickle.dump(le, open("location_encoder.pkl", "wb"))

print("\n  models & encoder saved successfully")
