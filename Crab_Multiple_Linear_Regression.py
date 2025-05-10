import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as sklearnLR
import Linear_Regression as customLR # Custom implementation of Linear Regression.

# Load the cleaned Crab Age Prediction dataset.
crabs = pd.read_csv("Crab_Cleaned.csv")

# Display the correlation matrix to justify feature selection.
features = ["Length", "Diameter", "Height", "Weight", "Shucked Weight",
            "Viscera Weight", "Shell Weight", "Sex_F", "Sex_M", "Sex_I", "Age"]
print("~~~~~ Correlation Matrix ~~~~~")
print(crabs[features].corr())

# Extract the target variable ('Age') and selected features.
y = crabs["Age"]
X = crabs[["Shell Weight", "Diameter", "Height"]]

# Performing Scikit-learn Linear Regression.
sk_model = sklearnLR().fit(X, y)
print("\n~~~~~ Scikit Learn Linear Regression ~~~~~")
print(f"R^2 Score: {sk_model.score(X, y):.5f}")
print(f"RMSE Score: {np.sqrt(np.average((y - sk_model.predict(X))**2.0)):.5f}")

# Perform Custom Linear Regression.
custom_model = customLR.LinearRegression().fit(X, y)
print("\n~~~~~ Self Linear Regression ~~~~~")
print(f"R^2 Score: {custom_model.score(X, y):.5f}")
print(f"RMSE Score: {custom_model.RMSE(X, y):.5f}")
