import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names

# Print the first 5 rows of the original data using pandas
print("First 5 rows of the original data:")
df_original = pd.DataFrame(X, columns=feature_names)
print(df_original.head())

# Print the first 5 target values (median house prices)
print("\nFirst 5 target values (median house prices in 100,000 dollars):")
print(y[:5])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the first 5 rows of the test data using pandas
print("\nFirst 5 rows of the test data:")
df_test = pd.DataFrame(X_test, columns=feature_names)
print(df_test.head())

# Scale/normalize the training and testing data
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Print the first 5 rows of the normalized training data using pandas
print("\nFirst 5 rows of the normalized training data:")
df_train_norm = pd.DataFrame(X_train_norm, columns=feature_names)
print(df_train_norm.head())

# Create and fit the regression model
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_train_norm, y_train)

# Test the model using the test data
y_pred = sgdr.predict(X_test_norm)

# Print the first few predictions and actual values
print(f"\nFirst 5 Predictions (median house prices in 100,000 dollars):")
print(y_pred[:5])

print(f"First 5 Actual Values (median house prices in 100,000 dollars):")
print(y_test[:5])

# Convert predictions and actual values to more readable format
y_pred_dollars = y_pred * 100000  # Convert from 100,000 dollars to dollars
y_test_dollars = y_test * 100000  # Convert from 100,000 dollars to dollars

print(f"\nFirst 5 Predictions (median house prices in dollars):")
print(y_pred_dollars[:5])

print(f"First 5 Actual Values (median house prices in dollars):")
print(y_test_dollars[:5])
