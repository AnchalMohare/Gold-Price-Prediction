import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load Dataset
df = pd.read_csv('data/gld_price_data.csv')

# Display first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Check missing values
print(df.isnull().sum())

# Correlation Heatmap
correlation = df.corr(numeric_only=True)

plt.figure(figsize=(8,8))

sns.heatmap(
    correlation,
    cbar=True,
    square=True,
    fmt='.1f',
    annot=True,
    annot_kws={'size':8},
    cmap='Blues'
)

plt.title("Correlation Heatmap")
plt.show()

# Gold Price Trend
plt.figure(figsize=(10,5))

plt.plot(df['GLD'])

plt.title('Gold Price Trend')
plt.xlabel('Days')
plt.ylabel('Gold Price')

plt.show()

# Features and Target
X = df.drop(['Date', 'GLD'], axis=1)

Y = df['GLD']

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2
)

# Create Model
regressor = RandomForestRegressor(n_estimators=100)

# Train Model
regressor.fit(X_train, Y_train)

# Prediction
test_data_prediction = regressor.predict(X_test)

# Evaluation

r2 = r2_score(Y_test, test_data_prediction)

mae = mean_absolute_error(Y_test, test_data_prediction)

print("R2 Score:", r2)

print("MAE:", mae)

# Compare Actual vs Predicted

Y_test = list(Y_test)

plt.figure(figsize=(10,5))

plt.plot(Y_test, color='blue', label='Actual Value')

plt.plot(test_data_prediction, color='green', label='Predicted Value')

plt.title('Actual Price vs Predicted Price')

plt.xlabel('Number of Values')

plt.ylabel('Gold Price')

plt.legend()

plt.show()

# Save Model

import pickle

filename = 'model/gold_price_model.sav'

pickle.dump(regressor, open(filename, 'wb'))

print("Model saved successfully!")