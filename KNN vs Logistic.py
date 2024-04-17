import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the dataset
data = pd.read_csv('house data.csv')

# Drop non-numerical columns
data = data.drop(['date', 'street', 'city', 'statezip', 'country'], axis=1)

# Separate features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Predict on the test set using KNN
knn_predictions = knn_model.predict(X_test_scaled)

# Calculate Mean Squared Error for KNN
knn_mse = mean_squared_error(y_test, knn_predictions)
print("KNN Mean Squared Error:", knn_mse)

# Convert y_train to binary classes for Logistic Regression comparison
median_price = y_train.median()
y_train_logreg = (y_train > median_price).astype(int)
y_test_logreg = (y_test > median_price).astype(int)

# Initialize and fit Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_scaled, y_train_logreg)

# Predict on the test set using Logistic Regression
logreg_predictions = logreg_model.predict(X_test_scaled)

# Calculate accuracy score for Logistic Regression
logreg_accuracy = accuracy_score(y_test_logreg, logreg_predictions)

# Display which model performs better
if knn_mse < logreg_accuracy:
    print("KNN performs better (lower MSE):")
    print("KNN Mean Squared Error:", knn_mse)
elif logreg_accuracy < knn_mse:
    print("Logistic Regression performs better (higher accuracy):")
    print("Logistic Regression Accuracy Score:", logreg_accuracy)
else:
    print("Both models perform equally well.")
