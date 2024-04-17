import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Assuming you have already downloaded the dataset as a CSV file named 'air_quality_data.csv'
data = pd.read_csv('/home/manic/PycharmProjects/NewPython/Application of ML/Air Quality/city_day.csv')


# Handle missing values, outliers, and inconsistencies
data.dropna(inplace=True)  # Drop rows with missing values
# Handle outliers if necessary


# Descriptive statistics
print(data.describe())
# Visualizations
sns.pairplot(data)
plt.show()


# Assuming features like pollutant levels, weather conditions, and geographical information are present
# Here, we're considering PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene as features


# No new features added in this example

# Step 6: Split the Dataset
X = data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']]  # Features
y = data['AQI']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Assuming Linear Regression and Random Forest Regression
linear_reg_model = LinearRegression()
rf_model = RandomForestRegressor()

#Train the Models
linear_reg_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

#Evaluate Model Performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)


print("Linear Regression Model Performance:")
evaluate_model(linear_reg_model, X_test, y_test)
print("\nRandom Forest Model Performance:")
evaluate_model(rf_model, X_test, y_test)

# Step 10: Visualize Predictions
plt.figure(figsize=(12, 6))

# Scatter plot for Linear Regression model
plt.subplot(1, 2, 1)
plt.scatter(y_test, linear_reg_model.predict(X_test), color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2, label='Actual')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Linear Regression: Actual vs. Predicted AQI")
plt.legend()

# Scatter plot for Random Forest model
plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_model.predict(X_test), color='green', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2, label='Actual')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Random Forest: Actual vs. Predicted AQI")
plt.legend()
plt.tight_layout()
plt.show()

# Step 11: Interpret Trained Models
# Coefficients for Linear Regression model
print("Linear Regression Coefficients:")
print(linear_reg_model.coef_)


#Feature importance for Random Forest model
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Random Forest: Feature Importances")
plt.show()
