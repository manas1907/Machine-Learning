import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("winequality-red.csv")

# Adjusting target variable to start from 0
data['quality'] -= 3

# Separate features and target variable
X = data.drop('quality', axis=1)
y = data['quality']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Counting predictions
good_predictions = (y_pred >= 3).sum()  # Counting predictions equal to or above 3 (considered "good")
bad_predictions = len(y_pred) - good_predictions  # Counting predictions below 3 (considered "bad")

# Determining majority class
if good_predictions > bad_predictions:
    print("Overall prediction: good")
else:
    print("Overall prediction: bad")
