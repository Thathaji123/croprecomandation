# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib  # Used to save the model

# Load the dataset
dataset = pd.read_csv('Crop_recommendation.csv')

# Display dataset info
print("First few records of the dataset:")
print(dataset.head())

# Check for missing values
print("\nChecking for missing values:")
print(dataset.isnull().sum())

# Show dataset info
print("\nDataset Information:")
print(dataset.info())

# Display unique crop labels and their counts
print("\nUnique crop labels and their counts:")
print(dataset['label'].value_counts())

# Separate features (X) and labels (y)
X = dataset.drop('label', axis=1)  # Features: All columns except 'label'
y = dataset['label']               # Labels: The 'label' column

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200)  # Set max_iter to 200 to avoid convergence warnings

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Logistic Regression model: {accuracy * 100:.2f}%")

# Save the trained model to a file
model_filename = 'crop_recommendation_model.pkl'
joblib.dump(model, model_filename)

print(f"\nModel saved to {model_filename}")
