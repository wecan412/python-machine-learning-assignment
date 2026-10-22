import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
# NOTE: Replace 'student_scores.csv' with your actual file name. 
try:
    df = pd.read_csv('student_scores.csv')
except FileNotFoundError:
    print("Error: Dataset not found. Please ensure 'student_scores.csv' is in the same directory.")
    exit()

# Display first few rows
print("Dataset Preview:")
print(df.head())

# 2. Prepare the data (Assume first column is feature X, second is target y)
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Target

# 3. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("\n--- Model Evaluation ---")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")

# 7. Visualize the results
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Supervised Learning: Predicting Exam Scores')
plt.xlabel('Input Feature (e.g., Hours Studied)')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True)
plt.show()
