
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('train.csv', header=None)
data.columns = ['class', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5']

# Separate the independent variables (features) and the target variable (class)
X = data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']]
y = data['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Example prediction (you can change the values as needed)
example_data = [[107, 10.1, 2.2, 0.9, 2.7]]
predicted_class = model.predict(example_data)
print(f"Predicted class for example data: {predicted_class[0]}")