import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import os

print("Model training script started.")

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')
    print("Created 'data' directory.")

# Load the dataset
try:
    data = pd.read_csv('data/personality_prediction.csv')
    print("Dataset loaded successfully.")
    print("Dataset head:\n", data.head())
except FileNotFoundError:
    print("Error: 'data/personality_prediction.csv' not found.")
    print("Please make sure the CSV file is in the 'data' directory.")
    exit()

# Drop unnecessary columns if they exist
columns_to_drop = ['Gender', 'Age']
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
print("Dropped unnecessary columns.")

# Encode the target variable 'Personality'
le = LabelEncoder()
data['Personality'] = le.fit_transform(data['Personality'])
print("Target variable 'Personality' encoded.")
print("Encoded classes:", list(le.classes_))

# Define features (X) and target (y)
X = data.drop('Personality', axis=1)
y = data['Personality']

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")
print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
print("Training RandomForestClassifier model...")
model.fit(X_train, y_train)
print("Model training completed.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
if hasattr(model, 'oob_score_'):
    print(f"Out-of-Bag Score: {model.oob_score_ * 100:.2f}%")


# Save the trained model and the label encoder to a file
model_payload = {
    'model': model,
    'label_encoder': le
}

with open('personality_prediction.pkl', 'wb') as file:
    pickle.dump(model_payload, file)

print("Trained model and label encoder saved to 'personality_prediction.pkl'.")
print("Model training script finished.")
