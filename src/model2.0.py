import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('../dataset/mentalhealth-data.csv')

# Add global mental health prevalence as new features
global_anxiety_avg = 4.10
global_depression_avg = 3.77
data['global_anxiety_avg'] = global_anxiety_avg
data['global_depression_avg'] = global_depression_avg

# Define Anxiety & PTSD labels
def classify_condition(row):
    if row['global_anxiety_avg'] > 4.5 and row['global_depression_avg'] > 4.0:
        return 'PTSD'
    elif row['global_anxiety_avg'] > 4.0 and row['global_depression_avg'] < 4.0:
        return 'Anxiety'
    else:
        return 'No Issue'

# Apply classification
data['mental_health_condition'] = data.apply(classify_condition, axis=1)

# Encode target labels
data['mental_health_condition'] = data['mental_health_condition'].map({'No Issue': 0, 'Anxiety': 1, 'PTSD': 2})

# Define feature columns and target variable
feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere', 'global_anxiety_avg', 'global_depression_avg']
X = data[feature_cols]
y = data['mental_health_condition']

# Handle class imbalance if multiple classes exist
if len(np.unique(y)) > 1:
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

# Compute class weights if multiple classes exist
if len(np.unique(y)) > 1:
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weight_dict = {i: class_weights[i] for i in np.unique(y)}
else:
    class_weight_dict = None

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train a multi-class RandomForestClassifier with class weights
model = RandomForestClassifier(n_estimators=100, random_state=0, class_weight=class_weight_dict)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# Save the updated model
joblib.dump(model, "../models/_random_forest_model_multiclass.pkl")

print("Updated multi-class model trained and saved successfully!")
