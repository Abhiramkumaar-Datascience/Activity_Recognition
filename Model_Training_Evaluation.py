# Import Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from joblib import dump

# Load the preprocessed data
data_path = r"C:\Users\AHAO\OneDrive - Capco\Desktop\Abhi\Kovai.co\Ko.co assign\Activity Recognition\Processed Data\processed_data.csv"
data = pd.read_csv(data_path)

# Define features and labels
X = data[['x_acceleration', 'y_acceleration', 'z_acceleration']]
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)
best_params = grid_search.best_params_

# Train the RandomForestClassifier using the best parameters
rf_classifier = RandomForestClassifier(**best_params, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train_resampled, y_train_resampled)

# Evaluate the model's performance
y_pred = rf_classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model to a file for future use
model_file = r"C:\Users\AHAO\OneDrive - Capco\Desktop\Abhi\Kovai.co\Ko.co assign\rf_classifier.joblib"
dump(rf_classifier, model_file)
