from flask import Flask, request, jsonify, send_from_directory
from joblib import load
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
#abhi

# Suppress specific sklearn warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# Create a Flask application
app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')

# Load the trained model and scaler only once when the server starts
model_file = r"C:\Users\AHAO\OneDrive - Capco\Desktop\Abhi\Kovai.co\Ko.co assign\MotionSense\static\rf_classifier.joblib"
scaler_file = r"C:\Users\AHAO\OneDrive - Capco\Desktop\Abhi\Kovai.co\Ko.co assign\MotionSense\static\scaler.joblib"
rf_classifier = load(model_file)
scaler = load(scaler_file)

# Activity Mapping
activity_mapping = {
    1: "Working at Computer",
    2: "Standing Up, Walking and Going up/down stairs",
    3: "Standing",
    4: "Walking",
    5: "Going Up/Down Stairs",
    6: "Walking and Talking with Someone",
    7: "Talking while Standing"
}

def predict_activity(x_acceleration, y_acceleration, z_acceleration, model, scaler):
    # Create a DataFrame with the same feature names as during training
    features = pd.DataFrame([[x_acceleration, y_acceleration, z_acceleration]],
                            columns=['x_acceleration', 'y_acceleration', 'z_acceleration'])

    # Scale the raw input data
    scaled_features = scaler.transform(features)

    # Use the model to predict the activity
    prediction = model.predict(scaled_features)

    # Convert NumPy int64 type to Python native int type for JSON serialization
    return int(prediction[0])

@app.route('/predict', methods=['POST'])
def make_prediction():
    # Get data from POST request
    data = request.get_json()
    x_accel = data['x_acceleration']
    y_accel = data['y_acceleration']
    z_accel = data['z_acceleration']

    # Make a prediction
    predicted_activity = predict_activity(x_accel, y_accel, z_accel, rf_classifier, scaler)
    activity_description = activity_mapping.get(predicted_activity, "Unknown Activity")

    return jsonify({'predicted_activity': predicted_activity, 'Activity': activity_description})

if __name__ == "__main__":
    app.run(debug=True)
