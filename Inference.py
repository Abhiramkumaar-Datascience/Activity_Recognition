from joblib import load
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress specific sklearn warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def predict_activity(x_acceleration, y_acceleration, z_acceleration, model, scaler):
    # Create a DataFrame with the same feature names as during training
    features = pd.DataFrame([[x_acceleration, y_acceleration, z_acceleration]],
                            columns=['x_acceleration', 'y_acceleration', 'z_acceleration'])

    # Scale the raw input data
    scaled_features = scaler.transform(features)

    # Use the model to predict the activity
    prediction = model.predict(scaled_features)
    return prediction[0]

def make_prediction(x_accel, y_accel, z_accel):
    predicted_activity = predict_activity(x_accel, y_accel, z_accel, rf_classifier, scaler)
    print(f"The predicted activity is: {predicted_activity}")

if __name__ == "__main__":
    # Load the trained model and scaler
    model_file = "C:/Users/AHAO/OneDrive - Capco/Desktop/Abhi/Kovai.co/Ko.co assign/MotionSense/rf_classifier.joblib"
    scaler_file = "C:/Users/AHAO/OneDrive - Capco/Desktop/Abhi/Kovai.co/Ko.co assign/Activity Recognition/Processed Data/scaler.joblib"
    rf_classifier = load(model_file)
    scaler = load(scaler_file)

    while True:
        # Get user input
        try:
            x_acceleration = float(input("Enter X acceleration: "))
            y_acceleration = float(input("Enter Y acceleration: "))
            z_acceleration = float(input("Enter Z acceleration: "))
            # Make prediction
            make_prediction(x_acceleration, y_acceleration, z_acceleration)
        except ValueError:
            print("Invalid input. Please enter numerical values.")
            continue

        # Ask the user if they want to continue or exit
        cont = input("Do you want to make another prediction? (yes/no): ").lower()
        if cont != 'yes':
            break
