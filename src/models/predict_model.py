# Import accuracy score
from sklearn.metrics import mean_absolute_error
import pickle
import numpy as np

# Function to predict and evaluate

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_price(model, sample_input):
    prediction = model.predict(sample_input.reshape(1, -1))
    return prediction[0]

def evaluate_model(model, X_test_scaled, y_test):
    # Predict the price on the testing set
    y_pred = model.predict(X_test_scaled)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)

    return mae


