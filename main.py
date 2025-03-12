# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Swapnil Kangralkar',
#     license='',
# )

from src.data.make_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_tree
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_model
from src.models.predict_model import evaluate_model
from src.models.predict_model import predict_price
from src.models.predict_model import load_model
import numpy as np

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/final.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    X, y = create_dummy_vars(df)

    # Train the Random Forest regression model
    rfmodel, X_test, y_test = train_model(X, y)
    
    
    # Evaluate the model
    mae = evaluate_model(rfmodel, X_test, y_test)
    print(f"Mean Absolute Error: {mae}")

    # Plot visualizations
    tree_image_path = plot_tree(rfmodel, X.columns.tolist())
    
    # Use the loaded pickled model to make predictions
    sample_input = np.array([2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0, 0])
    model = load_model('models/RFmodel.pkl')
    prediction = predict_price(model, sample_input)
    print(f"Predicted Price for Sample Input: {prediction}")
