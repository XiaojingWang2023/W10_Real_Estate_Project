
import pandas as pd

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the dataset from the given path.
    Without additional preprocessing because final.csv is already cleaned.
    """
    
    # Import the data from 'final.csv'
    df = pd.read_csv(data_path)
     
    
    return df

