import pandas as pd

# create dummy features
def create_dummy_vars(df):
    
    # store the processed dataset in data/processed
    df.to_csv('data/processed/Processed_final.csv', index=None)

    # Separate the input features and target variable
    X = df.drop('price', axis=1)
    y = df['price']

    return X, y