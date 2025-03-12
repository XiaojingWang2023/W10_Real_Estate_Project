
import pandas as pd

def load_and_preprocess_data(data_path):
    
    # Import the data from 'final.csv'
    df = pd.read_csv(data_path)


    # Impute all missing values in all the features
    df['property_tax'].fillna(df['property_tax'].median(), inplace=True)
    df['insurance'].fillna(df['insurance'].median(), inplace=True)
    df['basement'].fillna(0, inplace=True)  
    df['popular'].fillna(0, inplace=True)     
    df['recession'].fillna(0, inplace=True)   
    df['year_built'].fillna(df['year_built'].median(), inplace=True)
    df['sqft'].fillna(df['sqft'].median(), inplace=True)
    df['lot_size'].fillna(df['lot_size'].median(), inplace=True)
    df['property_age'].fillna(df['property_age'].median(), inplace=True)
    df['property_type_Bunglow'].fillna(0, inplace=True) 
    df['property_type_Condo'].fillna(0, inplace=True)      
    
    return df

