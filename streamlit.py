import matplotlib.pyplot as plt
import pickle
import streamlit as st
from src.models.predict_model import predict_price
import numpy as np



# Set the page title and description
st.title("Real Estate Price Predictor")
st.write("""
This app predicts the price of a real estate property based on various characteristics.
""")

# # Optional password protection (remove if not needed)
# password_guess = st.text_input("Please enter your password?")
# # this password is stores in streamlit secrets
# if password_guess != st.secrets["password"]:
#     st.stop()

# Load the pre-trained model
rf_pickle = open("models/RFmodel.pkl", "rb")
rf_model = pickle.load(rf_pickle)
rf_pickle.close()


# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Property Details")
    
    # Year Sold
    year_sold = st.number_input("Year Sold", min_value=1900, max_value=2050)
    
    # Property Tax
    property_tax = st.number_input("Property Tax", min_value=0, value=234)
    
    # Insurance
    insurance = st.number_input("Insurance", min_value=0, value=81)
    
    # Number of Beds
    beds = st.number_input("Number of Beds", min_value=1, value=1)
    
    # Number of Baths
    baths = st.number_input("Number of Baths", min_value=1, value=1)
    
    # Square Footage (sqft)
    sqft = st.number_input("Square Footage (sqft)", min_value=1, value=600)
    
    # Year Built
    year_built = st.number_input("Year Built", min_value=1900, max_value=2050)
    
    # Lot Size
    lot_size = st.number_input("Lot Size", min_value=0, value=0)
    
    # Basement Present
    basement = st.checkbox("Basement Present", value=False)
    
    # Popular Area
    popular = st.checkbox("Popular Area", value=False)
    
    # During Recession
    recession = st.checkbox("During Recession", value=False)
    
    # Property Age
    property_age = st.number_input("Property Age", min_value=0, value=10)
    
    # Property Type Bunglow
    property_type_Bunglow = st.checkbox("Bungalow Type", value=False)
    
    # Property Type Condo
    property_type_Condo = st.checkbox("Condo Type", value=False)
    
    # Submit button
    submitted = st.form_submit_button("Predict Price")# year sold input
    

# Handle the dummy variables to pass to the model
if submitted:
    
    # Convert checkbox values to integers
    basement_int = 1 if basement else 0
    popular_int = 1 if popular else 0
    recession_int = 1 if recession else 0
    property_type_Bunglow_int = 1 if property_type_Bunglow else 0
    property_type_Condo_int = 1 if property_type_Condo else 0
    

    # Prepare the input for prediction. This has to go in the same order as it was trained
    prediction_input = np.array([
        year_sold, property_tax, insurance, beds, baths, sqft,
        year_built, lot_size, basement_int, popular_int,
        recession_int, property_age, property_type_Bunglow_int, property_type_Condo_int
    ])


        
        
    # Make prediction
    predicted_price = rf_model.predict(prediction_input.reshape(1, -1))[0]

    # Display result
    st.subheader("Prediction Result:")
    st.write(f"The predicted price for the given property is: **${predict_price():.2f}**")







