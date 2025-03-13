import matplotlib.pyplot as plt
import pickle
import streamlit as st
from src.models.predict_model import predict_price
from datetime import datetime
import pandas as pd



# Set the page title and description
st.title("Real Estate Price Prediction")
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

# Get current year
current_year = datetime.now().year

# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Property Details")
    
    # Year Sold
    year_sold = st.number_input("Year Sold", min_value=1900, max_value=2025, value=current_year)
    
    # Property Tax
    property_tax = st.number_input("Property Tax", min_value=0, step=1000)
    
    # Insurance
    insurance = st.number_input("Insurance", min_value=0, step=1000)
    
    # Number of Beds
    beds = st.number_input("Number of Beds", min_value=1, max_value=5)
    
    # Number of Baths
    baths = st.number_input("Number of Baths", min_value=1, max_value=5)
    
    # Property Size (sqft)
    sqft = st.number_input("Square Footage (sqft)", min_value=100, step=1)
    
    # Year Built
    year_built = st.number_input("Year Built", min_value=1900, max_value=current_year)
    
    # Lot Size
    lot_size = st.number_input("Lot Size", min_value=100, step=1)
    
    # Basement
    basement = st.selectbox("Have Basement?", options = ["Yes", "No"])
    
    # Popular Area
    popular = st.selectbox("Is Popular Area", options = ["Yes", "No"])
    
    # During Recession
    recession = st.selectbox("Was the property sold during Recession", options = ["Yes", "No"])
    
    # Property Age
    property_age = st.number_input("Property Age", min_value=0, step=1)
    
    # Property Type: Bungalow
    property_type_bunglow = st.selectbox("Is it a Bungalow", option= ["Yes", "No"])
    
    # Property Type: Condo
    property_type_condo = st.selectbox("Is it a Condo", option = ["Yes", "No"])
    
    # Submit button
    submitted = st.form_submit_button("Predict Property Price")
    

# Handle the dummy variables to pass to the model
if submitted:
    
    # Convert checkbox values to integers
    basement = 1 if basement == "Yes" else 0
    popular = 1 if popular == "Yes" else 0
    recession = 1 if recession == "Yes" else 0
    property_type_bunglow = 1 if property_type_bunglow == "Yes" else 0
    property_type_condo = 1 if property_type_condo == "Yes" else 0
    

    # Prepare the input for prediction. This has to go in the same order as it was trained
    prediction_input = [[
        year_sold, property_tax, insurance, beds, baths, sqft,
        year_built, lot_size, basement, popular,
        recession, property_age, property_type_bunglow, property_type_condo
    ]]

        
    # Make prediction
    predicted_price = rf_model.predict(prediction_input)

    # Display result
    st.subheader("Predicted Property Price:")
    st.write(f"The predicted price for this property is: ${predict_price[0]:,.2f}")
    
st.write(
    """We used a machine learning (Random Forest) model to predict the property price. The features used in this prediction are ranked by relative
    importance below."""
)
st.image("decision_tree.png")







