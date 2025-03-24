# Real_Estate_application

This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://w10realestateproject-ie4ctutrjjvhkl4yhhfvs8.streamlit.app/)

password - streamlit

This application predicts real estate property based on user inputs. The model aims to help users estimate the value of properties using machine learning predictions.

## Features

- User-friendly interface powered by Streamlit.
- Input form to enter details such as year sold, property tax, insurance, lot size and other relevant factors.
- Real-time prediction of propperty prices based on the trained model.
- Accessible via Streamlit Community Cloud.

## Dataset

The application is trained on the **final.csv dataset**. It includes features like:

- Year Sold
- Property Tax
- Insurance
- Number of Beds
- Number of Baths
- Sqft
- Year Built
- Lot Size
- Basement
- popular
- recession
- Property Age
- Property Type

## Technologies Used

- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

## Model

The predictive model is trained using the final.csv dataset. It applies preprocessing steps like encoding categorical variables and scaling numerical features. The model used may include algorithms such as Logistic Regression, Random Forest, or XGBoost.

## Future Enhancements

- Adding support for multiple datasets.
- Incorporating explainability tools like SHAP to provide insights into predictions.
- Adding visualizations to better represent user input and model predictions.

## Installation (for local deployment)

If you want to run the application locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/credit_eligibility_application.git
   cd credit_eligibility_application

   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt

   ```

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

#### Thank you for using the Credit Eligibility Application! Feel free to share your feedback.
