from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle


# Function to train the model
def train_model(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    
    # Train the Random Forest regression model
    model = RandomForestRegressor(n_estimators=200,
                                   max_depth=10,
                                   max_features=5).fit(X_train, y_train)

    
    
    # Save the trained model
    with open('models/RFmodel.pkl', 'wb') as f:
        pickle.dump(model, f)
    

    return model, X_test, y_test


