
import matplotlib.pyplot as plt
from sklearn import tree


# Function to plot decision tree
def plot_tree(model, feature_names):
    '''
    Plot the decision tree of a trained model.
    
    Parameters:
    - model: Trained scikit-learn DecisionTree or RandomForest model.
    - feature_names: List of feature names corresponding to the model's input features.
    '''
    
    # Plot the tree
    plt.figure(figsize=(20, 10))
    tree.plot_tree(model, feature_names=feature_names, filled=True)
    plt.title("Decision Tree Visualization")
    plt.savefig("decision_tree.png", dpi=300)
    plt.close()
    


