
import matplotlib.pyplot as plt
from sklearn import tree



def plot_tree(model, feature_names):
    '''
    Plot the decision tree of a trained model.
    
    Parameters:
    - model: Trained scikit-learn DecisionTree or RandomForest model.
    - feature_names: List of feature names corresponding to the model's input features.
    '''
    # Extract one tree from the forest
    estimator = model.estimators_[0]
    
    # Define the path to save the image in the root directory
    image_path = 'tree.png'
    
    # Plot decistion tree
    plt.figure(figsize=(20, 10))
    tree.plot_tree(estimator, feature_names=feature_names, filled=True)
    plt.title("Decision Tree Visualization")
    plt.savefig(image_path, dpi=300)
    plt.close()
    
    return image_path


