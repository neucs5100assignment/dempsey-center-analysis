# Helper.py
# Helper functions for Data.py and Data2.py

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# load_dataset takes in a file path
# returns a pandas DataFrame
def load_dataset(file_path):
    # read_csv reads a CSV file
    return pd.read_csv(file_path, header=None)

# run_classifiers takes in training and testing data
def run_classifiers(X_train, X_test, y_train, y_test):
    # results is a dictionary to store the accuracy of each classifier
    results = {}
    # Naive Bayes
    # call make_pipeline to create a pipeline with MinMaxScaler and MultinomialNB
    nb = make_pipeline(MinMaxScaler(), MultinomialNB())
    # call fit to train the Naive Bayes classifier
    nb.fit(X_train, y_train)
    # call accuracy_score to calculate the accuracy
    results['NaiveBayes'] = accuracy_score(y_test, nb.predict(X_test))
    # Gaussian Naive Bayes
    gnb = GaussianNB(var_smoothing=1e-6)
    # call fit to train the GaussianNB classifier
    gnb.fit(X_train, y_train)
    # call accuracy_score to calculate the accuracy
    results['GaussianNB'] = accuracy_score(y_test, gnb.predict(X_test))
    # KNN (k=3)
    knn = KNeighborsClassifier(n_neighbors=3)
    # call fit to train the KNN classifier
    knn.fit(X_train, y_train)
    # call accuracy_score to calculate the accuracy
    results['KNN (k=3)'] = accuracy_score(y_test, knn.predict(X_test))
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    # call fit to train the Decision Tree classifier
    dt.fit(X_train, y_train)
    # call accuracy_score to calculate the accuracy
    results['DecisionTree'] = accuracy_score(y_test, dt.predict(X_test))
    # return the results dictionary
    return results

# plot_model_scores plots a histogram of model accuracies
def plot_model_scores(results):
    from pathlib import Path
    
    model_names = list(results.keys())
    accuracies = [score * 100 for score in results.values()]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, accuracies, color='steelblue', edgecolor='black')
    plt.title('Age Group Prediction Model Accuracies')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=15)

    # Add value labels on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 1,
            f'{acc:.2f}%',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    plt.tight_layout()
    
    # Save as PNG
    output_file = Path(__file__).resolve().parent / 'accuracies.png'
    plt.savefig(str(output_file), dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {output_file}")
    
    plt.show()

