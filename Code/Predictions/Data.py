# Data.py
# ML classification algorithms on numeric survey CSV

from sklearn.model_selection import train_test_split
from pathlib import Path
from Helper import load_dataset, run_classifiers, plot_model_scores

# load data (using reformatted numeric CSV)
dataset_file = Path(__file__).resolve().parent / '25_numeric.csv'
data = load_dataset(str(dataset_file))

# X is all columns except the last: all the features
X = data.iloc[:, :-1]
# y is the last column: Age groups as numeric (40, 50, 60, 70, 80, etc)
y = data.iloc[:, -1]

# split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# result is a dictionary of classifier names and their accuracies
results = run_classifiers(X_train, X_test, y_train, y_test)

# print results
print(f"Results on {dataset_file}:")
for name, acc in results.items():
    print(f"{name}: {acc*100:.2f}%")

# Plot histogram of model scores
plot_model_scores(results)
