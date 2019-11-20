This is to create machine learning models capable of classifying candidate exoplanets from the raw dataset using Logistic Regression., SVM, Decision Tree and Random Forest
The process is as below:

1. Preprocess the raw data
2. Tune the models
3. Compare the models

import pandas as pd

# Read the CSV and Perform Basic Data Cleaning

df = pd.read_csv("cumulative.csv")
df = df.drop(columns=["rowid", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_score", "koi_tce_delivname"])
# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')
# Drop the null rows
df = df.dropna()
df.head()

# Create a Train Test Split

Use `koi_disposition` for the y values

from sklearn.model_selection import train_test_split

y = df["koi_disposition"]
X = df.drop(columns=["koi_disposition"])


y.head()

X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

X_train.head()


X_test.head()

# Pre-processing

Scale the data using the MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
X_scaler = MinMaxScaler().fit(X_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

X_train_scaled


# Create and Train the Logistic Regression Model



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier

classifier.fit(X_train_scaled, y_train)

# classifier.fit(X_train, y_train)

print(f"Training Data Score: {classifier.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test_scaled, y_test)}")

classifier.fit(X_train, y_train)

print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")

# Train the Support Vector Machine

from sklearn.svm import SVC 
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

print(f"Training Data Score: {model.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {model.score(X_test_scaled, y_test)}")

# Hyperparameter Tuning

Use `GridSearchCV` to tune the `C` and `gamma` parameters

# Create the GridSearchCV model
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [1, 5, 10],
              'gamma': [0.0001, 0.001, 0.01]}
grid = GridSearchCV(model, param_grid, verbose=3)

# Train the model with GridSearch
grid.fit(X_train_scaled, y_train)

print(grid.best_params_)
print(grid.best_score_)

# Decision Tree Model

from sklearn import tree
import pandas as pd
import os

target = df["koi_disposition"]
target_names = ["negative", "positive"]

data = df.drop("koi_disposition", axis=1)
feature_names = data.columns
data.head()

from sklearn.model_selection import train_test_split
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(data, target, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train_scaled, y_train)
clf.score(X_test_scaled, y_test)

# Random Forest Model

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf = rf.fit(X_train_scaled, y_train)
rf.score(X_test_scaled, y_test)


