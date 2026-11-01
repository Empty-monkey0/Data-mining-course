## Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier ##Install the package "scikit-learn"
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import tree
from matplotlib import pyplot as plt
import os

os.chdir(r"/home/vboxuser/CIS3920/Lab 2 Decision Tree attached files Mar 31, 2026 235 PM")
## Set working directory


## Import Data
mydata = pd.read_csv("BankLoan.csv")


## Determine features and target (class)
feature_cols = ['Male',	'age',	'income',	'married',	'children', 'car', 'save_act', 'current_act', 'mortgage']
#split dataset in features and target variable
target_col = ['payback']
X = mydata[feature_cols] # Features
y = mydata[target_col].values.ravel() # Target variable



## Splitting the data into two parts: (1) a training set and (2) a test set.
# Split dataset into training set and test set
# The random_state parameter is used to control this randomness, providing a way to reproduce results or ensure consistency across different runs of the same code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1) # 70% training and 30% test


## Build Decision Tree
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=5)
# Train Decision Tree Classifer
results = clf.fit(X_train,y_train)


## Visualizing Decision Trees
plt.figure(figsize=(40,20))
tree.plot_tree(results, feature_names = X.columns)
plt.savefig('treeplotV1.png')
## You will see a .png file named decistion_tree.png being created in the main folder.

## Evaluating Model
# Predict the response for test dataset
y_pred1 = clf.predict(X_train)
y_pred2 = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy of train dataset(criterion=entropy, max_depth=5, min_samples_split=5):",metrics.accuracy_score(y_train, y_pred1))
print("Accuracy of test dataset(criterion=entropy, max_depth=5, min_samples_split=5):",metrics.accuracy_score(y_test, y_pred2))



## Optimize Your Decision Tree
# Create Decision Tree classifer object – set the criterion to entropy and control the maximum depths
clf = DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_split=5)
##criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
##The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
##min_samples_split: The minimum number of samples required to split an internal node. Default=2


# Train Decision Tree Classifer
results = clf.fit(X_train,y_train)
#Plot the results
plt.figure(figsize=(40,20))
tree.plot_tree(results, feature_names = X.columns)
plt.savefig('treeplotV2.png')
#Predict the response for test dataset
y_pred3 = clf.predict(X_train)
y_pred4 = clf.predict(X_test)

print("Accuracy of train dataset (criterion=entropy, max_depth=10, min_samples_split=5):",metrics.accuracy_score(y_train, y_pred3))
print("Accuracy of test dataset (criterion=entropy, max_depth=10, min_samples_split=5):",metrics.accuracy_score(y_test, y_pred4))

## Build Decision Tree
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_split=3)
# Train Decision Tree Classifer
results = clf.fit(X_train,y_train)


## Visualizing Decision Trees
plt.figure(figsize=(40,20))
tree.plot_tree(results, feature_names = X.columns)
plt.savefig('treeplot.png')
## You will see a .png file named decistion_tree.png being created in the main folder.

## Evaluating Model
# Predict the response for test dataset
y_pred1 = clf.predict(X_train)
y_pred2 = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy of train dataset(criterion=entropy, max_depth=4, min_samples_split=3):",metrics.accuracy_score(y_train, y_pred1))
print("Accuracy of test dataset(criterion=entropy, max_depth=4, min_samples_split=3):",metrics.accuracy_score(y_test, y_pred2))



### Random Forests
# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=1)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
predictions = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Random Forest Accuracy: {accuracy:.2f}")