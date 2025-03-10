#fitting of data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import joblib
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from random import randint


# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,  ExtraTreesClassifier

# grid search
from sklearn.model_selection import RandomizedSearchCV

# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score


#Bagging
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv('../dataset/mentalhealth-data.csv')
mentalhealth_df=pd.read_csv('../dataset/mentalhealth-data.csv')



# Define feature columns and target variable
feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
X = mentalhealth_df[feature_cols]
y = mentalhealth_df['treatment']  # Ensure y is correctly selected

# Split X and y into training and testing sets (corrected parameter name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Create dictionaries for final graph
methodDictionary = {}

#Random Forest
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

labels = [feature_cols[f] for f in range(X.shape[1])]

# Plot the feature importances of the forest
plt.figure(figsize=(12, 8))  # ✅ Fixed `figheight` issue

plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), labels, rotation='vertical')
plt.xlim([-1, X.shape[1]])

plt.show()


def evalClassModel(model, y_test, y_pred_class, plot=False):
    # Classification accuracy: percentage of correct predictions
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred_class))  # calculate accuracy

    # Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    print('Null accuracy:\n',
          y_test.value_counts())  # examine the class distribution of the testing set (using a Pandas Series method)

    print('Percentage of ones:', y_test.mean())  # calculate the percentage of ones

    # calculate the percentage of zeros
    print('Percentage of zeros:', 1 - y_test.mean())  # calculate the percentage of zeros

    # Comparing the true and predicted response values
    print('True:', y_test.values[0:25])
    print('Pred:', y_pred_class[0:25])

    # Confusion matrix
    # save confusion matrix and slice into four pieces
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    # [row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    # visualize Confusion Matrix
    sns.heatmap(confusion, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Metrics computed from a confusion matrix

    # Classification Accuracy: Overall, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred_class)
    print('Classification Accuracy:', accuracy)

    # Classification Error: Overall, how often is the classifier incorrect?
    print('Classification Error:', 1 - metrics.accuracy_score(y_test, y_pred_class))

    # False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
    false_positive_rate = FP / float(TN + FP)
    print('False Positive Rate:', false_positive_rate)

    # Precision: When a positive value is predicted, how often is the prediction correct?
    print('Precision:', metrics.precision_score(y_test, y_pred_class))

    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    print('AUC Score:', metrics.roc_auc_score(y_test, y_pred_class))

    # calculate cross-validated AUC
    print('Cross-validated AUC:', cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean())

    "----------------------------------------"
    # Adjusting the classification threshold
    "----------------------------------------"
    # print the first 10 predicted responses
    # 1D array (vector) of binary values (0, 1)
    print('First 10 predicted responses:\n', model.predict(X_test)[0:10])

    print('First 10 predicted probabilities of class members:\n',
          model.predict_proba(X_test)[0:10])  # print the first 10 predicted probabilities of class membership

    model.predict_proba(X_test)[0:10, 1]  # print the first 10 predicted probabilities for class 1

    y_pred_prob = model.predict_proba(X_test)[:, 1]  # store the predicted probabilities for class 1

    if plot == True:
        # histogram of predicted probabilities

        plt.rcParams['font.size'] = 12  # adjust the font size

        plt.hist(y_pred_prob, bins=8)  # 8 bins

        plt.xlim(0, 1)  # x-axis limit from 0 to 1
        plt.title('Histogram of predicted probabilities')
        plt.xlabel('Predicted probability of treatment')
        plt.ylabel('Frequency')

    # predict treatment if the predicted probability is greater than 0.3
    # it will return 1 for all values above 0.3 and 0 otherwise
    # results are 2D so we slice out the first column
    y_pred_prob = y_pred_prob.reshape(-1, 1)
    y_pred_class = binarize(y_pred_prob, threshold=0.3)

    # print the first 10 predicted probabilities
    print('First 10 predicted probabilities:\n', y_pred_prob[0:10])

    "------------------------------------------"
    # ROC Curves and Area Under the Curve (AUC)
    "------------------------------------------"
    roc_auc = metrics.roc_auc_score(y_test, y_pred_prob)

    # fpr: false positive rate
    # tpr: true positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    if plot == True:
        plt.figure()

        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for treatment classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.legend(loc="lower right")
        plt.show()

    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    # we pass y_test and y_pred_prob
    # we do not use y_pred_class, because it will give incorrect results without generating an error
    # roc_curve returns 3 objects fpr, tpr, thresholds

    # define a function that accepts a threshold and prints sensitivity and specificity
    def evaluate_threshold(threshold):
        # Sensitivity: When the actual value is positive, how often is the prediction correct?
        # Specificity: When the actual value is negative, how often is the prediction correct?print('Sensitivity for ' + str(threshold) + ' :', tpr[thresholds > threshold][-1])
        print('Specificity for ' + str(threshold) + ' :', 1 - fpr[thresholds > threshold][-1])

    # One way of setting threshold
    predict_mine = np.where(y_pred_prob > 0.50, 1, 0)
    confusion = metrics.confusion_matrix(y_test, predict_mine)
    print(confusion)

    return accuracy


def tuningCV(knn):
    # search for an optimal value of K for KNN
    k_range = list(range(1, 31))
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    print(k_scores)
    # plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def tuningMultParam(knn):
    # Searching multiple parameters simultaneously
    # define the parameter values that should be searched
    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']

    # create a parameter grid: map the parameter names to the values that should be searched
    param_grid = dict(n_neighbors=k_range, weights=weight_options)
    print(param_grid)

    # instantiate and fit the grid
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)

    # view the complete results
    print(grid.grid_scores_)

    # examine the best model
    print('Multiparam. Best Score: ', grid.best_score_)
    print('Multiparam. Best Params: ', grid.best_params_)


def tuningRandomizedSearchCV(model, param_dist):
    # Searching multiple parameters simultaneously
    # n_iter controls the number of searches
    rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)
    rand.fit(X, y)
    rand.cv_results_

    # examine the best model
    print('Rand. Best Score: ', rand.best_score_)
    print('Rand. Best Params: ', rand.best_params_)

    # run RandomizedSearchCV 20 times (with n_iter=10) and record the best score
    best_scores = []
    for _ in range(20):
        rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10)
        rand.fit(X, y)
        best_scores.append(round(rand.best_score_, 3))
    print(best_scores)


def logisticRegression():
    # Train a logistic regression model on the training set
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    # Make class predictions for the testing set
    y_pred_class = logreg.predict(X_test)

    print('----------- Logistic Regression -------------')

    # Ensure evalClassModel() works correctly
    accuracy_score = evalClassModel(logreg, y_test, y_pred_class, plot=True)

    # Store accuracy score in dictionary
    methodDictionary['Log. Regres.'] = accuracy_score * 100


logisticRegression()


def Knn():
    # Calculating the best parameters
    knn = KNeighborsClassifier(n_neighbors=5)

    # tuningCV(knn)
    # tuningGridSerach(knn)
    # tuningMultParam(knn)

    # define the parameter values that should be searched
    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']

    # train a KNeighborsClassifier model on the training set
    knn = KNeighborsClassifier(n_neighbors=27, weights='uniform')
    knn.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = knn.predict(X_test)

    print('------------ KNeighborsClassifier ------------')

    accuracy_score = evalClassModel(knn, y_test, y_pred_class, True)

    # Data for final graph
    methodDictionary['KNN'] = accuracy_score * 100


Knn()


def treeClassifier():
    # Calculating the best parameters
    tree = DecisionTreeClassifier()
    featuresSize = feature_cols.__len__()
    param_dist = {"max_depth": [3, None],
                  "max_features": randint(1, featuresSize),
                  "min_samples_split": randint(2, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}

    # train a decision tree model on the training set
    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=8, max_features=6, criterion='entropy',
                                  min_samples_leaf=7)
    tree.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = tree.predict(X_test)

    print('------------ Tree classifier -------------')

    accuracy_score = evalClassModel(tree, y_test, y_pred_class, True)

    # Data for final graph
    methodDictionary['Tree clas.'] = accuracy_score * 100


treeClassifier()


def randomForest():
    # Calculating the best parameters
    forest = RandomForestClassifier(n_estimators=20)

    featuresSize = feature_cols.__len__()
    param_dist = {"max_depth": [3, None],
                  "max_features": randint(1, featuresSize),
                  "min_samples_split": randint(2, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}

    # Building and fitting my_forest
    forest = RandomForestClassifier(max_depth=None, min_samples_leaf=8, min_samples_split=2, n_estimators=20,
                                    random_state=1)
    my_forest = forest.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(my_forest, "../models/_random_forest_model.pkl")

    # make class predictions for the testing set
    y_pred_class = my_forest.predict(X_test)

    print('------------ Random Forests ------------')

    accuracy_score = evalClassModel(my_forest, y_test, y_pred_class, True)

    # Data for final graph
    methodDictionary['R. Forest'] = accuracy_score * 100


randomForest()


def bagging():
    # Building and fitting
    bag = BaggingClassifier(DecisionTreeClassifier(), max_samples=1.0, max_features=1.0, bootstrap_features=False)
    bag.fit(X_train, y_train)

    # make class predictions for the testing set
    y_pred_class = bag.predict(X_test)

    print('-------------- Bagging -------------')

    accuracy_score = evalClassModel(bag, y_test, y_pred_class, True)

    # Data for final graph
    methodDictionary['Bagging'] = accuracy_score * 100


bagging()


def boosting():
    # Building and fitting the model
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)

    # Use `estimator` instead of `base_estimator` and explicitly set algorithm="SAMME"
    boost = AdaBoostClassifier(estimator=clf, n_estimators=500, algorithm="SAMME")

    boost.fit(X_train, y_train)

    # Make class predictions for the testing set
    y_pred_class = boost.predict(X_test)

    print('-------------- Boosting -----------------')

    # Evaluate the model
    accuracy_score = evalClassModel(boost, y_test, y_pred_class, plot=True)

    # Data for final graph
    methodDictionary['Boosting'] = accuracy_score * 100


boosting()


def plotSuccess():
    success = pd.Series(methodDictionary)
    success = success.sort_values(ascending=False)

    plt.figure(figsize=(12, 8))

    # Create bar plot
    ax = success.plot(kind='bar')
    plt.ylim([70.0, 90.0])  # Set y-axis limits
    plt.xlabel('Method')
    plt.ylabel('Percentage')
    plt.title('Success of Methods')

    plt.show()


plotSuccess()