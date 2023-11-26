import numpy as np

# Modelling
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV


def train(X_train, y_train, params):
    params = params

    search = RandomizedSearchCV(GradientBoostingClassifier(random_state=0, n_estimators=500, n_iter_no_change=5), 
                                params, n_iter=50, cv=3, n_jobs=-1)

    search.fit(X_train, y_train)

    print(search.best_params_)

    clf = search.best_estimator_

    clf.fit(X_train, y_train)
    
    return clf
    # score = clf.score(X_train, y_train)
    
    

def predict(X_test, y_test, clf):
    
    y_pred = clf.predict(X_test)

    # Get the confusion matrix performance of the dataset
    clf_accuracy = accuracy_score(y_test, y_pred)
    clf_precision = precision_score(y_test, y_pred)
    clf_recall = recall_score(y_test, y_pred)
    clf_f1 = 2 * (clf_precision * clf_recall) / (clf_precision + clf_recall)

    print("Accuracy:", clf_accuracy)
    print("Precision:", clf_precision)
    print("Recall:", clf_recall)
    print("F1 Score:", clf_f1)
    
    return y_pred