import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

file_path = r"/mnt/data/7d6ca593-29b8-4075-98da-913597534610.png"
try:
    df = pd.read_csv(file_path)
    X = df.drop(columns=df.columns[-1]).values
    y = pd.to_numeric(df[df.columns[-1]], errors='coerce').values
    mask = ~pd.isnull(y)
    X = X[mask]
    y = y[mask]
except Exception:
    iris = load_iris()
    X = iris.data
    y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

classes = np.unique(y_train)
n_classes = classes.size
n_features = X_train.shape[1]
priors = {}
means = {}
vars_ = {}
for c in classes:
    Xc = X_train[y_train==c]
    priors[c] = Xc.shape[0]/X_train.shape[0]
    means[c] = Xc.mean(axis=0)
    vars_[c] = Xc.var(axis=0) + 1e-9

def gaussian_log_likelihood(x, mean, var):
    return -0.5 * np.sum(np.log(2 * np.pi * var)) -0.5 * np.sum(((x-mean)**2)/var)

def predict_manual(X):
    preds = []
    for x in X:
        scores = []
        for c in classes:
            score = np.log(priors[c]) + gaussian_log_likelihood(x, means[c], vars_[c])
            scores.append(score)
        preds.append(classes[np.argmax(scores)])
    return np.array(preds)

y_pred_manual = predict_manual(X_test)
print("manual_gaussian_accuracy", accuracy_score(y_test, y_pred_manual))
print("manual_classification_report")
print(classification_report(y_test, y_pred_manual))

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_builtin = gnb.predict(X_test)
print("sklearn_gnb_accuracy", accuracy_score(y_test, y_pred_builtin))
print("sklearn_gnb_classification_report")
print(classification_report(y_test, y_pred_builtin))

param_grid = {'n_neighbors': list(range(1,21)), 'weights': ['uniform','distance']}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
best_knn = grid.best_estimator_
y_pred_knn = best_knn.predict(X_test)
print("grid_best_params", grid.best_params_)
print("grid_best_score_cv", grid.best_score_)
print("knn_test_accuracy", accuracy_score(y_test, y_pred_knn))
print("knn_classification_report")
print(classification_report(y_test, y_pred_knn))
