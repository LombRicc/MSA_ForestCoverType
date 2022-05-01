import numpy
import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier

from adaboost import AdaBoost

from datetime import datetime

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

dataset = numpy.loadtxt(open("covtype.csv", "rb"), delimiter=",", skiprows=1)

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Starting time =", current_time)

print(dataset)

X = np.delete(dataset, 54, 1)
y = dataset[:, 54]

#t number of boosting rounds
t = 10

#k number of folds
k = 4

accMeans = []

kf = KFold(n_splits=k)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = AdaBoost(boostingRounds=t)
    #clf = AdaBoostClassifier(n_estimators=t)

    maxPred = [-1] * len(y_test)
    accuracies = []
    for classifier in range(7):
        y_train[y_train == classifier + 1] = 1
        y_test[y_test == classifier + 1] = 1
        y_train[y_train != classifier + 1] = -1
        y_test[y_test != classifier + 1] = -1
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        for i in range(len(y_pred)):
            if(y_pred[i] > maxPred[i]):
                maxPred[i] = y_pred[i]

        accuracies.append(accuracy(y_test, maxPred))

    accMeans.append(sum(accuracies) / len(accuracies))

    print("Accuracy:", round(sum(accMeans)/len(accMeans), 2))

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Ending time =", current_time)
