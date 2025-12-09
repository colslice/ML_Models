#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    for k in [1, 3, 10]:
        knn = KNN(k)
        knn.fit(X, y)
        y_train_hat = knn.predict(X)
        error_train = np.mean(np.array(y_train_hat) != y)

        y_test_hat = knn.predict(X_test)
        error_test = np.mean(np.array(y_test_hat) != y_test)

        print(f"k={k} training error: {error_train:.3f} test error: {error_test:.3f}")

        if (k == 1):
            plot_classifier(knn, X, y)
            fname = Path("..", "figs", "q1_3_knnPlot")
            plt.savefig(fname)

            sk_knn = KNeighborsClassifier(k)
            sk_knn.fit(X, y)
            plot_classifier(sk_knn, X, y)
            fname_1 = Path("..", "figs", "q1_3_knnPlot_sklearn")
            plt.savefig(fname_1)

@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))

    cv_accs = []
    
    indices = np.arange(len(X))
    folds = np.array_split(indices, 10)

    for k in ks:

        knn = KNN(k)
        fold_accs = []

        for i in range(10):
            mask = np.ones(len(X), dtype=bool)
            mask[folds[i]] = False

            X_train = X[mask]
            X_validate = X[~mask]
            
            y_train = y[mask]
            y_validate = y[~mask]
            
            knn.fit(X_train, y_train)
            y_hat = knn.predict(X_validate)

            acc = np.mean(np.array(y_hat) == y_validate)
            fold_accs.append(acc)

        cv_accs.append(np.mean(fold_accs))

    test_acc = []

    for k in ks:
        
        knn = KNN(k)
        knn.fit(X, y)
        y_test_hat = knn.predict(X_test)
        acc = np.mean(np.array(y_test_hat) == y_test)
        test_acc.append(acc)


    plt.figure(figsize=(8, 5))
    plt.plot(ks, cv_accs, marker='o', label="Cross-Validation Accuracy")
    plt.plot(ks, test_acc, marker='s', label="Test Accuracy")

    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Accuracy")

    plt.title("Cross-validation and Test Accuracy vs k")
    plt.legend()
    plt.grid(True)
    fname = Path("..", "figs", "q2_2_cv_test_acc")
    plt.savefig(fname)

    train_errors = []
    
    for k in ks:
        knn = KNN(k)
        knn.fit(X, y)
        y_train_hat = knn.predict(X)
        error_train = np.mean(np.array(y_train_hat) != y)
        train_errors.append(error_train)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, train_errors, marker='o', label="Training Error")

    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Error Rate")

    plt.title("Training Error vs k")
    plt.grid(True)
    fname = Path("..", "figs", "q2_4_training")
    plt.savefig(fname)
    
    




@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    print(y[802])
    print(groupnames)



@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    mask = (y == 0)
    X_class0 = X[mask]

    feature_counts = X_class0.sum(axis=0)

    class_count = mask.sum()
    p_x_giveny0 = (feature_counts) / class_count
    print(p_x_giveny0)

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)
    y_hat = model.predict(X)

    mask = (y == 0)
    X_class0 = X[mask]
    print(X_class0)

    feature_counts = X_class0.sum(axis=0)
    class_count = mask.sum()

    p_x_given_y0 = (feature_counts + 10000) / (class_count + 2 * 10000)
    print(p_x_given_y0)





@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
    print("Random Tree")
    evaluate_model(RandomForest(num_trees=1, max_depth=np.inf))


@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]
    min_error_list = []
    ks = list(range(1, 11))
    
    for k in ks:
        best_model = Kmeans(k=k)
        best_model.fit(X)
        y = best_model.predict(X)
        lowest_error = best_model.error(X, y, best_model.means)

        for _ in range(49):
            model = Kmeans(k=k)
            model.fit(X)
            y = model.predict(X)
            current_error = model.error(X, y, model.means)

            if (current_error < lowest_error):
                best_model = model
                lowest_error = current_error
            
        
        min_error_list.append(lowest_error)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, min_error_list, marker='o', label="Minimum error vs k")

    plt.xlabel("k (number of clusters)")
    plt.ylabel("Minimum error")

    plt.title("Minimum Error vs k")
    plt.grid(True)
    fname = Path("..", "figs", "q5_2_minimum_error")
    plt.savefig(fname)
    
         



if __name__ == "__main__":
    main()
