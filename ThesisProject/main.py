#!/usr/bin/env python
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from common import classification_metrics, describe_data, test_env


def read_data(file):
    """Return pandas dataFrame read from csv file"""
    try:
        return pd.read_excel(file)
    except FileNotFoundError:
        sys.exit('ERROR: ' + file + ' not found')


def encode_categorical(df, categorical_columns):
    for column in categorical_columns:
        df[column] = df[column].fillna(value='Missing')
        df = pd.get_dummies(df, prefix=[column], columns=[column],
                            drop_first=True)

    return df


def preprocess_data(df, verbose=False):
    y_column = 'In university after 4 semesters'

    # Features can be excluded by adding column name to list
    drop_columns = []

    categorical_columns = [
        'Faculty',
        'Paid tuition',
        'Study load',
        'Previous school level',
        'Previous school study language',
        'Recognition',
        'Study language',
        'Foreign student'
    ]

    # Handle dependent variable
    if verbose:
        print('Missing y values: ', df[y_column].isna().sum())

    y = df[y_column].values
    # Encode y. Naive solution
    y = np.where(y == 'No', 0, y)
    y = np.where(y == 'Yes', 1, y)
    y = y.astype(float)

    # Drop also dependent variable variable column to leave only features
    drop_columns.append(y_column)
    df = df.drop(labels=drop_columns, axis=1)

    # Remove drop columns for categorical columns just in case
    categorical_columns = [
        i for i in categorical_columns if i not in drop_columns]

    # encode categorical features
    df = encode_categorical(df, categorical_columns)

    # Handle missing data. At this point only exam points should be missing
    # It seems to be easier to fill whole data frame as only particular columns
    if verbose:
        describe_data.print_nan_counts(df)

    # handle missing values
    df.fillna(value=0, inplace=True)

    if verbose:
        describe_data.print_nan_counts(df)

    # Return features data frame and dependent variable
    return df, y


# function implementing logistic regression on dataset
def logistic_regression(X_df, y, verbose=False):
    X = X_df.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = LogisticRegression(solver='sag', random_state=0, penalty='l2',
                             max_iter=1000, multi_class='multinomial')
    clf.fit(X_train, y_train)

    classification_metrics.print_metrics(y_test, clf.predict(X_test),
                                         'Logistic regresssion test data', verbose=verbose)

    return clf, sc


# function implementing KNN on dataset
def k_nn(X_df, y, verbose=False):
    X = X_df.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
    clf.fit(X_train, y_train)

    classification_metrics.print_metrics(y_test, clf.predict(X_test),
                                         'KNN test data', verbose=verbose)

    return clf, sc


# function implementing SVM on dataset
def svm_clf(X_df, y, verbose=False):
    X = X_df.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = SVC(kernel='sigmoid', C=1.0, gamma=0.2,
              probability=True, tol=1e-6, random_state=0)
    clf.fit(X_train, y_train)

    classification_metrics.print_metrics(y_test, clf.predict(X_test),
                                         'SCV test data',
                                         verbose=verbose)
    return clf, sc


# function implementing naive bayes on dataset
def naive_bayes_clf(X_df, y, verbose=False):
    X = X_df.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    clf = ComplementNB()
    clf.fit(X_train, y_train)

    classification_metrics.print_metrics(y_test, clf.predict(X_test),
                                         'Naive bayes test data',
                                         verbose=verbose)

    return clf


# function implementing decision tree on dataset
def decision_tree_clf(X_df, y, verbose=False):
    X = X_df.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)

    classification_metrics.print_metrics(y_test, clf.predict(X_test),
                                         'Decision tree test data',
                                         verbose=verbose)

    return clf


# function implementing random forest on dataset
def random_forest_clf(X_df, y, verbose=False):
    X = X_df.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)

    classification_metrics.print_metrics(y_test, clf.predict(X_test),
                                         'Random forest classifier test data',
                                         verbose=verbose)

    return clf


if __name__ == '__main__':
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    students = read_data('data/students.xlsx')

    # show histograms
    students.hist()
    pyplot.show()

    # scatter Matrix Plot
    sm = scatter_matrix(students, alpha=0.2, figsize=(4, 4), diagonal='kde')

    # Change label rotation
    [s.xaxis.label.set_rotation(-20) for s in sm.reshape(-1)]
    [s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]

    # May need to offset label when rotating to prevent overlap of figure
    [s.get_yaxis().set_label_coords(-0.3, 0.5) for s in sm.reshape(-1)]

    # Hide all ticks
    [s.set_xticks(()) for s in sm.reshape(-1)]
    [s.set_yticks(()) for s in sm.reshape(-1)]
    # show scatter matrix
    pyplot.show()

    # Correlation Matrix Heatmap
    f, ax = pyplot.subplots()
    corr = students.corr()
    hm = sns.heatmap(corr, annot=True, cmap="coolwarm", xticklabels="", yticklabels="")
    t = f.suptitle('Students scores Correlation Heatmap', fontsize=14)
    # show heat map
    pyplot.show()

    # PRINT_OVERVIEW AND PRINT_CATEGORICAL FUNCTIONS
    # FILE NAME AS ARGUMENT
    describe_data.print_overview(students, file='results/students_overview.txt')
    describe_data.print_categorical(students, file='results/students_categorical_features.txt')

    students_X, students_y = preprocess_data(students)

    X_train, X_test, y_train, y_test = train_test_split(
        students_X, students_y, test_size=0.25, random_state=0)

    # CLASSIFIERS FUNCTIONS
    verbose = False

    log_reg_clf, log_reg_sc = logistic_regression(
        students_X, students_y, verbose=verbose)
    k_nn_clf, k_nn_sc = k_nn(students_X, students_y, verbose=verbose)
    svm_clf = svm_clf(students_X, students_y, verbose=verbose)
    nb_clf = naive_bayes_clf(students_X, students_y, verbose=verbose)
    dt_clf = decision_tree_clf(students_X, students_y, verbose=verbose)
    rf_clf = random_forest_clf(students_X, students_y, verbose=verbose)

    # Spot Check Algorithms
    models = [('LR', LogisticRegression(solver='sag', random_state=0, penalty='l2',
                                        max_iter=5000, multi_class='multinomial')),
              ('KNN', KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)),
              ('SVM', SVC(kernel='sigmoid', C=1.0, gamma=0.2,
                          probability=True, tol=1e-6, random_state=0)),
              ('NB', ComplementNB()),
              ('CART', DecisionTreeClassifier(random_state=0)),
              ('RD', RandomForestClassifier(n_estimators=100, random_state=0))]

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    print('Done')
