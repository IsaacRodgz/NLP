import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn import metrics, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC


def read_data():

    # Training set
    with open('mex_train.txt', 'r') as f:
        corpus_train = f.readlines()
    with open('mex_train_labels.txt', 'r') as f:
        labels_train = f.readlines()
    labels_train = [int(lab.strip('\n')) for lab in labels_train]
    tweets_train = [tw.strip('\n') for tw in corpus_train]

    # Validation set
    with open('mex_val.txt', 'r') as f:
        corpus_val = f.readlines()
    with open('mex_val_labels.txt', 'r') as f:
        labels_val = f.readlines()
    labels_val = [int(lab.strip('\n')) for lab in labels_val]
    tweets_val = [tw.strip('\n') for tw in corpus_val]

    # Test set
    with open('mex_test.txt', 'r') as f:
        corpus_test = f.readlines()
    with open('mex_test_labels.txt', 'r') as f:
        labels_test = f.readlines()
    labels_test = [int(lab.strip('\n')) for lab in labels_test]
    tweets_test = [tw.strip('\n') for tw in corpus_test]

    return tweets_train, labels_train, tweets_val, labels_val, tweets_test, labels_test


if __name__ == '__main__':

    X_train, y_train, X_val, y_val, X_test, y_test = read_data()

    print("{0} for training\n{1} for validation\n{2} for test\n".format(len(X_train), len(X_val), len(X_test)))

    vectorizer = CountVectorizer(analyzer = 'word', ngram_range = (2, 2), min_df = 1, max_df = 2000)
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    parameters = {'C': [2, 2.5, 3, 3.5, 4], 'gamma': [.05, .12, .25, .5, 2]}

    svr = SVC()

    grid = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring="f1_macro", cv=5, verbose=3)

    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_val)

    p, r, f, _ = precision_recall_fscore_support(y_val, y_pred, average='macro', pos_label=None)

    print(metrics.classification_report(y_val, y_pred))

    print(grid.cv_results_)

    #y_pred_test = grid.predict(X_test)
    #print(metrics.classification_report(y_test, y_pred_test))
