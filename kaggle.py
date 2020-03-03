from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn import metrics, preprocessing

from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

reader = CategorizedPlaintextCorpusReader('./', r'mex.*\.txt', cat_pattern=r'(\w+)/*')

tweets_train = reader.raw('mex_train.txt').split('\n')[:-1]
labels_train = reader.raw('mex_train_labels.txt').split('\n')[:-1]
labels_train = list(map(int, labels_train))

tweets_val = reader.raw('mex_val.txt').split('\n')[:-1]
labels_val = reader.raw('mex_val_labels.txt').split('\n')[:-1]
labels_val = list(map(int, labels_val))

tweets_test = reader.raw('mex_test.txt').split('\n')[:-1]

#vectorizer = CountVectorizer(min_df=3, max_df=500, ngram_range=(1, 4), analyzer='word')

vectorizer = TfidfVectorizer(min_df=3, max_df=1000, stop_words='spanish', ngram_range=(4, 8), analyzer='char')

vtf = vectorizer.fit_transform(tweets_train)

print(vtf.shape)

#pca = PCA(n_components=20000, svd_solver='auto')
#X = pca.fit_transform(vtf.toarray())
#print(X.transpose().shape)

parameters = {'C': [0.007, 0.01, 0.03, .05, .12, .25, .5, 1, 2, 4, 8]}

svr = svm.LinearSVC(class_weight='balanced', max_iter=1500)
grid = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring="f1_macro", cv=5)

grid.fit(vtf, labels_train)

labels_pred = grid.predict(vectorizer.transform(tweets_val))

p, r, f, _ = precision_recall_fscore_support(labels_val, labels_pred, average='macro', pos_label=None)

print(p, r, f)
print(confusion_matrix(labels_val, labels_pred))
print(metrics.classification_report(labels_val, labels_pred))

pred_test = grid.predict(vectorizer.transform(tweets_test))
result = pd.DataFrame(data=pred_test, index=np.array(range(len(pred_test))), columns=["Expected"])
result.to_csv("result.csv", index=True)
