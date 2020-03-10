from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse.linalg import svds, eigs
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.feature_selection import VarianceThreshold
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn import metrics, preprocessing

from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import nltk

def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

def emo_vect_tfidf_norm(tweets, emotions_dict):

    words_emo = []

    for tweet in tweets:

        tweet_vect = np.zeros(len(emotions_dict.iloc[0]))

        # Transforma cada tweet a su vector de emociones
        for word in tweet.split():

            # Se verifica si la palabra está incluida en el diccionario de emociones
            try:
                if len(emotions_dict.loc[word]) == 1:
                    w_vect = np.array(emotions_dict.loc[word])
                else:
                    w_vect = np.array(emotions_dict.loc[word].sum(axis = 0, skipna = True))
                tweet_vect += np.array(w_vect)
            except:
                pass

        words_emo.append(tweet_vect)

    words_emo = np.array(words_emo)

    for i in range(words_emo.shape[0]):
        emotions_index = np.nonzero(words_emo[i])[0]
        for j in emotions_index:
            num_tweets = len(np.nonzero(words_emo[:, j])[0])
            words_emo[i][j] = words_emo[i][j]*np.log10((words_emo.shape[0]/num_tweets) + 1)

    for i in range(words_emo.shape[0]):
        norm = np.linalg.norm(words_emo[i])

        if norm == 0:
            norm = 1

        words_emo[i] = words_emo[i]/norm

    return words_emo

def emo_vect_sel_tfidf_norm(tweets, emotions_dict):

    words_emo = []
    emo_list = list(sel_emotions_dict.Categoría.unique())
    num_cols = len(emo_list)

    for tweet in tweets:

        tweet_vect = np.zeros(len(emotions_dict.iloc[0]))

        # Transforma cada tweet a su vector de emociones
        for word in tweet.split():

             # Se verifica si la palabra está incluida en el diccionario de emociones
            try:
                if len(emotions_dict.loc[word]) == 1:
                    w_vect = np.array(sel_emotions_dict.loc[word][:-1], dtype=np.float)
                else:
                    w_vect = np.array(emotions_dict.loc[word][:-1].sum(axis = 0, skipna = True), dtype=np.float)

                tweet_vect += w_vect
            except:
                pass

        words_emo.append(tweet_vect)

    words_emo = np.array(words_emo)

    for i in range(words_emo.shape[0]):
        emotions_index = np.nonzero(words_emo[i])[0]
        for j in emotions_index:
            num_tweets = len(np.nonzero(words_emo[:, j])[0])
            words_emo[i][j] = words_emo[i][j]*np.log10((words_emo.shape[0]/num_tweets) + 1)

    for i in range(words_emo.shape[0]):
        norm = np.linalg.norm(words_emo[i])

        if norm == 0:
            norm = 1

        words_emo[i] = words_emo[i]/norm

    return words_emo

def bow2(tr_txt, V, dict_indices):
    BOW = np.zeros((len(tr_txt),len(V)), dtype=int)
    cont_doc = 0
    for tr in tr_txt:
        fdist_doc = nltk.FreqDist(tr.split())
        for word in fdist_doc:
            if word in dict_indices:
                BOW[cont_doc, dict_indices[word]] = fdist_doc[word]
        cont_doc += 1

    return BOW

def bow3(tr_txt, V, dict_indices):
    BOW = np.zeros((len(tr_txt),len(V)), dtype=np.float64)
    cont_doc = 0
    for tr in tr_txt:
        fdist_doc = nltk.FreqDist(tr.split())
        for word in fdist_doc:
            if word in dict_indices:
                BOW[cont_doc, dict_indices[word]] = fdist_doc[word]
        cont_doc += 1

    for i in range(BOW.shape[0]):
        norm = np.linalg.norm(BOW[i])

        if norm == 0:
            norm = 1

        BOW[i] = BOW[i]/norm

    return BOW

def bow5(tr_txt, V, dict_indices):

    BOW = np.zeros((len(tr_txt),len(V)), dtype=np.float64)
    cont_doc = 0

    df = {}

    for word in dict_indices.keys():
        for tr in tr_txt:
            if word in tr:
                if not word in df:
                    df[word] = 1
                else:
                    df[word] += 1

    for tr in tr_txt:
        fdist_doc = nltk.FreqDist(tr.split())
        for word in fdist_doc:
            if word in dict_indices:
                BOW[cont_doc, dict_indices[word]] = fdist_doc[word]*np.log10((len(tr_txt)/df[word]))
        cont_doc += 1


    for i in range(BOW.shape[0]):
        norm = np.linalg.norm(BOW[i])

        if norm == 0:
            norm = 1

        BOW[i] = BOW[i]/norm

    return BOW

"""
# Recurso de emociones CANADA
emotions_dict = pd.read_csv("emolex.csv")
emotions_dict = emotions_dict.set_index('Spanish (es)')

# Recurso de emociones SEL
sel_emotions_dict = pd.read_csv("SEL_full.txt", sep='\t', encoding = "ISO-8859-1")
sel_emotions_dict = sel_emotions_dict.set_index('Palabra')
"""

# Lee corpus de tweets
reader = CategorizedPlaintextCorpusReader('./', r'mex.*\.txt', cat_pattern=r'(\w+)/*')

tweets_train = reader.raw('mex_train.txt').split('\n')[:-1]
labels_train = reader.raw('mex_train_labels.txt').split('\n')[:-1]
labels_train = list(map(int, labels_train))

tweets_val = reader.raw('mex_val.txt').split('\n')[:-1]
labels_val = reader.raw('mex_val_labels.txt').split('\n')[:-1]
labels_val = list(map(int, labels_val))

tweets_test = reader.raw('mex_test.txt').split('\n')[:-1]

"""
corpus_palabras = []
for doc in tweets_train:
    corpus_palabras += doc.split()
fdist = nltk.FreqDist(corpus_palabras)

V = sortFreqDict(fdist)
V = V[:5000]

dict_indices = dict()
cont = 0
for weight, word in V:
    dict_indices[word] = cont
    cont += 1
"""

vectorizer = TfidfVectorizer(analyzer = 'char', ngram_range = (4, 7), min_df = 10, max_df = 1000)
X_train = vectorizer.fit_transform(tweets_train)
X_val = vectorizer.transform(tweets_val)
X_test = vectorizer.transform(tweets_test)
print("Data dims: ", X_train.shape)

"""
U, D, V = svds(X_train, k=5000, which='LM')
print("V dims: ", V.shape)
Vr = V.T
X_train = X_train@Vr
X_train = X_val@Vr
X_train = X_test@Vr
"""

"""
pca = SparsePCA(n_components=10000)
X_train = pca.fit_transform(X_train.toarray())
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)
"""

#print("Dims reduced: ", X_train.shape)

parameters = {'C': [2, 2.5, 3, 3.5, 4], 'gamma': [.05, .12, .25, .5, 2]}

#svr = svm.LinearSVC(class_weight='balanced')
#svr = SVC()
svr = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

grid = GridSearchCV(estimator=svr, param_grid=parameters, n_jobs=8, scoring="f1_macro", cv=5, verbose=3)

"""
BOW_train_trad=bow2(tweets_train, V, dict_indices)
BOW_val_trad=bow2(tweets_val, V, dict_indices)

BOW_train_emo=emo_vect_tfidf_norm(tweets_train, emotions_dict)
BOW_val_emo=emo_vect_tfidf_norm(tweets_val, emotions_dict)

BOW_train_emo_sel=emo_vect_sel_tfidf_norm(tweets_train, sel_emotions_dict)
BOW_val_emo_sel=emo_vect_sel_tfidf_norm(tweets_val, sel_emotions_dict)

BOW_train = np.concatenate((BOW_train_trad, BOW_train_emo, BOW_train_emo_sel), axis=1)
BOW_val = np.concatenate((BOW_val_trad, BOW_val_emo, BOW_val_emo_sel), axis=1)

BOW_test_trad=bow5(tweets_test, V, dict_indices)
BOW_test_emo=emo_vect_tfidf_norm(tweets_test, emotions_dict)
BOW_test_emo_sel=emo_vect_sel_tfidf_norm(tweets_test, sel_emotions_dict)
BOW_test=np.concatenate((BOW_test_trad, BOW_test_emo, BOW_test_emo_sel), axis=1)
"""

#grid.fit(BOW_train, labels_train)
grid.fit(X_train, labels_train)

#labels_pred = grid.predict(BOW_val)
labels_pred = grid.predict(X_val)

p, r, f, _ = precision_recall_fscore_support(labels_val, labels_pred, average='macro', pos_label=None)

print(metrics.classification_report(labels_val, labels_pred))

#pred_test = grid.predict(BOW_test)
pred_test = grid.predict(X_test)
result = pd.DataFrame(data=pred_test, index=np.array(range(len(pred_test))), columns=["Expected"])
result.to_csv("result.csv", index=True)
