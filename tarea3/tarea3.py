from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist
import preprocessor as p

with open('mex_train.txt', 'r') as f:
    corpus = f.readlines()

punct = set(['.', ',', ';', ':', '-', '!', '¡', '¿', '?', '"', '\'', '...', '<url>', '*', '@usuario'])

def process_word(w):
    is_punct = True if w in punct else False
    is_digit = w.isnumeric()
    is_stopword = w in stopwords.words('spanish')

    return "" if is_punct or is_digit or is_stopword else w.lower()

def process_sentence(sent):
    s = []
    for w in sent:

        is_punct = True if w in punct else False
        is_digit = w.isnumeric()
        is_stopword = w in stopwords.words('spanish')

        if not(is_punct or is_digit or is_stopword):
            s.append(w.lower())

    return " ".join(s)

tk = TweetTokenizer()
"""
tokens = [process_word(w) for sent in corpus for w in tk.tokenize(sent)]
tokens = list(filter(None, tokens))
dist = FreqDist(tokens)
"""

sentences = [process_sentence(sent) for sent in corpus for w in tk.tokenize(sent)]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
print(vectorizer.get_feature_names()[:50])
