from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import bigrams
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

tokens = [process_word(w) for sent in corpus for w in tk.tokenize(sent)]
tokens = list(filter(None, tokens))
dist = FreqDist(tokens)

"""
sentences = [process_sentence(sent) for sent in corpus for w in tk.tokenize(sent)]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
print(vectorizer.get_feature_names()[:50])
"""

# Unigrams

total_words = len(tokens)

unigrams = {}

for w in dist.keys():
    unigrams[w] = float(tokens.count(w))/total_words

common = [(k,v) for k, v in sorted(unigrams.items(), key=lambda item: item[1], reverse=True)]
#print(common[:20])

# Bigrams

sentences = [process_sentence(sent) for sent in corpus for w in tk.tokenize(sent)]
bigrams_w = list(bigrams(sentences))
print(bigrams_w[:10])
