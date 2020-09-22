import pandas as pd

import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

from gensim.models import Word2Vec

import pickle

# Reading
data = pd.read_csv("../data/raw/Shakespeare_data.csv", usecols=['PlayerLine'])

# Preprocessing
text = data.apply(lambda x: ' '.join(x), axis=0)

text = text.str.lower()
text = text.apply(lambda line: re.sub(r'[^A-Za-z0-9 ]', '', line))
text = text.apply(lambda line: word_tokenize(line))

stop_words = stopwords.words('english')
text = text.apply(lambda line: [w for w in line if not w in stop_words])

text = text.apply(lambda line: [WordNetLemmatizer().lemmatize(w) for w in line])

# Training Word2Vec
model = Word2Vec(text, min_count=1)

# Saving model
filename = '../models/Shakespeare_Word2Vec.sav'
pickle.dump(model, open(filename, 'wb'))