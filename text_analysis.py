import numpy as np
import pandas as pd
import spacy
import nltk
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import os
import spacy.cli
#spacy.cli.download("en_core_web_sm")

files = ['1 - A Game of Thrones.txt','2 - A Clash of Kings.txt', '3 - A Storm of Swords.txt','4 - A Feast for Crows.txt','5 - A Dance with Dragons.txt']
document = []

file_paths = [f"data/{file}" for file in files]

for path in file_paths:
    with open(path, 'r') as f:
        document.append(f.read())

#spacy NLP model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 20_000_000

from spacy.lang.en.stop_words import STOP_WORDS as stopwords
custom_stopwords = {"and", "like"}
stop_words = stopwords | custom_stopwords

def process_text(text):
    """
    This function creates tokens from the spacy nlp model
    The tokens are then processed to remove lemmas and stop_words, and to keep only alphabets
    """
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]

document = str(document)
print(type(document))
token = process_text(document)
print(len(token))


