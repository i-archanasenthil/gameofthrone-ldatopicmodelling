
import spacy
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import TfidfModel
import os
import spacy.cli
#spacy.cli.download("en_core_web_sm")

files = ['1 - A Game of Thrones.txt','2 - A Clash of Kings.txt', '3 - A Storm of Swords.txt','4 - A Feast for Crows.txt','5 - A Dance with Dragons.txt']
document = []

file_paths = [f"data/{file}" for file in files]

with open(f"data/1 - A Game of Thrones.txt", 'r') as f:
    document = f.read()

#spacy NLP model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 20_000_000

from spacy.lang.en.stop_words import STOP_WORDS as stopwords
custom_stopwords = {"and", "like", "say", "come", "know", "look","ask", "page", "go", "want"}
stop_words = stopwords | custom_stopwords

def process_text(text):
    """
    This function creates tokens from the spacy nlp model
    The tokens are then processed to remove lemmas and stop_words, and to keep only alphabets
    """
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]


def compute_coherence(lda_model, texts, dictionary):
    """
    Compute the the coherence score for the trained LDA model
    c_v is the metric for measuring the semantic similarity and co-occurence of those words
    """
    coherence_model_lda = CoherenceModel(model = lda_model, texts = texts, dictionary = dictionary, coherence = 'c_v')
    return coherence_model_lda.get_coherence()


document = [line for line in text.split('\n') if line.strip()]
documents = [process_text(doc) for doc in document]

texts = documents
id2word = corpora.Dictionary(texts)
id2word.filter_extremes(no_below=5, no_above=0.5)
corpus = [id2word.doc2bow(text) for text in texts]

tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

#Identifying the best number of topics and the best corpus based on the coherence score and no overlapping topics
scores = []
for k in range(2, 21):
    model = LdaModel(corpus=corpus_tfidf, id2word=id2word, num_topics=k, passes=10)
    coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
    score = coherencemodel.get_coherence()
    scores.append((k, score))

# Print best topic number
best = max(scores, key=lambda x: x[1])
print(f"Best num_topics: {best[0]} with coherence score: {best[1]}")

#Identifying the top 5 to find a model with distinct topics
top_5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
# Print top 5 results
for rank, (num_topics, coherence) in enumerate(top_5, start=1):
    print(f"Rank {rank}: num_topics = {num_topics}, coherence score = {coherence:.4f}")

#running the final model with best param
lda_model = LdaModel(
    corpus = corpus,
    id2word = id2word,
    num_topics = 4,
    passes = 10,
    alpha='asymmetric', 
    eta='auto'
)
for idx, topic in lda_model.print_topic(-1):
    print(f"Topic #{idx}: {topic}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    coherence_lda = compute_coherence(lda_model, texts, id2word)
    print('\nCoherence Score: ', coherence_lda)

# Visualize the LDA model
vis = gensimvis.prepare(lda_model, corpus, id2word)
pyLDAvis.display(vis)

