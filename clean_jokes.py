import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import gensim
import nltk
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

def extract_key_words(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    
    return continuous_chunk
    

def clean_joke(joke):
    joke = re.sub(r'([^\.\s\w]|_)+', '', joke).replace(".", ". ")
    joke = joke.replace('\r', '') 
    joke = joke.replace('\n', '')
    joke = joke.replace('<br />', '')
    joke = joke.replace('<p>', '')
    joke = joke.replace('&quot;', '')
    joke = joke.replace('&#039;', '')
    #joke = " ".join(extract_key_words(joke))
    #print joke
    return joke

# Basic script to remove special characters and html tags
def clean_joke(input_str, remove_stop = True):
    input_str = re.sub('<[^<]+?>', ' ', input_str)
    out = re.sub('[^A-Za-z0-9]+', ' ', input_str.lower())
    out = ' '.join([i for i in out.split() if i not in stop_words])
    #out = ' '.join([ wordnet_lemmatizer.lemmatize(porter_stemmer.stem(i)) for i in out.split() if i not in stop_words])
    return out

def clean_data():
	f = open("../data/Jokes_labelling.txt")
	f_out = open("../data/Jokes_labelling_cleaned.txt", "w")

	header = True
	for line in f:
		if not header:
			l = line.strip().split('\t')
			joke = clean_joke(l[1])
			l[1] = joke.lower()
			out = "\t".join(l)
			f_out.write(out + "\n")
		else:
			f_out.write(line)
			header = False

# Loading pre-trained word embeddings from Google, using them as extension to TFIDF Features 
def get_word_embeddings(data, model):
    out_sent_vectors = []
    for sent in data:
        sent_vec = []
        for word in sent.split():
            if word in model.vocab:
                sent_vec.append(model.wv[word])

        sent_vec = np.array(sent_vec)
        sent_vec =  np.mean(sent_vec, axis=0)
        out_sent_vectors.append(sent_vec)

    return out_sent_vectors



def get_vectors():
	model = gensim.models.Word2Vec.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)

	f = open("../data/Jokes_labelling.txt")
	f_out = open("../data/Jokes_id_with_vectors.txt", "w")

	joke_ids = []
	jokes = []
	joke_vectors = []
	classes = []

	header = True
	for line in f:
		l = line.strip().split('\t')
		joke = clean_joke(l[1])
		l[1] = joke.lower()
		joke_ids.append(l[0])
		jokes.append(l[1])
		classes.append(l[3])

	vectors = get_word_embeddings(jokes[1:], model)
	vectors = list(np.array(vectors))

	glv_feat_head = ""
	for i in range(300):
		glv_feat_head = glv_feat_head + "glv_" + str(i) + "\t"


	heder_line = joke_ids[0] + "\t" + glv_feat_head

	f_out.write(heder_line + "\n")
	for id,v in zip(joke_ids[1:], vectors):
		v = "\t".join([str(i) for i in v])
		out = id + "\t" + v + "\n"
		f_out.write(out)

	print vectors[0]


#clean_data()
get_vectors()

