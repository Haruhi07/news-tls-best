from sentence_transformers import SentenceTransformer

model = SentenceTransformer('stsb-roberta-base')

sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)
print(sentence_embeddings)

a = [1, 2]
b = [3, 4]

import numpy as np

print(np.vstack((a,b)), np.vstack([a,b]))