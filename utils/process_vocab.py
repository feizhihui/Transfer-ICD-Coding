# encoding=utf-8
import pickle
import numpy as np

word_dict = {}
embeddings = []
with open('../DATA/embeddings.100', 'r') as file:
    file.readline()
    for i, line in enumerate(file.readlines()):
        rows = line.split()
        word_dict[rows[0]] = i
        embeddings.append(rows[1:])

embeddings = np.array(embeddings, dtype=np.float32)

with open('../PKL/word_dict.pkl', 'wb') as file:
    pickle.dump(word_dict, file)

with open('../PKL/embeddings.pkl', 'wb') as file:
    pickle.dump(embeddings, file)
