# encoding=utf-8

import pickle
import numpy as np

embeddings = []
lookup_dict = {}
embeddings.append(np.zeros(shape=[100], dtype=np.float32))
lookup_dict['#PADDING#'] = 0
with open('../DATA/embeddings.100', 'r', encoding='utf-8') as file:
    file.readline()
    for rowid, line in enumerate(file.readlines()):
        row_split = line.split()
        word = row_split[0]
        values = np.array(row_split[1:], dtype=np.float32)
        assert len(values) == 100
        embeddings.append(values)
        lookup_dict[word] = rowid

embeddings = np.array(embeddings, dtype=np.float32)
with open('../PKL/lookup_dict.pkl', 'wb') as file:
    pickle.dump(lookup_dict, file)

with open('../PKL/lookup_matrix.pkl', 'wb') as file:
    pickle.dump(embeddings, file)
