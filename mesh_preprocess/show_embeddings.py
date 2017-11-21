# encoding=utf-8
import pickle

with open('../PKL/embeddings.pkl', 'rb') as file:
    embeddings = pickle.load(file)

print('embeddings dictionary size:', len(embeddings))
print('embeddings dimensions:', len(embeddings['a']))
