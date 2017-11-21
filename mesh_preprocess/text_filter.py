# encoding=utf-8
import pickle
import gc

print('begin to load pickle...')
with open('../PKL/all_abstract.pkl', 'rb') as file:
    all_abstract = pickle.load(file)

with open('../PKL/lookup_dict.pkl', 'rb') as file:
    embeddings = pickle.load(file)

print('begin to filter...')
filter_abstract = []
for line in all_abstract:
    filter_article = []
    for word in line:
        if word in embeddings:
            filter_article.append(word)
    filter_abstract.append(filter_article)

del all_abstract
gc.collect()
print('begin to dump all_abstract_filter...')
with open('../PKL/all_abstract_filter.pkl', 'wb') as file:
    pickle.dump(filter_abstract, file)
