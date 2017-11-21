# encoding=utf-8
import pickle
import gc

print('begin to load pickle...')
with open('../PKL/all_text.pkl', 'rb') as file:
    all_text = pickle.load(file)

with open('../PKL/lookup_dict.pkl', 'rb') as file:
    lookup_dict = pickle.load(file)

print('begin to filter...')
filter_text = []
for line in all_text:
    filter_article = []
    for word in line:
        if word in lookup_dict:
            filter_article.append(word)
    filter_text.append(filter_article)

del all_text
gc.collect()
print('begin to dump all_text_filter...')
with open('../PKL/all_text_filter.pkl', 'wb') as file:
    pickle.dump(filter_text, file)
