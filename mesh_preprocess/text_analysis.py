# encoding=utf-8
import json
import pickle
import time
import gc

# total 12834584
front_select = 6834585

start_time = time.time()
big_path = '/media/cb201/8A5EE1D45EE1B959/fzh/allMeSH_2017.json'


# 判断字符串是否是英文单词
def isalpha(word):
    if word.isalpha():
        return True
    index = word.find('-')
    if index <= 0:
        return False
    if word[:index].isalpha() and word[index + 1:].isalpha():
        return True
    return False


print('loading json...')
with open(big_path, 'r', encoding='utf-8', errors='ignore') as file:
    doc = json.load(file)

articles = doc['articles'][:front_select]
print('total articles:', len(doc['articles']), 'and selection:', len(articles))

print('loading stopwords...')
stopwords_set = set()
with open('../DATA/stopwords.txt', 'r') as file:
    for line in file.readlines():
        stopwords_set.add(line.strip())

# note the abstractText without stop words
print('word filtering and counting...')
all_abstract = []
for cita in articles:
    abstract = cita['abstractText'].split()
    abstract_list = []
    for word in abstract:
        word = word.lower()
        if word in stopwords_set:
            continue
        if isalpha(word):
            abstract_list.append(word)
        elif isalpha(word[:-1]):  # 如果是句尾单词,则隔开
            abstract_list.append(word[:-1])
            abstract_list.append(word[-1])

    all_abstract.append(abstract_list)

del doc
gc.collect()
print('dumping pickle...')
with open('../PKL/all_abstract.pkl', 'wb') as file:
    pickle.dump(all_abstract, file)
del all_abstract
gc.collect()
print('all_abstract.pkl dumped!')

end_time = time.time()
print((end_time - start_time) / 60, 'minutes')

