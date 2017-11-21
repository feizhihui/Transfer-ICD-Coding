# encoding=utf-8

from gensim.models import word2vec

# 引入日志配置
import logging
import pickle
import time

start_time = time.time()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

with open('../PKL/all_text.pkl', 'rb') as file:
    sentences = pickle.load(file)

# 构建模型
print('begin to train word2vec model...')
model = word2vec.Word2Vec(sentences, min_count=1)  # default size 100
model.save('../MODEL/word2vec.model')
# 保存词向量
model.wv.save_word2vec_format('../DATA/embeddings.100', binary=False)

end_time = time.time()
print((end_time - start_time) / 60, 'minutes')
