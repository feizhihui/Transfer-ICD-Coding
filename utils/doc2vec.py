# encoding=utf-8

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from random import shuffle

import numpy
import pickle
import time

import logging
import sys

print('loading doc2vec module')

start_time = time.time()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


class TaggedLineSentence(object):
    def __init__(self):
        with open('../PKL/all_text.pkl', 'rb') as file:
            self.sources = pickle.load(file)

    def to_array(self):
        self.docs = []
        # 文件名，prefix标签
        for id, doc in enumerate(self.sources):
            self.docs.append(TaggedDocument(doc, [str(id)]))
        return self.docs

    # 重新排列
    def docs_perm(self):
        shuffle(self.docs)
        return self.docs


log.info('source load')

log.info('TaggedDocument')
# 读取文件,并对句子进行标注
sent_obj = TaggedLineSentence()
# ==================================
log.info('D2V')
model = Doc2Vec(min_count=1, window=10, size=128, sample=1e-4, negative=5, workers=7)
# 建立字典库
model.build_vocab(sent_obj.to_array())

log.info('Epoch')
for epoch in range(10):
    log.info('EPOCH: {}'.format(epoch))
    # 重新排列，然后训练
    model.train(sent_obj.docs_perm())

log.info('Model Save')
model.save('../MODEL/imdb.d2v')
# ======================
model = Doc2Vec.load('../MODEL/imdb.d2v')

log.info('Sentiment')
# 读取训练样本的embedding向量与样本标签
# 读取训练样本的embedding向量 读取训练样本的embedding向量与样本标签
doc_embeddings = []
for id in range(len(sent_obj.sources)):
    doc_vec = model.docvecs[str(id)]
    doc_embeddings.append(doc_vec)

# print(sentences.sources)
print(time.time() - start_time)

with open('../PKL/doc_embeddings.pkl', 'wb') as file:
    pickle.dump(doc_embeddings, file)
