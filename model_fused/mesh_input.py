# encoding=utf-8
import pickle
import numpy as np
import random
import time

embedding_size = 100
time_steps = 700



class data_master:
    def __init__(self):
        start_time = time.time()
        print('loading pickle lookup_matrix...')
        with open('../PKL/lookup_matrix.pkl', 'rb') as file:
            lookup_matrix = pickle.load(file)
        with open('../PKL/lookup_dict.pkl', 'rb') as file:
            lookup_dict = pickle.load(file)
        print('loading pickle meshDict...')
        with open('../PKL/meshDict.pkl', 'rb') as file:
            meshDict = pickle.load(file)

        print('loading pickle all_abstract_filter...')
        with open('../PKL/all_abstract_filter.pkl', 'rb') as file:
            abstracts = pickle.load(file)
            print((time.time() - start_time) / 60, 'minutes', 'train samples %d' % len(abstracts))
            train_sentences = []
            print('begin to process training data...')
            count = 0
            for line in abstracts:
                # no embedding here
                sentence = np.zeros([time_steps], dtype=np.int32)
                values = line[0:time_steps]
                padding_size = 0
                if len(values) < time_steps:
                    padding_size = time_steps - len(values)
                for i, value in enumerate(values):
                    sentence[padding_size + i] = lookup_dict[value]
                # print(padding_size, len(values), sentence[0])
                try:
                    assert padding_size + i + 1 == time_steps
                except:
                    print('%d-th sample is empty.' % count, values)
                train_sentences.append(sentence)
                count += 1


        train_labels = []
        print('loading pickle all_mesh...')

        with open('../PKL/all_mesh.pkl', 'rb') as file:
            all_mesh = pickle.load(file)
            for line in all_mesh:
                mesh_array = []
                for mesh in line:
                    mesh_array.append(meshDict[mesh])
                train_labels.append(mesh_array)

        self.train_sentences = np.array(train_sentences, dtype=np.int32)
        self.train_labels = np.array(train_labels)  # dtype='o'
        self.class_num = len(meshDict)
        self.lookup_matrix = lookup_matrix
        self.meshDict = meshDict
        self.lookup_dict = lookup_dict

        print('DataMaster prepared!', self.train_sentences.shape)
        print((time.time() - start_time) / 60, 'minutes')

    def batch_iter(self, batch_size):
        for iter, index in enumerate(range(0, len(self.train_labels), batch_size)):
            batch_x = self.train_sentences[index:index + batch_size]
            batch_y = self.train_labels[index:index + batch_size]
            yield batch_x, self.change_multi_hot(batch_y)

    def shuffle(self):
        indices = list(range(len(self.train_labels)))
        random.shuffle(indices)
        self.train_sentences = self.train_sentences[indices]
        self.train_labels = self.train_labels[indices]

    def change_multi_hot(self, y):
        new_y = np.zeros(shape=(len(y), self.class_num), dtype=np.int32)
        for i, line in enumerate(y):
            for label in line:
                new_y[i][label] = 1
        return new_y


if __name__ == '__main__':
    master = data_master()
