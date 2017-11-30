# encoding=utf-8
import pickle
import numpy as np
import random

time_steps = 700
class_num = 6984
training_rate = 0.9
shuffle = False


class data_master:
    def __init__(self):
        with open('../PKL/lookup_matrix.pkl', 'rb') as file:
            self.embeddings = pickle.load(file)
        with open('../PKL/doc_embeddings.pkl', 'rb') as file:
            doc_embeddings = pickle.load(file)
        with open('../PKL/lookup_dict.pkl', 'rb') as file:
            self.word_dict = pickle.load(file)
        with open('../PKL/code_dict.pkl', 'rb') as file:
            self.code_dict = pickle.load(file)
        with open('../PKL/all_text.pkl', 'rb') as file:
            all_text = pickle.load(file)
        with open('../PKL/all_code.pkl', 'rb') as file:
            all_code = pickle.load(file)

        data_sentences = []
        for line in all_text:
            sentence = np.zeros([time_steps])
            # print('sentence lens:', len(line))
            line = line[:time_steps]
            paddings = 0
            if len(line) < time_steps:
                paddings = time_steps - len(line)
            for i, word in enumerate(line):
                sentence[paddings + i] = self.word_dict[word]
            assert paddings + i + 1 == time_steps
            data_sentences.append(sentence)

        data_labels = all_code
        data_sentences = np.array(data_sentences, dtype=np.float32)
        doc_embeddings = np.array(doc_embeddings, dtype=np.float32)
        data_labels = np.array(data_labels)
        if shuffle:
            data_sentences, doc_embeddings, data_labels = self.shuffle(
                data_sentences,
                doc_embeddings,
                data_labels)

        split_line = int(training_rate * len(data_sentences))
        self.train_sentences = data_sentences[:split_line]
        self.train_docs = doc_embeddings[:split_line]
        self.train_labels = data_labels[:split_line]

        self.test_sentences = data_sentences[split_line:]
        self.test_docs = doc_embeddings[split_line:]
        self.test_labels = data_labels[split_line:]

    def batch_iter(self, batch_size):
        for iter, index in enumerate(range(0, len(self.train_labels), batch_size)):
            batch_x = self.train_sentences[index:index + batch_size]
            batch_y = self.train_labels[index:index + batch_size]
            batch_docx = self.train_docs[index:index + batch_size]
            batch_y = self.change_multi_hot(batch_y)
            yield batch_x, batch_docx, batch_y

    def shuffle(self, data_sentences=None, doc_embeddings=None, data_labels=None):
        if data_sentences is None:
            indices = list(range(len(self.train_labels)))
            random.shuffle(indices)
            self.train_sentences = self.train_sentences[indices]
            self.train_labels = self.train_labels[indices]
            self.train_docs = self.train_docs[indices]

        else:
            indices = list(range(len(data_sentences)))
            random.shuffle(indices)
            data_sentences = data_sentences[indices]
            data_labels = data_labels[indices]
            doc_embeddings = doc_embeddings[indices]

            return data_sentences, doc_embeddings, data_labels

    def change_multi_hot(self, batch_y):
        multi_hot_label = np.zeros([len(batch_y), class_num], dtype=np.int32)
        for i, code_set in enumerate(batch_y):
            for code in code_set:
                code = self.code_dict[code]
                multi_hot_label[i, code] = 1
        return multi_hot_label


if __name__ == '__main__':
    master = data_master()
    print(master.test_sentences)
    print(master.test_labels)
    print(len(master.test_labels))
