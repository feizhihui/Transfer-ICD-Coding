# encoding=utf-8
import tensorflow as tf
import data_input2
from FusedModel2 import TextCNN
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

master = data_input2.data_master()

batch_size = 256
epoch_num_cnn = 30
keep_pro = 0.9

model = TextCNN(master.embeddings)


def validataion(model_prediction):
    # model.prediction_fused
    print('begin to test:')
    test_labels = master.change_multi_hot(master.test_labels)
    outputs = []
    for i in range(0, len(master.test_sentences), batch_size):
        test_sentences_batch = master.test_sentences[i:i + batch_size]
        test_labels_batch = test_labels[i:i + batch_size]
        output = sess.run(model_prediction,
                          feed_dict={model.x: test_sentences_batch,
                                     model.y: test_labels_batch, model.dropout_keep_prob: 1.0})
        outputs.append(output)

    outputs = np.concatenate(outputs, axis=0)

    MiP, MiR, MiF, P_NUM, T_NUM = micro_score(outputs, test_labels)
    print(">>>>>> Final Result:  PredictNum:%.2f, TrueNum:%.2f" % (P_NUM, T_NUM))
    print(">>>>>> Micro-Precision:%.3f, Micro-Recall:%.3f, Micro-F Measure:%.3f" % (MiP, MiR, MiF))
    MaP, MaR, MaF = macro_score(outputs, test_labels)
    print(">>>>>> Macro-Precision:%.3f, Macro-Recall:%.3f, Macro-F Measure:%.3f" % (MaP, MaR, MaF))


def micro_score(output, label):
    N = len(output)
    # K = len(output[0])
    total_P = np.sum(output)
    total_R = np.sum(label)
    TP = float(np.sum(output * label))
    MiP = TP / max(total_P, 1e-12)
    MiR = TP / max(total_R, 1e-12)
    MiF = 2 * MiP * MiR / (MiP + MiR)
    return MiP, MiR, MiF, total_P / N, total_R / N


def macro_score(output, label):
    total_P = np.sum(output, axis=0)
    total_R = np.sum(label, axis=0)
    TP = np.sum(output * label, axis=0)
    MiP = np.mean(TP / np.maximum(total_P, 1e-12))
    MiR = np.mean(TP / np.maximum(total_R, 1e-12))
    MiF = 2 * MiP * MiR / (MiP + MiR)
    return MiP, MiR, MiF


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('pretraining CNN Part')
    for epoch in range(epoch_num_cnn):
        master.shuffle()
        for iter, (batch_x, batch_y) in enumerate(master.batch_iter(batch_size)):
            loss_fetch, output, _ = sess.run([model.loss_cnn, model.prediction_cnn, model.optimizer_cnn],
                                             feed_dict={model.x: batch_x, model.y: batch_y,
                                                        model.dropout_keep_prob: keep_pro})
            if iter % 100 == 0:
                print("===CNNPart===")
                MiP, MiR, MiF, P_NUM, T_NUM = micro_score(output, batch_y)
                print("epoch:%d  iter:%d, mean loss:%.3f,  PNum:%.2f, TNum:%.2f" % (
                    epoch + 1, iter + 1, loss_fetch, P_NUM, T_NUM))
                print("Micro-Precision:%.3f, Micro-Recall:%.3f, Micro-F Measure:%.3f" % (MiP, MiR, MiF))
        if epoch >= 10:
            validataion(model.prediction_cnn)
