# encoding=utf-8
import tensorflow as tf
from tensorflow.contrib import layers

embedding_size = 128

cnn_feature_size = 128

sequence_lens = 700
class_num = 6984
filter_num = 96
learning_rate = 0.005
# fixed size 3
filter_sizes = [2, 3, 4]
threshold = 0.2


class TextCNN(object):
    def __init__(self, embeddings):
        weights = {
            'wc1': tf.Variable(tf.truncated_normal([filter_sizes[0], embedding_size, filter_num], stddev=0.1)),
            'wc2': tf.Variable(tf.truncated_normal([filter_sizes[1], embedding_size, filter_num], stddev=0.1)),
            'wc3': tf.Variable(tf.truncated_normal([filter_sizes[2], embedding_size, filter_num], stddev=0.1))
        }

        biases = {
            'bc1': tf.Variable(tf.truncated_normal([filter_num], stddev=0.1)),
            'bc2': tf.Variable(tf.truncated_normal([filter_num], stddev=0.1)),
            'bc3': tf.Variable(tf.truncated_normal([filter_num], stddev=0.1))
        }

        # define placehold
        W = tf.Variable(embeddings, name="W", dtype=tf.float32)
        self.x = tf.placeholder(tf.int32, [None, sequence_lens])
        x_emb = tf.nn.embedding_lookup(W, self.x)
        self.y = tf.placeholder(tf.float32, [None, class_num])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        x_convs = self.multi_conv(x_emb, weights, biases)
        print('after multiply convolutions: ', x_convs)
        x_convs = tf.reshape(x_convs, [-1, 3 * filter_num])

        with tf.name_scope("CNN_Part"):
            x_convs = tf.nn.dropout(x_convs, self.dropout_keep_prob)
            print('x_convs:', x_convs)
            # logits_cnn = layers.fully_connected(x_convs, cnn_feature_size,
            #                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            #                                     biases_initializer=tf.truncated_normal_initializer(stddev=0.1),
            #                                     activation_fn=None)

            output = layers.fully_connected(x_convs, class_num,
                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            biases_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            activation_fn=None)

            self.loss_cnn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=output))
            self.optimizer_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_cnn)
            self.score_cnn = tf.nn.sigmoid(output)
            ones = tf.ones_like(self.score_cnn)
            zeros = tf.zeros_like(ones)
            self.prediction_cnn = tf.cast(tf.where(tf.greater(self.score_cnn, threshold), ones, zeros), tf.int32)

    def conv1d(sef, x, W, b):
        x = tf.reshape(x, shape=[-1, sequence_lens, embedding_size])
        x = tf.nn.conv1d(x, W, 1, padding='SAME')
        x = tf.nn.bias_add(x, b)
        # shape=(n,time_steps,filter_num)
        h = tf.nn.relu(x)

        print('conv size:', h.get_shape().as_list())

        pooled = tf.reduce_max(h, axis=1)
        print('pooled size:', pooled.get_shape().as_list())
        return pooled

    def multi_conv(self, x, weights, biases):
        # Convolution Layer
        conv1 = self.conv1d(x, weights['wc1'], biases['bc1'])
        conv2 = self.conv1d(x, weights['wc2'], biases['bc2'])
        conv3 = self.conv1d(x, weights['wc3'], biases['bc3'])
        #  n*time_steps*(3*filter_num)
        convs = tf.concat([conv1, conv2, conv3], 1)
        return convs
