# encoding=utf-8
import tensorflow as tf
import data_input

# weight initialize
# dropout

# master = data_input.data_master()


embedding_size = 100
doc_embedding_size = 128
doc_hidden_size = 64
time_steps = 600
class_num = 6984
mesh_class_num = 27677

filter_num = 96
learning_rate = 0.005
# fixed size 3
filter_sizes = [2, 3, 4]
threshold = 0.20


class FusedModel(object):
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
        self.x = tf.placeholder(tf.int32, [None, time_steps])
        x_emb = tf.nn.embedding_lookup(W, self.x)
        self.doc_x = tf.placeholder(tf.float32, [None, doc_embedding_size])
        self.y = tf.placeholder(tf.float32, [None, class_num])
        self.y_t = tf.placeholder(tf.float32, [None, mesh_class_num])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        x_convs = self.multi_conv(x_emb, weights, biases)
        print('after multiply convolutions: ', x_convs)
        x_convs = tf.reshape(x_convs, [-1, 3 * filter_num])
        # name="dW",
        dW = tf.Variable(tf.truncated_normal([doc_embedding_size, doc_hidden_size], stddev=0.1),
                         name="doc_embedding",
                         dtype=tf.float32)
        doc_x_ = tf.matmul(self.doc_x, dW)

        ones = tf.ones_like(self.y)
        zeros = tf.zeros_like(self.y)

        with tf.name_scope("D2V_Part"):
            doc_x_ = tf.nn.dropout(doc_x_, self.dropout_keep_prob)
            weight_d2v = tf.Variable(tf.truncated_normal([doc_hidden_size, class_num], stddev=0.1))
            biase_d2v = tf.Variable(tf.truncated_normal([class_num], stddev=0.1))
            logits_d2v = tf.matmul(doc_x_, weight_d2v) + biase_d2v
            self.loss_d2v = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits_d2v))
            self.optimizer_d2v = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_d2v)
            self.score_d2v = tf.nn.sigmoid(logits_d2v)
            self.prediction_d2v = tf.cast(tf.where(tf.greater(self.score_d2v, threshold), ones, zeros), tf.int32)

        x_convs = tf.nn.dropout(x_convs, self.dropout_keep_prob)
        with tf.name_scope("CNN_Part_Transfer"):

            weight_cnn = tf.Variable(tf.truncated_normal([3 * filter_num, mesh_class_num], stddev=0.1))
            biase_cnn = tf.Variable(tf.truncated_normal([mesh_class_num], stddev=0.1))
            logits_cnn = tf.matmul(x_convs, weight_cnn) + biase_cnn
            self.loss_cnn_t = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_t, logits=logits_cnn))
            self.optimizer_cnn_t = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_cnn_t)
            self.score_cnn_t = tf.nn.sigmoid(logits_cnn)
            ones_t = tf.ones_like(self.y_t)
            zeros_t = tf.zeros_like(self.y_t)
            self.prediction_cnn_t = tf.cast(tf.where(tf.greater(self.score_cnn_t, threshold), ones_t, zeros_t), tf.int32)

        with tf.name_scope("CNN_Part"):
            weight_cnn = tf.Variable(tf.truncated_normal([3 * filter_num, class_num], stddev=0.1))
            biase_cnn = tf.Variable(tf.truncated_normal([class_num], stddev=0.1))
            logits_cnn = tf.matmul(x_convs, weight_cnn) + biase_cnn
            self.loss_cnn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits_cnn))
            self.optimizer_cnn = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_cnn)
            self.score_cnn = tf.nn.sigmoid(logits_cnn)
            self.prediction_cnn = tf.cast(tf.where(tf.greater(self.score_cnn, threshold), ones, zeros), tf.int32)

        with tf.name_scope("Fused_Part"):
            logits_fused = logits_cnn + logits_d2v
            self.loss_fused = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits_fused))
            self.optimizer_fused = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_fused)
            self.score_fused = tf.nn.sigmoid(logits_fused)
            self.prediction_fused = tf.cast(tf.where(tf.greater(self.score_fused, threshold), ones, zeros), tf.int32)

    def conv1d(sef, x, W, b):
        x = tf.reshape(x, shape=[-1, time_steps, embedding_size])
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
