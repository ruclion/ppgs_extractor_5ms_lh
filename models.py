import tensorflow as tf
from tensorflow import keras


from modules import BLSTMlayer


class DNNClassifier(object):
    def __init__(self, out_dims, hiddens, drop_rate, name):
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.dense_layers = []
            for i, units in enumerate(hiddens):
                dense = keras.layers.Dense(units, activation=tf.nn.sigmoid,
                                           name='dense{}'.format(i))
                self.dense_layers.append(dense)
            self.dropout_layer = keras.layers.Dropout(rate=drop_rate)
            self.output_dense = keras.layers.Dense(units=out_dims)

    def __call__(self, inputs, use_dropout=True, labels=None, lengths=None):
        """
        :param inputs: [batch, time, in_dims]
        :param labels: [batch, time, out_dims], should be one-hot
        :param lengths: [batch]
        :return:
        """
        cur_layer = inputs
        for layer in self.dense_layers:
            cur_layer = layer(cur_layer)
            cur_layer = tf.cond(tf.constant(use_dropout, dtype=tf.bool),
                                lambda: self.dropout_layer(cur_layer),
                                lambda: cur_layer)
        logits = self.output_dense(cur_layer)  # [batch, time, out_dims]
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(labels, axis=-1),
            logits=logits) if labels is not None else None
        mask = tf.sequence_mask(lengths, dtype=tf.float32) if lengths is not None else 1.0
        cross_entropy = tf.reduce_mean(cross_entropy * mask) if cross_entropy is not None else None
        return {'logits': logits,  # [time, dims]
                'cross_entropy': cross_entropy}


class CnnDnnClassifier(object):
    def __init__(self, out_dims, n_cnn, cnn_hidden, dense_hiddens,
                 name='cnn_dnn_classifier'):
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('multi_scale_cnn'):
                self.conv_stack = []
                for i in range(n_cnn):
                    cnn_layer = keras.layers.Conv1D(filters=cnn_hidden,
                                                    kernel_size=i + 1,
                                                    padding='same',
                                                    name='cnn{}'.format(i + 1))
                    self.conv_stack.append(cnn_layer)
            self.pooling_layer = keras.layers.MaxPool1D(pool_size=2, strides=1,
                                                        padding='same')
            with tf.variable_scope('dnn_layers'):
                self.dense_layers = []
                for i, hidden in enumerate(dense_hiddens):
                    dense_layer = keras.layers.Dense(units=hidden,
                                                     activation=tf.nn.sigmoid,
                                                     name='dense{}'.format(i))
                    self.dense_layers.append(dense_layer)
            self.output_layer = keras.layers.Dense(units=out_dims, activation=None,
                                                   name='output_dense')

    def __call__(self, inputs, labels=None, lengths=None):
        # 1. conv1d stack
        cnn_stack_outs = tf.concat([conv_layer(inputs) for conv_layer in self.conv_stack],
                                   axis=-1)
        # 2. maxpooling over time
        maxpool_outs = self.pooling_layer(cnn_stack_outs)

        # 3. DNN stack
        dnn_outs = maxpool_outs
        for dense in self.dense_layers:
            dnn_outs = dense(dnn_outs)

        # 4. output layer
        logits = self.output_layer(dnn_outs)

        # compute loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(labels, axis=-1),
            logits=logits) if labels is not None else None
        mask = tf.sequence_mask(lengths, dtype=tf.float32) if lengths is not None else 1.0
        cross_entropy = tf.reduce_mean(cross_entropy * mask) if cross_entropy is not None else None
        return {'logits': logits,  # [time, dims]
                'cross_entropy': cross_entropy}


class CNNBLSTMCalssifier(object):#使用这个
    def __init__(self, out_dims, n_cnn, cnn_hidden,
                 cnn_kernel, n_blstm, lstm_hidden,
                 name='cnn_blstm_classifier'):
        self.name = name
        with tf.variable_scope(self.name):
            self.cnn_layers = []
            for i in range(n_cnn):
                conv_layer = keras.layers.Conv1D(filters=cnn_hidden,
                                                 kernel_size=cnn_kernel,
                                                 strides=1, padding='same',
                                                 activation=tf.nn.relu,
                                                 name='conv_layer{}'.format(i))
                self.cnn_layers.append(conv_layer)
            self.blstm_layers = []
            for i in range(n_blstm):
                blstm_layer = BLSTMlayer(lstm_hidden,
                                         name='blstm_layer{}'.format(i))
                self.blstm_layers.append(blstm_layer)
            self.output_projection = keras.layers.Dense(units=out_dims)

    def __call__(self, inputs, labels=None, lengths=None):
        # 1. CNN block
        cnn_outs = inputs
        for layer in self.cnn_layers:
            cnn_outs = layer(cnn_outs)
        # 2. BLSTM layers
        blstm_outs = cnn_outs
        for layer in self.blstm_layers:
            blstm_outs = layer(blstm_outs, seq_lens=lengths)
        # 3. output projection
        logits = self.output_projection(blstm_outs)

        # 4. compute loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.argmax(labels, axis=-1),
            logits=logits) if labels is not None else None
        mask = tf.sequence_mask(lengths, dtype=tf.float32) if lengths is not None else 1.0
        cross_entropy = tf.reduce_mean(cross_entropy * mask) if cross_entropy is not None else None
        return {'logits': logits,  # [time, dims]
                'cross_entropy': cross_entropy}
