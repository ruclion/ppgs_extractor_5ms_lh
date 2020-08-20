import tensorflow as tf
from tensorflow import keras


class BiGRUlayer(object):
    def __init__(self, hidden, name='bi-gru'):
        self.hidden = hidden
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.bigru_layer = keras.layers.Bidirectional(
                keras.layers.GRU(units=self.hidden,
                                 return_sequences=True),
                merge_mode='concat'
            )

    def __call__(self, inputs, seq_lens=None):
        with tf.variable_scope(self.name):
            mask = tf.sequence_mask(seq_lens, dtype=tf.float32) \
                if seq_lens is not None else None
            return self.bigru_layer(inputs, mask=mask)


class BLSTMlayer(object):
    def __init__(self, hidden, name='blstm'):
        self.hidden = hidden
        self.name = name
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.blstm_layer = keras.layers.Bidirectional(
                keras.layers.CuDNNLSTM(units=self.hidden,
                                       return_sequences=True),
                merge_mode='concat'
            )

    def __call__(self, inputs, seq_lens=None):
        with tf.variable_scope(self.name):
            #mask = tf.sequence_mask(seq_lens, dtype=tf.float32) \
            #    if seq_lens is not None else None
            return self.blstm_layer(inputs)
