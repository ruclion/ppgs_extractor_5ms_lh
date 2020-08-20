import tensorflow as tf
import numpy as np
from timit_dataset import test_generator
#
import os
from models import CNNBLSTMCalssifier
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True


Model_Path = '/home/zhaoxt20/ppgs_extractor-master/experiment/models/'   #'./utils/models/'

MFCC_DIM = 39
PPG_DIM = 345

os.environ["CUDA_VISIBLE_DEVICES"]="1"


def main():

    test_set = tf.data.Dataset.from_generator(test_generator,
                                              output_types=(
                                                  tf.float32, tf.float32, tf.int32),
                                              output_shapes=(
                                                  [None, MFCC_DIM], [None, PPG_DIM], []))

    # 设置repeat()，在get_next循环中，如果越界了就自动循环。不计上限
    test_set = test_set.padded_batch(1,#args.batch_size,
                                     padded_shapes=([None, MFCC_DIM],
                                                    [None, PPG_DIM],
                                                    []))#.repeat()
    test_iterator = test_set.make_initializable_iterator()

    batch_data = test_iterator.get_next()

    classifier = CNNBLSTMCalssifier(out_dims=PPG_DIM, n_cnn=3, cnn_hidden=256,
                                    cnn_kernel=3, n_blstm=2, lstm_hidden=128)

    results_dict = classifier(batch_data[0], batch_data[1], batch_data[2])#如果是生成，则batch1和batch2就不需要了
    predicted = tf.nn.softmax(results_dict['logits'])
    mask = tf.sequence_mask(batch_data[2], dtype=tf.float32)
    accuracy = tf.reduce_sum(#如果是生成，则不需要accuracy了
        tf.cast(  # bool转float
            tf.equal(tf.argmax(predicted, axis=-1),  # 比较每一行的最大元素
                     tf.argmax(batch_data[1], axis=-1)),
            tf.float32) * mask  # 乘上mask，是因为所有数据都被填充为最多mfcc的维度了。所以填充部分一定都是相等的，于是需要将其mask掉。
    ) / tf.reduce_sum(tf.cast(batch_data[2], dtype=tf.float32))

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('predicted',
                     tf.expand_dims(
                         tf.transpose(predicted, [0, 2, 1]),
                         axis=-1), max_outputs=1)
    tf.summary.image('groundtruth',
                     tf.expand_dims(
                         tf.cast(
                             tf.transpose(batch_data[1], [0, 2, 1]),
                             tf.float32),
                         axis=-1), max_outputs=1)

    loss = results_dict['cross_entropy']

    tf.summary.scalar('cross_entropy', loss)



    #saver = tf.train.import_meta_graph(Model_Path + 'vqvae.ckpt-62000.meta')  # 读取图结构
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:

        sess.run(test_iterator.initializer)
        sess.run(init)
        saver.restore(sess, Model_Path+'vqvae.ckpt-233000')
        print("start")
        a= 0
        count=0
        max = 0
        min = 120
        while a<800:#True:
            a=a+1
            acc = sess.run(accuracy)
            if max < acc:
                max = acc
            if min > acc:
                min = acc
            count = count + sess.run(accuracy)

            #print(sess.run(accuracy))
            #a = sess.run(predicted)
            #np.savetxt('./answer.txt',sess.run(predicted),)
            #print(saver)

        print("max: " + str(max))
        print("min: " + str(min))
        print("average:" + str(count/a))

if  __name__ =='__main__':
        main()