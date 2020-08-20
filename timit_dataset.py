import os
import numpy as np
import time

#生成数据的代码
#train/test每一行都只是一个文件名
TRAIN_FILE = './small_train.txt'#'/media/luhui/experiments_data/librispeech/train.txt'
TEST_FILE = './small_train.txt'#'/media/luhui/experiments_data/librispeech/dev.txt'
MFCC_DIR =  './utils/mfcc_zhao'      #'/media/luhui/experiments_data/librispeech/mfcc_hop12.5'#生成MFCC的目录
PPG_DIR =   './utils/PPGs'   #'/media/luhui/experiments_data/librispeech/phone_labels_hop12.5'
MFCC_DIM = 39
PPG_DIM = 345


def text2list(file):
    file_list = []
    with open(file, 'r') as f:
        for line in f:
            file_list.append(line.split()[0])
    return file_list


def onehot(arr, depth, dtype=np.float32):
    assert len(arr.shape) == 1 #不为1则异常
    onehots = np.zeros(shape=[len(arr), depth], dtype=dtype)
    arr=arr.astype(np.int64)

    arr = arr-1  #下标从0开始
    arr = arr.tolist()#不知为何，array类型无法遍历

    onehots[np.arange(len(arr)), arr] = 1
    return onehots


def get_single_data_pair(fname, mfcc_dir, ppg_dir):
    assert os.path.isdir(mfcc_dir) and os.path.isdir(ppg_dir)

    mfcc_f = os.path.join(os.path.join(os.path.join(mfcc_dir, fname.split('-')[0]),fname.split('-')[1]),fname+'.npy')#fname+'.npy')
    ppg_f = os.path.join(os.path.join(os.path.join(ppg_dir, fname.split('-')[0]),fname.split('-')[1]),fname+'.npy')#os.path.join(ppg_dir, fname+'.npy')

   # print(mfcc_f)
    #print(ppg_f)
    #time.sleep(10)
    mfcc = np.load(mfcc_f)
    # cut the MFCC into the same time length as PPGs
    # mfcc = mfcc[2:mfcc.shape[0]-3, :]
    ppg = np.load(ppg_f)
    ppg = onehot(ppg, depth=PPG_DIM)
    assert mfcc.shape[0] == ppg.shape[0],fname+' 维度不相等'
    return mfcc, ppg


def train_generator():
    file_list = text2list(file=TRAIN_FILE)
    for f in file_list:

        mfcc, ppg = get_single_data_pair(f, mfcc_dir=MFCC_DIR, ppg_dir=PPG_DIR)
        yield mfcc, ppg, mfcc.shape[0]


def test_generator():
    file_list = text2list(file=TEST_FILE)
    for f in file_list:
        mfcc, ppg = get_single_data_pair(f, mfcc_dir=MFCC_DIR, ppg_dir=PPG_DIR)
        yield mfcc, ppg, mfcc.shape[0]#mfcc数目不固定（帧数目不同）


def tf_dataset():
    import tensorflow as tf
    batch_size = 128
    train_set = tf.data.Dataset.from_generator(train_generator,
                                               output_types=(
                                                   tf.float32, tf.float32, tf.int32),
                                               output_shapes=(
                                                   [None, MFCC_DIM], [None, PPG_DIM], []))#[]代表输出一个数，不能加数字
    train_set = train_set.padded_batch(batch_size,#会自动给数据加一维，batch_size大小。然后把每个data组的东西添加进去
                                       padded_shapes=([None, MFCC_DIM],
                                                      [None, PPG_DIM],#表示在批处理之前每个输入元素的各个组件应填充到的形状。None表示在这一维度上填充至这维度最大的值
                                                      [])).repeat()#第三个是[]，因为每个data第三个是常数，一个batch会变成(batchsize,)的形状，不需要填充。
    #为什么要用padded_batch？the input elements to be batched may have different shape
    train_iterator = train_set.make_initializable_iterator()
    batch_data = train_iterator.get_next()

    with tf.Session() as sess:
        sess.run(train_iterator.initializer)#
        for i in range(10):
            data = sess.run(batch_data)#从迭代器取一值
            print(data[0].shape, data[1].shape, data[2].shape)
    return


if __name__ == '__main__':
    tf_dataset()
    # file_list = text2list(file=TEST_FILE)
    # for f in file_list:
    #     mfcc, ppg = get_single_data_pair(f, mfcc_dir=MFCC_DIR, ppg_dir=PPG_DIR)
    #     print(mfcc.shape, ppg.shape)
