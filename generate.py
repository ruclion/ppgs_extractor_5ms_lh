import tensorflow as tf
import numpy as np
import argparse
import time


from models import DNNClassifier
import os

MFCC_PATH = '/home/zhaoxt20/vae_tac_myself/mfcc/911_130578_000006_000000.npy'#'/data/data/vc_data/zhiling'
SAVE_NAME = '/home/zhaoxt20/vae_tac_myself/ppg_extracted/911_130578_000006_000000.npy'
CKPT = '/home/zhaoxt20/ppgs_extractor-master/experiment/models/vqvae.ckpt-233000'#'./saved_models/vqvae.ckpt-99000'
MFCC_DIM = 39
PPG_DIM = 345

os.environ["CUDA_VISIBLE_DEVICES"]="1"
def read_inputs(mfcc_path):
    mfcc = np.load(mfcc_path)
    return mfcc


def get_arguments():
    parser = argparse.ArgumentParser(description="PPGs extractor inference script")
    parser.add_argument('--mfcc_path', type=str, default=MFCC_PATH)
    parser.add_argument('--save_name', type=str, default=SAVE_NAME)
    parser.add_argument('--ckpt', type=str, default=CKPT)
    return parser.parse_args()


def main():
    args = get_arguments()

    # load data
    mfcc = np.load(args.mfcc_path)

    # Set up network
    mfcc_pl = tf.placeholder(dtype=tf.float32,
                             shape=[None, MFCC_DIM],
                             name='mfcc_pl')
    classifier = DNNClassifier(out_dims=PPG_DIM, hiddens=[256, 256, 256],
                               drop_rate=0.2, name='dnn_classifier')
    predicted_ppgs = classifier(mfcc_pl, use_dropout=False)['logits']

    # set up a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    start_time = time.time()

    # load saved model
    saver = tf.train.Saver(tf.trainable_variables())
    # sess.run(tf.global_variables_initializer())
    print('Restoring model from {}'.format(args.ckpt))
    saver.restore(sess, args.ckpt)

    ppgs = sess.run(predicted_ppgs, feed_dict={mfcc_pl: mfcc})
    np.save(args.save_name, ppgs)
    duration = time.time() - start_time
    print("PPGs file generated in {:.3f} seconds".format(duration))
    sess.close()


if __name__ == '__main__':
    main()
