# generate ppgs from wav batch
# usage: python3 generate_batch.py --wav_dir /path/to/wavs --ppg_dir /path/to/ppg
import tensorflow as tf
import numpy as np
import argparse
import time
import os

from tqdm import tqdm

from models import CnnDnnClassifier, DNNClassifier,CNNBLSTMCalssifier
from audio import load_wav, wav2mfcc

WAV_PATH ='/home/zhaoxt20/vae_tac_myself/exp_multi_Libritts_2020.4.15_same_speaker_dif_sentenAndVisualization/test_datas/WarcraftWavs/WindRunner' #'/home/zhaoxt20/vae_tac_myself/LibriTTS_training_data/wavs_16khz'#'/notebooks/projects/share/zhiling/wavs'
PPG_PATH = '/home/zhaoxt20/vae_tac_myself/exp_multi_Libritts_2020.4.15_same_speaker_dif_sentenAndVisualization/test_datas/WarcraftWavs/WindRunner_ppgs'#'/home/zhaoxt20/vae_tac_myself/LibriTTS_training_data/ppg_extracted'#'/notebooks/projects/share/zhiling/ppgs_luhui'
CKPT = '/home/zhaoxt20/ppgs_extractor-master/experiment/models/vqvae.ckpt-233000'#'./saved_models/vqvae.ckpt-343000'
MFCC_DIM = 39
PPG_DIM = 345

os.environ["CUDA_VISIBLE_DEVICES"]="6"
def read_inputs(mfcc_path):
    mfcc = np.load(mfcc_path)
    return mfcc


def get_arguments():
    parser = argparse.ArgumentParser(description="PPGs extractor inference script")
    parser.add_argument('--wav_dir', type=str, default=WAV_PATH)
    parser.add_argument('--ppg_dir', type=str, default=PPG_PATH)
    parser.add_argument('--ckpt', type=str, default=CKPT)
    return parser.parse_args()


def main():
    args = get_arguments()

    # validate directories
    wav_dir = args.wav_dir
    ppg_dir = args.ppg_dir
    if not os.path.isdir(wav_dir):
        raise ValueError('wav directory not exists!')
    if not os.path.isdir(ppg_dir):
        print('PPG save directory not exists! Create it as {}'.format(ppg_dir))
        os.makedirs(ppg_dir)

    # get wav file path list
    wav_list = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir)
                if f.endswith('.wav') or f.endswith('.WAV')]
    ppg_list = [os.path.join(ppg_dir, f.split('.')[0] + '.npy') for f in os.listdir(wav_dir)
                if f.endswith('.wav') or f.endswith('.WAV')]

    # Set up network
    mfcc_pl = tf.placeholder(dtype=tf.float32,
                             shape=[None, None, MFCC_DIM],
                             name='mfcc_pl')
    # classifier = DNNClassifier(out_dims=PPG_DIM, hiddens=[256, 256, 256],
    #                            drop_rate=0.2, name='dnn_classifier')
    # classifier = CnnDnnClassifier(out_dims=PPG_DIM, n_cnn=5,
    #                               cnn_hidden=64, dense_hiddens=[256, 256, 256])
    classifier = CNNBLSTMCalssifier(out_dims=PPG_DIM, n_cnn=3, cnn_hidden=256,
                                    cnn_kernel=3, n_blstm=2, lstm_hidden=128)
    predicted_ppgs = tf.nn.softmax(classifier(inputs=mfcc_pl)['logits'])

    # set up a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    # load saved model
    saver = tf.train.Saver()
    # sess.run(tf.global_variables_initializer())
    print('Restoring model from {}'.format(args.ckpt))
    saver.restore(sess, args.ckpt)
    for wav_f, ppg_f in tqdm(zip(wav_list, ppg_list)):
        wav_arr = load_wav(wav_f)
        mfcc = wav2mfcc(wav_arr)
        ppgs = sess.run(predicted_ppgs,
                        feed_dict={mfcc_pl: np.expand_dims(mfcc, axis=0)})
        np.save(ppg_f, np.squeeze(ppgs))
    duration = time.time() - start_time
    print("PPGs file generated in {:.3f} seconds".format(duration))
    sess.close()


if __name__ == '__main__':
    main()
