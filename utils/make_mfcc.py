import argparse
import os
import numpy as np
from audio import wav2mfcc_v2, load_wav


hparams = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 400,
    'hop_length': 80,
    'win_length': 400,
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 30.,
    'fmax': 7600.,
    'ref_db': 20,  #
    'min_db': -80.0,  # restrict the dynamic range of log power
    'iterations': 100,  # griffin_lim #iterations
    'silence_db': -28.0,
    'center': False,
}
# hparams = {
#     'sample_rate': 16000,
#     'preemphasis': 0.97,
#     'n_fft': 400,
#     'hop_length': 160,
#     'win_length': 400,
#     'num_mels': 80,
#     'n_mfcc': 13,
#     'window': 'hann',
#     'fmin': 30.,
#     'fmax': 7600.,
#     'ref_db': 20,  #
#     'min_db': -80.0,  # restrict the dynamic range of log power
#     'iterations': 100,  # griffin_lim #iterations
#     'silence_db': -28.0,
#     'center': False,
# }


def main():
    parser = argparse.ArgumentParser('MFCC extraction')
    parser.add_argument('--wav_dir', type=str, required=True)
    parser.add_argument('--mfcc_dir', type=str, required=True)
    args = parser.parse_args()
    wav_dir = args.wav_dir

    for second_dir in os.listdir(wav_dir):
        for third_dir in os.listdir(os.path.join(wav_dir,second_dir)):
            mfcc_dir = os.path.join(os.path.join(args.mfcc_dir,second_dir),third_dir)
            third_wav_dir = os.path.join(os.path.join(args.wav_dir,second_dir),third_dir)
            #print('Now in the '+mfcc_dir+' from '+ third_wav_dir)
            if not os.path.exists(mfcc_dir):
                os.makedirs(mfcc_dir)

            wav_files = [os.path.join(third_wav_dir, f) for f in os.listdir(third_wav_dir) if f.endswith('.wav')]
            print('Extracting MFCC from {} to {}...'.format(third_wav_dir, mfcc_dir))
            cnt = 0
            for wav_f in wav_files:
                wav_arr = load_wav(wav_f, sr=hparams['sample_rate'])
                mfcc_feats = wav2mfcc_v2(wav_arr, sr=hparams['sample_rate'],
                                         n_mfcc=hparams['n_mfcc'], n_fft=hparams['n_fft'],
                                         hop_len=hparams['hop_length'], win_len=hparams['win_length'],
                                         window=hparams['window'], num_mels=hparams['num_mels'],
                                         center=hparams['center'])
                save_name = wav_f.split('/')[-1].split('.')[0] + '.npy'
                save_name = os.path.join(mfcc_dir, save_name)
                np.save(save_name, mfcc_feats)
                cnt += 1
                print('Processed {} files'.format(cnt), end='\r')
    # mfcc_dir = args.mfcc_dir
    # if not os.path.exists(mfcc_dir):
    #     os.makedirs(mfcc_dir)
    # for wave_name in os.listdir(wav_dir):
    #
    #     wav_dir=os.path.join(wav_dir, wave_name)
    #     save_name = wav_dir.split('/')[-1].split('.')[0] + '.npy'
    #     save_name = os.path.join(mfcc_dir, save_name)
    #     print('Extracting MFCC from {} to {}...'.format(wav_dir,save_name))
    #     cnt = 0
    #     wav_arr = load_wav(wav_dir, sr=hparams['sample_rate'])
    #     mfcc_feats = wav2mfcc_v2(wav_arr, sr=hparams['sample_rate'],
    #                                      n_mfcc=hparams['n_mfcc'], n_fft=hparams['n_fft'],
    #                                      hop_len=hparams['hop_length'], win_len=hparams['win_length'],
    #                                      window=hparams['window'], num_mels=hparams['num_mels'],
    #                                      center=hparams['center'])
    #
    #
    #     np.save(save_name, mfcc_feats)
    #     cnt += 1
    #     print('Processed {} files'.format(cnt), end='\r')
    return


if __name__ == '__main__':
    main()
