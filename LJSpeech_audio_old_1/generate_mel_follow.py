import numpy as np
import os
from tqdm import tqdm
from audio import load_wav, wav2mfcc, spectrogram, wav2melspec, griffin_lim

MEL_DIM = 80

def wav2linear_for_ppg_cbhg(wav_arr):
    # Before Norm, maybe use power, but this code use 绝对值
    return spectrogram(wav_arr)['magnitude']


def wav2mel_for_ppg_cbhg(wav_arr):
    # 用的平方值,输出的是没有取log的mel原始值
    return wav2melspec(wav_arr)


def main():
    data_dir = '.'
    wav_dir = os.path.join(data_dir, 'wavs_16000')
    mel_dir = os.path.join(data_dir, 'mel_from_generate_batch_follow')
    if not os.path.isdir(wav_dir):
        raise ValueError('wav directory not exists!')
    if not os.path.isdir(mel_dir):
        print('Linear save directory not exists! Create it as {}'.format(mel_dir))
        os.makedirs(mel_dir)

    # get wav file path list
    wav_list = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir)
                if f.endswith('.wav') or f.endswith('.WAV')]

    mel_list = [os.path.join(mel_dir, f.split('.')[0] + '.npy') for f in os.listdir(wav_dir)
                if f.endswith('.wav') or f.endswith('.WAV')]

    for wav_f, mel_f in tqdm(zip(wav_list, mel_list)):
        wav_arr = load_wav(wav_f)
        mel = wav2mel_for_ppg_cbhg(wav_arr)
        np.save(mel_f, mel) 
        # break


if __name__ == '__main__':
    main()
