import os
import numpy as np
from tqdm import tqdm
from audio import hparams as audio_hparams
from audio import load_wav, wav2unnormalized_mfcc, wav2normalized_db_mel, wav2normalized_db_spec
from audio import write_wav, normalized_db_mel2wav, normalized_db_spec2wav


# 超参数个数：16
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
    'ref_db': 20,  
    'min_db': -80.0,  
    'griffin_lim_power': 1.5,
    'griffin_lim_iterations': 60,  
    'silence_db': -28.0,
    'center': True,
}


assert hparams == audio_hparams


# meta_path = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/meta_small.txt'
meta_path = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/meta.txt'
wav_dir = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/wavs_16000'
ppg_dir = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/ppg_from_generate_batch'

mfcc_dir = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/mfcc_5ms_by_audio_2'
mel_dir = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/mel_5ms_by_audio_2'
spec_dir = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/spec_5ms_by_audio_2'
rec_wav_dir = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/rec_wavs_16000'

assert os.path.exists(wav_dir) is True
assert os.path.exists(ppg_dir) is True

# assert os.path.exists(mfcc_dir) is False and os.path.exists(mel_dir) is False and os.path.exists(spec_dir) is False
# assert os.path.exists(rec_wav_dir) is False
os.makedirs(mfcc_dir, exist_ok=True)
os.makedirs(mel_dir, exist_ok=True)
os.makedirs(spec_dir, exist_ok=True)
os.makedirs(rec_wav_dir, exist_ok=True)

good_meta_path = '/datapool/home/hujk17/chenxueyuan/DataBaker_Bilingual_CN/meta_good.txt'
f_good_meta = open(good_meta_path, 'w')

def main():
    #这一部分用于处理LJSpeech格式的数据集
    a = open(meta_path, 'r').readlines()
    b = []
    i = 0
    while i < len(a):
        t = a[i][0:6]
        b.append(t)
        i += 2
    print(b[:2])
    a = b
    # a = [i.strip().split('|')[0] for i in a]
    cnt = 0
    cnt_list = []
    bad_cnt = 0
    bad_list = []
    for fname in tqdm(a):
        try:
            # 提取声学参数
            wav_f = os.path.join(wav_dir, fname + '.wav')
            wav_arr = load_wav(wav_f)
            mfcc_feats = wav2unnormalized_mfcc(wav_arr)
            mel_feats = wav2normalized_db_mel(wav_arr)
            spec_feats = wav2normalized_db_spec(wav_arr)
            
            # 验证声学参数提取的对
            save_name = fname + '.npy'
            save_mel_rec_name = fname + '_mel_rec.wav'
            save_spec_rec_name = fname + '_spec_rec.wav'
            # 这句话有可能错，不知道为什么，可能是服务器临时变动有关
            ppg_already_feats = np.load(os.path.join(ppg_dir, save_name))

            assert ppg_already_feats.shape[0] == mfcc_feats.shape[0]
            assert mfcc_feats.shape[0] == mel_feats.shape[0] and mel_feats.shape[0] == spec_feats.shape[0]
            write_wav(os.path.join(rec_wav_dir, save_mel_rec_name), normalized_db_mel2wav(mel_feats))
            write_wav(os.path.join(rec_wav_dir, save_spec_rec_name), normalized_db_spec2wav(spec_feats))
            
            # 存储声学参数
            mfcc_save_name = os.path.join(mfcc_dir, save_name)
            mel_save_name = os.path.join(mel_dir, save_name)
            spec_save_name = os.path.join(spec_dir, save_name)
            np.save(mfcc_save_name, mfcc_feats)
            np.save(mel_save_name, mel_feats)
            np.save(spec_save_name, spec_feats)

            f_good_meta.write(fname + '\n')
            cnt_list.append(fname)
            cnt += 1
        except:
            bad_list.append(fname)
            bad_cnt += 1
        
        # print(cnt)
        # break

    print(cnt)
    print('bad:', bad_cnt)
    print(bad_list)

    return


if __name__ == '__main__':
    main()
