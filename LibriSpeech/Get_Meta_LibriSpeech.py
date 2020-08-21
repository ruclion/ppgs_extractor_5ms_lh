import os
# import numpy as np
# from audio import wav2mfcc_v2, load_wav



wavs_dir = 'wavs'
ppgs_dir = 'alignments'
zhaoxt_train = 'train.txt'
zhaoxt_test = 'test.txt'

meta_list_fromWavs = []
meta_list_fromPPGs = []
meta_list_fromZhaoxt = []
meta_list = []
meta_path = 'meta.txt'

def main():
    # 2391-145015-0048
    f = open(zhaoxt_train, 'r')
    a = [t.strip() for t in f.readlines()]
    meta_list_fromZhaoxt.extend(a)
    f = open(zhaoxt_test, 'r')
    a = [t.strip() for t in f.readlines()]
    meta_list_fromZhaoxt.extend(a)
    print('Zhaoxts:', len(meta_list_fromZhaoxt), meta_list_fromZhaoxt[0])

    # wavs
    for second_dir in os.listdir(wavs_dir):
        for third_dir in os.listdir(os.path.join(wavs_dir,second_dir)):
            third_wavs_dir = os.path.join(os.path.join(wavs_dir,second_dir),third_dir)
            wav_files = [f[:-4] for f in os.listdir(third_wavs_dir) if f.endswith('.wav')]
            # print('Extracting MFCC from {}...'.format(third_wavs_dir))
            meta_list_fromWavs.extend(wav_files)
    print('Wavs:', len(meta_list_fromWavs), meta_list_fromWavs[0])

    # 100-121669-0000 1 1 1 1 1 1 1
    for f_path in os.listdir(ppgs_dir):
        f = open(os.path.join(ppgs_dir, f_path), 'r')
        a = f.readlines()
        for line in a:
            line = line.strip().split(' ')
            meta_list_fromPPGs.append(line[0])
    print('PPGs:', len(meta_list_fromPPGs), meta_list_fromPPGs[0])

    # 主要用欣陶的list，辅助看看wavs和ppgs有没有；会跑1分钟，也就暴力看又没有了
    for idx in meta_list_fromZhaoxt:
        if idx in meta_list_fromPPGs and idx in meta_list_fromWavs:
            meta_list.append(idx)
        else:
            print('为什么不用:', idx)
        
        # break

    f = open(meta_path, 'w')
    for idx in meta_list:
        f.write(idx + '\n')
    return


if __name__ == '__main__':
    main()
