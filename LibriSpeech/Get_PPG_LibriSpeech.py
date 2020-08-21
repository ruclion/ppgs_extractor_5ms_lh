import os
import numpy as np
# from audio import wav2mfcc_v2, load_wav



# wavs_dir = 'wavs'
alignments_dir = 'alignments'
ppgs_dir = 'PPGs'

def save_ppg_as_mfcc_path(idx, ppg):
    # print(idx, ppg, ppg.shape, ppg.dtype)
    # 100-121669-0000 1 1 1 1 1 1 1
    book_dir_path = idx.split('-')[0]
    chapter_dir_path = idx.split('-')[1]
    # segment_dir_path = id.split('-')[2]
    dir_path = os.path.join(ppgs_dir, os.path.join(book_dir_path, chapter_dir_path))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 1246-135815-0004.npy
    save_name = idx + '.npy'
    save_name = os.path.join(dir_path, save_name)
    np.save(save_name, ppg)

def main():
    cnt = 0
    for f_path in os.listdir(alignments_dir):
        if not f_path.endswith('.txt'):
            continue
        f = open(os.path.join(alignments_dir, f_path), 'r')
        a = f.readlines()
        for line in a:
            line = line.strip().split(' ')
            idx = line[0]
            ppg = np.asarray(line[1:])
            save_ppg_as_mfcc_path(idx, ppg)
            cnt += 1
            # break

        # break
    print(cnt)

    return


if __name__ == '__main__':
    main()
