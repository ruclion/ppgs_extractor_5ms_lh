import os
import numpy as np
#/tmp/luhui_share/alignments 是txt文件存储的位置 这是一个存储ppgs groundtruth的文本

#from txt path we found all ali.x.txt
#for each ali.x.txt,i get the ppg vectors from it.
#each ppg vectors i generate a new npy file for it , as well as a similar path like:
#PPGS/speaker/sentence/speaker-sentence-number000x.npy 103-1240-0000.npy

#每一行是一个PPGs映射，但是比我们的mfcc多了一个维度

#我们应该给ppgs的最后一个维度舍弃掉



PPGS_SAVE_ROOT_PATH = './PPGS'


PPGS_TOTAL_PATH ='/home/zhaoxt20/MYDATA/libritts/ppg_alingment/alignments' #'../my_test/ppggroundtruth'#'/tmp/luhui_share/alignments'

if not os.path.exists(PPGS_SAVE_ROOT_PATH):
    os.mkdir(PPGS_SAVE_ROOT_PATH)


def main():
    PPGS_TXTlist = os.listdir(PPGS_TOTAL_PATH)

    for txt in PPGS_TXTlist:
        with open(os.path.join(PPGS_TOTAL_PATH,txt)) as f:
            lines = f.readlines()

            for line in lines:
                line = line.split(' ')
                line = line[:-1]
                print(len(line))
                PPGname = line[0]
                SAVEPATH =os.path.join(os.path.join(PPGS_SAVE_ROOT_PATH,PPGname.split('-')[0]),PPGname.split('-')[1])
                if not os.path.exists(SAVEPATH):
                    os.makedirs(SAVEPATH)
                np.save(os.path.join(SAVEPATH,PPGname+'.npy'),np.array(line[1:]))
                if PPGname.split('-')[0][0]=='2':
                    print(SAVEPATH+PPGname + '.npy has been saved!')





if __name__ == '__main__':
    main()

