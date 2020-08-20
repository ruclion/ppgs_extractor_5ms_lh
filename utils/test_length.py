import numpy as np
import os
from audio import load_wav

#test for my mfcc here 这是一个观察Kaldi生成的维度和我生成mfcc的维度是否一样

#检测结果，我生成的mfcc少一帧是因为我用的librispeech是转成wav格式的 而源文件是flac的格式
MFCC_PATH = './mfcc_zhao/103/1240'
MFCC_TO_PPGS = '../my_test/ppggroundtruth/ali.1.txt'
WAVE_PATH = '../my_test/1240/'
def mfccgenerate():
    with open(MFCC_TO_PPGS) as f:
        lists = f.readlines()
        for i in lists:
            yield i.split(' ')

def main():
    #mfcc_npys = os.listdir(MFCC_PATH)
    #wav_files = [os.path.join(WAVE_PATH, f) for f in os.listdir(WAVE_PATH) if f.endswith('.wav')]
    i=mfccgenerate()
    ppg = next(i)
    counter = 0
    correct = 0
    while ppg[0][0:8] == '103-1240':
        #print(ppg[0])
        counter+=1
        myppgnum = ppg[0][9:13]
        npyname = '103-1240' +'-' +myppgnum+'.npy'
        wavename='103-1240' + '-' + myppgnum +'.wav'
        npy = np.load(os.path.join(MFCC_PATH, npyname))
        npy2 = np.load(os.path.join('../utils/PPGs/103/1240/', npyname))
        wav_arr = load_wav(os.path.join(WAVE_PATH, wavename), sr=16000)
        if len(npy) == len(ppg[1:])-1:
            correct+=1
        else:
            print(npyname)
        print(len(npy),len(ppg[1:]),len(wav_arr),len(npy2))
        ppg = next(i)


    print('accuracy:'+str(correct / counter))

if __name__=='__main__':
    main()