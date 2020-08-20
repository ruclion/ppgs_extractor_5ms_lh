import numpy as np
import os
#from audio import load_wav
#测试自己生成的mfcc和ppgalignment是否长度一样

#test for my mfcc here
MFCC_PATH = './LibritMFCC'
PPGS = './PPGS'#

#some of the ppg and the mfcc are not eual in length but most of them are eual.. why?
#7067-76048-0021.npy is error:not eualized length
#only this have not eual mfcc and ppg
#so i deleted it
#MFCC_PATH:Contain all the mfccs obtained by my net. test all the mfcc and the PPG from Kaldi in order to confirm that
#All the MFCCs have same dims with PPG from Kaldi
#MFCC_PATH/Speaker/sentence/Speaker-sentence-000x.npy is the MFCC path。
#PPGS: Is similar to MFCC_PATH, contain PPGs from Kaldi and have the same root name.
#PPGS/SPeaker/sentence/Speaker-sentence-000x.npy is the PPG path。

def main():



    counter = 0
    correct = 0
    for speaker in os.listdir(MFCC_PATH):
         sentencespath = os.path.join(MFCC_PATH,speaker)
         ppgsentencepath = os.path.join(PPGS,speaker)
         for sentence in os.listdir(sentencespath):
            sentenceroot = os.path.join(sentencespath,sentence)
            ppgsentenceroot = os.path.join(ppgsentencepath,sentence)
            for file in os.listdir(sentenceroot):
                counter += 1

                npy = np.load(os.path.join(sentenceroot, file))
                npy2 = np.load(os.path.join(ppgsentenceroot, file))


        #print(ppg[0])
                if len(npy) == len(npy2):
                    correct+=1


                else:
                    print(file +' is error:not eualized length')
                    print(len(npy), len(npy2))


    print('accuracy:'+str(correct / counter))

if __name__=='__main__':
    main()