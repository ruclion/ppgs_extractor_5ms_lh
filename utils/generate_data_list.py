import os
import time

#Generate test/train file names from /tmp/luhui_share/alignments which contain ali.x.txt
#Generate a txt which contain name-only information.
#I get the first split in ali.x.txt which are audio names.

#MFCC_PATH ='/tmp/luhui_share/alignments'# #'../my_test/ppggroundtruth'#         ali.x.txt
MFCC_PATH = './LibritMFCC/'
def main():
    namelist=[]
    MFCCFile_list = os.listdir(MFCC_PATH)
    counter=0
    for second_dir in MFCCFile_list:
        for third_dir in os.listdir(os.path.join(MFCC_PATH,second_dir)):
            mfcc_dir = os.path.join(os.path.join(MFCC_PATH,second_dir),third_dir)
            third_wav_dir = os.path.join(os.path.join(MFCC_PATH,second_dir),third_dir)
            for wavs in os.listdir(third_wav_dir):
                namelist.append(wavs.split('.')[0])
            #print('Now in the '+mfcc_dir+' from '+ third_wav_dir)
            # if not os.path.exists(mfcc_dir):
            #     os.makedirs(mfcc_dir)
    # for file in MFCCFile_list:
    #     counter+=1
    #     print(counter)
    #     with open(os.path.join(MFCC_PATH,file)) as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             namelist.append(line.split(' ')[0])
    from random import shuffle
    shuffle(namelist)
    print('in total,there are ' ,len(namelist),' in list')
    #time.sleep(5)
    with open('./train.txt','w') as f:
        for i in range(int(len(namelist)*0.8)):
            f.write(namelist[i])
            f.write('\n')


    with open('./test.txt','w') as f:
        i = int(len(namelist)*0.8)
        while i<len(namelist):
            f.write(namelist[i])
            f.write('\n')
            i+=1

#timit_dataset中，不会对数据进行随机排序。所以这样写进去，使得每次拿到的batch都属于一个人 ，会造成训练的抖动
if __name__ == '__main__':
    main()