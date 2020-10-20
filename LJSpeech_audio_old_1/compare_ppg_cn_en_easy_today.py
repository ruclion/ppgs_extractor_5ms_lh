import os
import numpy as np
import time

#生成数据的代码
#train/test每一行都只是一个文件名
TRAIN_FILE = 'LJSpeech-1.1_Norm_Sort/sorted_train.txt'#'/media/luhui/experiments_data/librispeech/train.txt'
# TEST_FILE = 'LJSpeech-1.1_Norm_Sort/sorted_test.txt'#'/media/luhui/experiments_data/librispeech/dev.txt'

# Linears_DIR = 'LJSpeech-1.1_Norm_Sort/norm_db_linear'      #'/media/luhui/experiments_data/librispeech/mfcc_hop12.5'#生成MFCC的目录
PPGs_DIR = 'LJSpeech-1.1_Norm_Sort/norm_ppg'   #'/media/luhui/experiments_data/librispeech/phone_labels_hop12.5'
# Linear_DIM = 201
PPG_DIM = 345


def en_text2list(file):
    file_list = []
    with open(file, 'r') as f:
        for line in f:
            # !!!!!!!!!!!!!!!!
            file_list.append(line.split('|')[0])
    return file_list


# 000001	那些#1庄稼#1田园#2在#1果果#1眼里#2感觉#1太亲切了#4
#	na4 xie1 zhuang1 jia5 tian2 yuan2 

def cn_text2list(file):
    file_list = []
    with open(file, 'r') as f:
        a = [i.strip() for i in f.readlines()]
        print(a[0])
        print(a[1])
        i = 0
        while i < len(a):
            fname = a[:6]
            file_list.append(fname)
            i += 2
    return file_list


# def onehot(arr, depth, dtype=np.float32):
#     assert len(arr.shape) == 1 #不为1则异常
#     onehots = np.zeros(shape=[len(arr), depth], dtype=dtype)
#     arr=arr.astype(np.int64)

#     arr = arr-1  #下标从0开始
#     arr = arr.tolist()#不知为何，array类型无法遍历

#     onehots[np.arange(len(arr)), arr] = 1
#     return onehots


def get_single_data_pair(fname, ppgs_dir, linears_dir):
    assert os.path.isdir(ppgs_dir) and os.path.isdir(linears_dir)

    # mfcc_f = os.path.join(os.path.join(os.path.join(mfcc_dir, fname.split('-')[0]),fname.split('-')[1]),fname+'.npy')#fname+'.npy')
    ppg_f = os.path.join(ppgs_dir, fname+'.npy')#os.path.join(ppg_dir, fname+'.npy')
    linear_f = os.path.join(linears_dir, fname+'.npy')#os.path.join(ppg_dir, fname+'.npy')

   # print(mfcc_f)
    #print(ppg_f)
    #time.sleep(10)
    # mfcc = np.load(mfcc_f)
    # cut the MFCC into the same time length as PPGs
    # mfcc = mfcc[2:mfcc.shape[0]-3, :]
    ppg = np.load(ppg_f)
    linear = np.load(linear_f)
    # ppg = onehot(ppg, depth=PPG_DIM)
    assert ppg.shape[0] == linear.shape[0],fname+' 维度不相等'
    return ppg, linear



def for_loop_en():
    file_list = text2list(file=TRAIN_FILE)
    en_ppgs_ls = []
    for f in file_list:
        wav_ppgs, linears = get_single_data_pair(f, ppgs_dir=PPGs_DIR, linears_dir=Linears_DIR)
        # 需要确认下
        # en_ppgs_ls.extend(list(wav_ppgs))
        # 或者
        for i in range(wav_ppgs.shape[0]):
            # ppg[i]
            en_ppgs_ls.append(wav_ppgs[i])
            # find_jin(ppg[i])
    # shuffule
    # wav_id, frame_id
    return en_ppgs_ls

def dist(ppg_e, ppg_c):
    # array, 345 dim
    assert ppg_c.shape[0] == 345
    ans = 0
    for i in range(345):
        ans += (ppg_e[i] - ppg_c[i]) * 2
    ans = ans ** 0.5
    return ans
    
# def dist2(ppg_e, ppg_c):
#     # array, 345 dim
#     ans = ppg_e - ppg_c
    


def hjk_main1():
    # in cn & in en
    en_l = []
    cn_l = []

    en_final_cn_idx = np.zeros((len(en_l))) # a[1000000]
    for i, e in enumerate(en_l):
        ans = 1e100
        ans_id = -1
        ans_id_etc = -1
        for j, c in enumerate(cn_l):
            if dist(e, c) < ans:
                ans = dist(e, c)
                ans_id = c
                ans_id_etc = c
        en_final_cn_idx[i] = ans_id
    np.save('en_final_cn_idx', en_final_cn_idx)


def cluster_hjk(a, K):
    # a = [a1, a2, ...], y = [label, label,...]
    class_as_index = k_means(a, K)
    return class_as_index

def hjk_main2():
    K = 20000
    en_l = []
    cn_l = []
    all_l = en_l + cn_l
    
    # 需要快速的聚类
    all_class = cluster_hjk(all_l, K)

    #... a[100], a[0].1, 2, 3,...  
    class_cn_ppgs = []
    for i in range(K):
        l = list()
        class_cn_ppgs[i].append(l)

    # int a[10][10];
    # a[0][0] = 888
    # a[0][1] = 999
    # a[0][2] = -1
    # a[1][0] = 777
    
    # 构造类的信息, 筛选出每个类里都有哪些中文的ppg; 并且平均每个类有100个中文ppg
    for i in range(len(cn_l)):
        idx = i + len(en_l)
        now_class = all_class[idx]
        class_cn_ppgs[now_class].append(i)

    # 开始寻找en对应的类内所有中文ppg离他最近的
    en_final_cn_idx = np.zeros((len(en_l))) # a[1000000]
    for i in range(len(en_l)):
        now_class = all_class[i]
        ans = 1e100
        ans_id = -1
        ans_id_etc = -1
        for j in class_cn_ppgs[now_class]:
            e = en_l[i]
            c = cn_l[j]
            if dist(e, c) < ans:
                ans = dist(e, c)
                ans_id = j
                ans_id_etc = c
        # 已经找到最接近的了，记下来
        en_final_cn_idx[i] = ans_id
    np.save('en_final_cn_idx', en_final_cn_idx)


if __name__ == '__main__':
    hjk_main1()
