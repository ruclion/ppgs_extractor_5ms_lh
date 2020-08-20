

#PPG-extractor：

/* 本代码以mfcc作为输入，以ppgs作为输出。





一.目录文件及作用

/

	1.audio.py : 

		其作用是将.wav声纹文件转化为mfcc。

		使用其 wav2mfcc_v2函数


		hparams = {
		    'sample_rate': 16000,#一秒16000个采样点
		    'preemphasis': 0.97,
		    'n_fft': 512,
		    'hop_length': 80,#80个采样点为帧移动步长 5ms
		    'win_length': 400,#400个采样点为帧宽度，25ms
		    'num_mels': 80,
		    'n_mfcc': 13,
		    'window': 'hann',
		    'fmin': 30.,
		    'fmax': 7600.,
		    'ref_db': 20,  #
		    'min_db': -80.0,  # restrict the dynamic range of log power
		    'iterations': 100,  # griffin_lim #iterations
		    'silence_db': -28.0,
		    'center': True,#是否将MFCC作为当前帧中间向量的结果。（数个向量作为一帧生成一个mfcc)
		}
		是audio.py生成mfcc的分帧、帧移的参数设定。


	2.models.py : 

		定义了以mfcc作为输入，ppgs作为输出的一个CNN+BLSTM网络。

		使用了CNNBLSTMClassifier()这个类

	3.timit_dataset.py

		定义了数据的迭代器。在train/test中得以应用

		需要指定：
			TRAIN_FILE:一个txt文本，内容是训练数据文件名。如(103-1240-0000),每一行一个文件名。
			TEST_FILE:一个txt文本，内容同上

			MFCC_DIR:存储生成的mfcc数据的目录，目录结构如下：
					MFCC_DIR/
								103/
										1240/
												103-1240-000x.npy
								说话人	语句		语句分段

			PPG_DIR:存储PPGs数据的目录，该数据由Kaldi生成，与我们生成的mfcc数据应一一对应。（帧的划分，位移参数相同）

					目录组织方式同上。

			MFCC_DIM = 39 ，PPG_DIM = 345 指定每帧生成的MFCC/ppg的维度。

	4.train.py

		训练网络的代码，指定了网络计算图和tensorboard的监视功能。

/utils/


	1. make_mfcc:

		调用audio.py中的wav2mfcc_v2函数，使用参数与其中相同。

		将每个语句分段按帧滑动分割，对每一帧加窗后生成一个MFCC。这和Kaldi按帧生成的PPGs应能一一对应。

		（实践中，发现我们的MFCC会比Kaldi生成的ppgs 帧少1，但是应问题不大）

		我们生成的mfcc，之所以会比ground-truth ppgs少一帧，是因为padding选择不同。但是基本是对的上的

	2. generate_data_list.py

		使用Kaldi的结果，即ali.x.txt，每一行是 语句分段.wav Kaldi生成的PPGs

		这个只是为了提取出在timit_dataset.py中所需的TRAIN_FILE和TEST_FILE.也就是只提取每一行的第一个关键字

	3. collect_ppgs.py

		基于ali.x.txt，提取每一行的PPgs字段，存储在npy文件中。

		并且文件的组织方式与MFCC的组织方式一样。都遵循最开始该数据集的组织方式

	4. test_data_length.py

		检测我们对每个语句分段 生成MFCC的数目是否和PPGs的数目相同，如果相同，则说明我们分帧对应上了

		写这个脚本的原因是，7067-76048-0021我们生成的和Kaldi生成的不一样长，分帧没对应上。

		我把0021删去了

	5. test_length

		是检测我们生成的mfcc 至少在103-1240中是否与ali.x.txt里的ppgs是一样多的

	6. mfcc_zhao/ PPGs/ 是对应的mfcc存储路径和ppgs存储路径


/experiment/
	
	存储实验日志


/model/

	存储最好的一次模型，第62000steps

/my_test/

	存储103-1240源数据，ali.1.txt 和没有对齐的数据。







训练策略：

         基于 /tmp/luhui_share/alignments 对齐数据
             /home/data/LibriSpeech/train-clean-100 音频数据
         1.利用Kaldi，生成PPGs-ground-truth。保存在上面第一个路径中，ali.x.txt

         2.利用make_mfcc，为源数据生成mfcc路径

         3.利用collect_ppgs.py ,为ppgs单独生成类似mfcc的路径

         4.利用test_data_length.py 检查是不是每个语句的mfcc数量都和ppgs数量相同

         5.测试timit_dataset.py 能不能获取每个语句片段的 mfcc ppgs mfcc数目 三元组

         6.运行train.py












