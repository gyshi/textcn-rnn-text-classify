# Text Classification with CNN and RNN

使用卷积神经网络以及循环神经网络进行中文文本分类

参考：https://github.com/gaussic/text-classification-cnn-rnn

CNN做句子分类的论文可以参看: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

还可以去读dennybritz大牛的博客：[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

以及字符级CNN的论文：[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)

## 环境

- Python 3.6
- TensorFlow 1.3以上
- numpy
- scikit-learn
- nltk
- jieba

## 数据集

使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议。

本次训练使用了其中的10个分类，每个分类6500条数据。

类别如下：

```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```

数据集划分如下：

- 训练集: 5000*10
- 验证集: 500*10
- 测试集: 1000*10

- cnews.train.txt: 训练集(50000条)
- cnews.val.txt: 验证集(5000条)
- cnews.test.txt: 测试集(10000条)

## 预处理

`data/loader.py`为数据的预处理文件。

- `read_file()`: 读取文件数据;
- `build_vocab()`: 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
- `read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `to_words()`: 将一条由id表示的数据重新转换为文字;
- `process_file()`: 将数据集从文字转换为固定长度的id序列表示;
- `batch_iter()`: 为神经网络的训练准备经过shuffle的批次的数据。

- `*_model` : 为模型文件
- `*_trian` : 为训练数据
- `*_text` : 为测试数据
- `*_predict` : 为预测数据

```
word2vec 微调（no-static）
self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size,   
         self.config.embedding_size], 
         initializer=tf.constant_initializer(self.config.pre_trianing),trainable=True)


self.embedding_inputs= tf.nn.embedding_lookup(self.embedding, self.input_x)


word2vec 微调（static）
self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size,   
         self.config.embedding_size], 
         initializer=tf.constant_initializer(self.config.pre_trianing),trainable=False)


self.embedding_inputs= tf.nn.embedding_lookup(self.embedding, self.input_x)

随机初始化
self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size,   
         self.config.embedding_size], 
         initializer=tf.truncated_normal_initializer(0.0,stddev=0.2))

     

self.embedding_inputs= tf.nn.embedding_lookup(self.embedding, self.input_x)




```

```

                            train_acc               val_acc                  test_acc
cnn： 
      word2vec(static):       96.074%               96.660%                  96.850%
      word2vec(NO-static)：   98.438%               97.620%                  97.280%
      random：                98.438%               97.220%                  97.560%

RNN(lstm):
      word2vec(static):       93.750%               92.320%                  93.380%
      word2vec(NO-static)：   98.438%               97.680%                  97.400%
      random：                20%                     ---                      ---


RNN(gru):
      word2vec(static):       98.438%               97.580%                  97.380%
      word2vec(NO-static)：   98.438%               97.560%                  97.790%
      random：                1.0000%               95.550%                  95.589%









