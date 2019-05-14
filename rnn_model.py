#encoding:utf-8
import  tensorflow as tf

class TextConfig():

    embedding_size=100     #dimension of word embedding
    vocab_size=8000        #number of vocabulary
    pre_trianing = None   #use vector_char trained by word2vec

    num_layer = 2
    seq_length=600         #max length of sentence
    num_classes=10         #number of labels
    rnn = 'lstm'
    num_filters=128        #number of convolution kernel



    keep_prob=0.8          #droppout
    lr= 1e-3               #learning rate
    lr_decay= 0.9          #learning rate decay
    clip= 6.0              #gradient clipping threshold
    l2_reg_lambda=0.01     #l2 regularization lambda

    num_epochs=10          #epochs
    batch_size=64         #batch_size
    print_per_batch =100   #print result

    train_filename='./data/cnews.train.txt'  #train data
    test_filename='./data/cnews.test.txt'    #test data
    val_filename='./data/cnews.val.txt'      #validation data
    vocab_filename='./data/vocab.txt'        #vocabulary
    vector_word_filename='./data/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='./data/vector_word.npz'   # save vector_word to numpy file

class TextRNN(object):


    def __init__(self,config):

        self.config = config

        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2_loss = tf.constant(0.0)

        self.rnn()

    def rnn(self):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.num_filters, state_is_tuple=True)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.num_filters)

        def dropout():
            if self.config.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()

            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = self.keep_prob)

        with tf.device('/cpu:0'):

            # self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
            #                                  initializer=tf.constant_initializer(self.config.pre_trianing),trainable=True)

            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.truncated_normal_initializer(0.0,stddev=0.2))

            # self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
            #                                  initializer=tf.zeros_initializer())

            self.embedding_inputs= tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope('rnn'):
            cellls = [dropout() for _ in range(self.config.num_layer)]

            rnn_cell = tf.contrib.rnn.MultiRNNCell(cellls, state_is_tuple=True)

            _output, _ = tf.nn.dynamic_rnn(cell = rnn_cell, inputs = self.embedding_inputs, dtype= tf.float32)
            self.last = _output[:, -1, :]




        with tf.name_scope('output'):
            fc = tf.layers.dense(self.last, self.config.num_filters, name='fc1')
            fc = tf.contrib.layers.dropout(fc,self.keep_prob)
            fc = tf.nn.relu(fc, name='relu')
            self.logits = tf.layers.dense(fc, self.config.num_classes, name = 'fc2')
            self.y_pred_cls = tf.arg_max(tf.nn.softmax(self.logits), 1)
            self.prob = tf.nn.softmax(self.logits)



        with tf.name_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)


        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

























