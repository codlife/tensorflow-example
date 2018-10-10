# coding: utf8
# import tensorflow as tf
# import numpy as np
#
#
# cell = tf.nn.rnn_cell.BasicRNNCell(num_units = 128)
# print(cell.state_size)
#
# inputs = tf.placeholder(np.float32, shape=(32,100))
# h0 = cell.zero_state(32, np.float32)
# output, h1 = cell.call(inputs, h0)
# print(h1.shape)


import tensorflow as tf
import numpy as np
#
#  lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=128)
# inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
# h0 = lstm_cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
# output, h1 = lstm_cell.call(inputs, h0)
#
# print(h1.h)  # shape=(32, 128)
# print(h1.c)  # shape=(32, 128)

# embedding = tf.Variable('embedd',[2,4])
embedding = tf.get_variable('embedding',[2,4])
sess = tf.Session()
print(sess.run(tf.global_variables_initializer()))
print(sess.run(embedding))