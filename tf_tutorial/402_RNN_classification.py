import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper parameters
BATCH_SIZE = 64
# rnn time step  image height
TIME_STEP = 28
# rnn input size image width
INPUT_SIZE = 28
LR = 0.01

# data
mnist = input_data.read_data_sets('../MNIST_data', one_hot = True)

test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot one example
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
plt.imshow(mnist.train.images[0].reshape((28,28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()

# tensorflow placeholders
# shape  batch 784
tf_x = tf.placeholder(tf.float32, [None,TIME_STEP * INPUT_SIZE])
# ()
image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])
tf_y = tf.placeholder(tf.int32, [None, 10])

# RNN

rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units = 64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,
    image,
    initial_state=None,
    dtype = tf.float32,
    time_major = False
)
output = tf.layers.dense(outputs[:, -1, :], 10)
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits = output)

train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(
    labels = tf.argmax(tf_y, axis = 1),
    predictions = tf.argmax(output, axis = 1),
)[1]
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
for step in range(1200):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ =  sess.run([train_op, loss],{tf_x:b_x, tf_y:b_y})
    if step % 50 == 0: # testing
        accuracy_ = sess.run(accuracy, {tf_x:b_x, tf_y:b_y})
        print('train Loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

#print 10 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:10]})
print(test_output)
pred_y = np.argmax(test_output,1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')

a = np.array([[1,2],
     [3,4]])
print(a[:,])
