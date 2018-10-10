import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

tf.set_random_seed(1)

# hyper param
BATCH_SIZE = 64
LR = 0.002
N_TEST_IMG = 5

# Mnist digits
mnist = input_data.read_data_sets("../MNIST_data", one_hot=False)
test_x = mnist.test.images[:200]
test_y = mnist.test.labels[:200]

# plot one example
plt.imshow(mnist.train.images[0].reshape((28,28)),cmap='gray')
plt.title("%i" % np.argmax(mnist.train.labels[0]))
plt.show()

tf_x  = tf.placeholder(tf.float32, [None,28*28])

# encoder
en0 = tf.layers.dense(tf_x, 128, tf.nn.relu)
en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
en2 = tf.layers.dense(en1, 12, tf.nn.tanh)
encoded = tf.layers.dense(en0, 3)
# encoded = tf.layers.dense(en2, 28*28, tf.nn.relu)

# decoder
de0 = tf.layers.dense(encoded, 12, tf.nn.tanh)
de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
de2 = tf.layers.dense(de1, 128, tf.nn.tanh)
decoded = tf.layers.dense(de1, 28*28, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=
                                    decoded)
train = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()

# 直接加载已经训练好的模型，下面注释掉
# sess.run(tf.global_variables_initializer())
# # initialize figure
# # f, a = plt.subplot(2, N_TEST_IMG, figsize = (5,2))
# f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
# plt.ion()
#
# # original data for viewing
# view_data = mnist.test.images[:N_TEST_IMG]
# for i in range(N_TEST_IMG):
#     a[0][i].imshow(np.reshape(view_data[i],(28, 28)), cmap='gray')
#     a[0][i].set_xticks(())
#     a[0][i].set_yticks(())
# for step in range(8000):
#     b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
#     _, encoded_, decoded_, loss_ = sess.run([train, encoded, decoded, loss],{
#         tf_x:b_x
#     })
#     # print(encoded_)
#     if step % 100 == 0:
#         print('train loss: %.4f' % loss_)
#         decoded_data = sess.run(decoded, {tf_x: view_data})
#         for i in range(N_TEST_IMG):
#             a[1][i].clear()
#             a[1][i].imshow(np.reshape(decoded_data[i],(28,28)),cmap='gray')
#             a[1][i].set_xticks(())
#             a[1][i].set_yticks(())
#         plt.draw()
#         plt.pause(0.001)
# plt.ioff()
#
# # visualize in 3D plot
# view_data = test_x[:200]
# encoded_data = sess.run(encoded,{tf_x:view_data})
#
# print(type(encoded_data))
# fig  = plt.figure(2)
# ax =Axes3D(fig)
# X, Y, Z = encoded_data[:, 0], encoded_data[:,1], encoded_data[:,2]
# for x,y,z,s in zip(X,Y,Z,test_y):
#     c = cm.rainbow(int(255*s/8))
#     ax.text(x,y,z,s,backgroundcolor = c)
# ax.set_xlim(X.min(),X.max())
# ax.set_ylim(Y.min(),Y.max())
# ax.set_zlim(Z.min(),Z.max())
# plt.show()

saver = tf.train.Saver()
# saver.save(sess, './autoencoder/autoEncoder_model')
saver.restore(sess, './autoencoder/autoEncoder_model')

item = np.array([ 2.2909867 , 2.3272301,  -2.4833374 ])
encoded_data = sess.run(encoded,{tf_x:mnist.test.images[:2]})
print(mnist.test.images[:2].shape)
print("this is encoded_data of %d" % mnist.test.labels[0])
plt.imshow(mnist.test.images[:2])
print(encoded_data)
decoded_data = sess.run(decoded,{encoded: item.reshape((1,3))})
# print(decoded_data)
plt.imshow(decoded_data.reshape((28,28)),cmap='gray')

plt.show()

    