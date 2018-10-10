import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

n_data = np.ones((100,2))
# random.normal 第一个参数均值，第二个参数方差
x0 = np.random.normal(2 * n_data, 1)
y0 = np.zeros(100)
x1 = np.random.normal(-2 * n_data,1)
y1 = np.ones(100)

x = np.vstack((x0,x1))
# vstack 按row拼接在一起 row wise
# hstack 按列拼接在一起 col wise
# print(x[0])
# print(x0[0])
# print(x1[0])
# print(x.shape)
# print(x0.shape)
# print(x1.shape)
y = np.hstack((y0,y1))
# print(y)
# print(y0.shape)

# plot data
print(x)
# c  @color 如果是是数组，后面必须有cmap映射
# plt.scatter(x[:,0], x[:,1], c = y, s= 100, lw=0, cmap='RdYlGn')
# plt.show()
one_hot_y=tf.one_hot(y,depth=2)
tf_x = tf.placeholder(tf.float32,x.shape)
tf_y = tf.placeholder(tf.int32, one_hot_y.shape)
tf_y1 = tf.placeholder(tf.int32, y.shape)

# neural network layers
l1 = tf.layers.dense(tf_x, 100, tf.nn.relu)
output = tf.layers.dense(l1, 2)

print(y)


# 注意softmax_cross_entropy 和 sparse_softmax_cross_entropy的区别
# sparse_softmax_cross_entropy 不需要自己进行one_hot encoding
# softmax_cross_entropy 需要进行one_hot encoding
# softmax label 必须介于 [0,num_classes)之间
loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y1,logits = output)
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits = output)
accuracy = tf.metrics.accuracy(labels = tf.argmax(tf_y,1),
                               predictions = tf.argmax(output,axis = 1),)[1]
optimizer = tf.train.GradientDescentOptimizer(0.05)
train_op = optimizer.minimize(loss)
sess = tf.Session()
# accuracy 函数会创建两个局部变量，需要调用tf.local_variables_initializer()
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init_op)

plt.ion()
value = sess.run(one_hot_y)
for step in range(100):
    # train and net output
    _, acc, pred = sess.run([train_op, accuracy, output],{tf_x:x, tf_y:value})
    if step % 10 == 0:
        # plot and show learning process
        plt.cla()
        # print(type(pred))
        # print(pred[101])
        # print(pred[101].argmax())
        plt.scatter(x[:,0],x[:,1],c = pred.argmin(1), s = 100,
                    lw=0, cmap='RdYlGn')
        plt.text(1.5,-4, 'Accuracy=%.2f' % acc, fontdict={'size':20,
                                                          'color':'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()

a = np.array([[1,3],[2,4],[7,6]])
print(a.argmax(1))