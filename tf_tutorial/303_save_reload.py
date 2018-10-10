import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# fake data
x = np.linspace(-1,1,100)[:,np.newaxis]
x = np.linspace(-1,1,100).reshape([100,1])
# numpy.newaxis 作用，添加一列，场景，比如从矩阵中取出了一列，想还原成矩阵，可以使用np.newaxis
print(x)
noise = np.random.normal(0, 0.1, size = x.shape)

y = np.power(x,2) + noise

def save():
    print("this is save")
    # build neural network
    
    tf_x = tf.placeholder(tf.float32, x.shape)
    tf_y = tf.placeholder(tf.float32, y.shape)
    
    l = tf.layers.dense(tf_x, 10, tf.nn.relu)
    o = tf.layers.dense(l,1)
    
    loss = tf.losses.mean_squared_error(tf_y,o)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # 定义一个 saver
    saver = tf.train.Saver()
    for step in range(100):
        sess.run(train_step, {tf_x:x, tf_y:y})
    
    saver.save(sess, './params', write_meta_graph=False)
    
    # plotting
    pred, l = sess.run([o,loss],{tf_x:x, tf_y:y})
    
    plt.figure(1,figsize=(10,5))
    plt.subplot(121)
    plt.scatter(x,y)
    
    plt.plot(x, pred, 'r-', lw=5)
    
    plt.text(-1,1.2, 'Save Loss=%.4f' %l, fontdict={'size': 15, 'color':'red'})
    
    
def reload():
    print("this is reload")
    
    # build entire net again and restore
    tf_x = tf.placeholder(tf.float32, x.shape)
    tf_y = tf.placeholder(tf.float32, y.shape)
    
    l_ = tf.layers.dense(tf_x, 10, tf.nn.relu)
    o_ = tf.layers.dense(l_, 1)
    loss_ = tf.losses.mean_squared_error(tf_y,o_)
    
    sess = tf.Session()
    # don't need to initialize variables, just restoring trained variables
    
    saver = tf.train.Saver()
    saver.restore(sess, './params')
    #plotting
    pred,l = sess.run([o_,loss_],{tf_x:x, tf_y:y})
    plt.subplot(122)
    
    plt.scatter(x,y)
    
    plt.plot(x, pred, 'r-', lw=5)
    plt.text(-1,1.2, 'Reload Loss=%.4f' % l,fontdict={'size':15,
                                                      'color':'red'})
    plt.show()
    
save()
# destroy previous net
tf.reset_default_graph()

reload()