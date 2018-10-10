from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#搭建网络，None 表示输入可以是任意个样本
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])


def add_layer(inputs,input_size,output_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([input_size,output_size]))
    biases = tf.Variable(tf.zeros([1,output_size])+0.1)
    wx_plus_b = tf.matmul(inputs,Weights) + biases
    
    if activation_function is None:
        return wx_plus_b
    else:
        return activation_function(wx_plus_b)
    
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),1))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prection = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prection,tf.float32))
    result = sess.run(accuracy, feed_dict = {xs: v_xs, ys: v_ys})
    return result
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
# print(mnist.test.images.shape)
# print(mnist.test.labels.shape)
# x = tf.Variable(tf.constant(0.1,shape =[10]))
# sess.run(tf.global_variables_initializer())
# print(sess.run(x))
# print(sess.run(tf.argmax([1,2,3],1)))