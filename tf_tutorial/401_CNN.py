import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001

mnist = input_data.read_data_sets('../MNIST_data', one_hot = True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot one example
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
plt.imshow(mnist.train.images[0].reshape((28,28)),cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
print(mnist.train.labels[0])
plt.show()

tf_x = tf.placeholder(tf.float32,[None, 28*28]) / 255
image = tf.reshape(tf_x, [-1, 28, 28,1])
tf_y = tf.placeholder(tf.int32,[None,10])

# CNN
# shape(28,28,1)
conv1 = tf.layers.conv2d(
    inputs = image,
    filters = 16,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    activation = tf.nn.relu
)

pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size = 2,
    strides = 2
)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1,'same', activation = tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
flat = tf.reshape(pool2, [-1, 7*7*32])
output = tf.layers.dense(flat,10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits = output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)


accuracy = tf.metrics.accuracy(labels = tf.argmax(tf_y, axis = 1),
                               predictions=tf.argmax(output,axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# following functions is for visualization
sess.run(init_op)
plt.ion()

# following func is for vis
from matplotlib import cm
try:
    from sklearn.manifold import TSNE
    HAS_SK = True
except:
    HAS_SK = False
    print("Please install sklearn for visualization\n")

def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X,Y = lowDWeights[:, 0], lowDWeights[:,1]
    for x,y,s in zip(X,Y,labels):
        c = cm.rainbow(int(255 *s /9))
        plt.text(x,y,s,backgroundcolor = c, fontsize = 9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(),Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.1)
    
    
for step in range(600):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op,loss], {tf_x: b_x, tf_y:b_y})
    
    if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat],{
            tf_x:test_x, tf_y:test_y
        })
        print(accuracy_)
        print('Step:', step,'| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
        print(flat_representation.shape)
        if HAS_SK:
            tsne = TSNE(perplexity=30,
                        n_components=2,
                        init='pca',
                        n_iter = 5000)
            plot_only = 500
            low_dim_embs = tsne.fit_transform(
                flat_representation[:plot_only,:]
            )
            labels = np.argmax(test_y, axis = 1)[:plot_only]
            plot_with_labels(low_dim_embs,labels)
        
plt.ioff()
plt.show()

# print 10 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output,1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10],1),'real number')

