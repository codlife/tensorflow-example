import tensorflow as tf
import numpy as np

#load data
npx = np.random.uniform(-1,1, (1000,1))
npy = np.power(npx,2 ) + np.random.normal(0,0.1, size=npx.shape)

npx_train, npx_test = np.split(npx, [800])
npy_train, npy_test = np.split(npy, [800])

# use placeholder, later you may diff data, pass the diff data into placeholder

tfx = tf.placeholder(npx_train.dtype, npx_train.shape)
tfy = tf.placeholder(npy_train.dtype, npy_train.shape)

# create dataloader

dataset = tf.data.Dataset.from_tensor_slices((tfx, tfy))
# choose data randomly from this buffer
dataset = dataset.shuffle(buffer_size = 1000)
# batch size you will use
dataset = dataset.batch(32)
#repeat for 3 epochs
dataset = dataset.repeat(3)

iterator = dataset.make_initializable_iterator()

# your network
bx,by = iterator.get_next()
l1 = tf.layers.dense(bx, 10, tf.nn.relu)
out = tf.layers.dense(l1, npy.shape[1])

loss = tf.losses.mean_squared_error(by, out)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()


# need to initialize the iterator in this case
sess.run([iterator.initializer, tf.global_variables_initializer()],feed_dict=
         {tfx: npx_train, tfy:npy_train})
for step in range(201):
    try:
        _, train1 = sess.run([train, loss])
        if step % 10 == 0:
            test1 = sess.run(loss, {bx:npx_test,by:npy_test})
            print('step: %i/200' % step, '|train loss:', train1,'|test loss:', test1)
    except tf.errors.OutOfRangeError:
        print('Finish the last epoch.')
        break
        