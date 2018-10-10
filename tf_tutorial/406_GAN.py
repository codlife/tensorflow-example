import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hpyer Para
BATCH_SIZE = 64
# learning rate for generator
LR_G = 0.0001
# learning rate for discriminator
LR_D = 0.0001
# think of this as number of ideas for generating an art work
N_IDEAS = 5
# it colud be total point G can draw in the canvas
ART_COMPONENTS = 15
print(np.array([np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE)]).shape)
# PAINT_POINTS = np.vstack([np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE)])
PAINT_POINTS = np.array([np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE)])

print(PAINT_POINTS.shape)
# show out beautiful painting range
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0],2) + 1, c = '#74BCFF',
         lw = 3, label = 'upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0],2) + 0, c = '#FF9359',
         lw = 3, label = 'lower bound')
plt.legend(loc = 'upper right')
plt.show()


# painting from the famous artist(real target)
def artist_works():
    a = np.random.uniform(1, 2, size = BATCH_SIZE)[:, np.newaxis]
    print(a)
    painting = a * np.power(PAINT_POINTS, 2) + (a - 1)
    return painting

with tf.variable_scope('Generator'):
    # random ideas (could from normal distribution)
    G_in = tf.placeholder(tf.float32, [None, N_IDEAS])
    G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)
    G_out= tf.layers.dense(G_l1, ART_COMPONENTS)
    
with tf.variable_scope('Discriminator'):
    real_art = tf.placeholder(tf.float32, [None,ART_COMPONENTS],
                              name = 'real_in')
    D_l0 = tf.layers.dense(real_art, 128, tf.nn.relu, name = 'l')
    prob_artist0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name = 'out')
    # reuse layers for generator
    D_l1 = tf.layers.dense(G_out, 128, tf.nn.relu, name = 'l', reuse = True)
    prob_artist1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name = 'out', reuse = True)

# D_loss = - tf.reduce_mean(tf.log(prob_artist0) + tf.log(1 - prob_artist1))
D_loss = - tf.reduce_mean(tf.log(prob_artist0) + tf.log(1 - prob_artist1) )

G_loss = tf.reduce_mean(tf.log(1- prob_artist1))

train_D = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope = 'Discriminator')
)
train_G = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss, var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope = 'Generator')
)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
plt.ion()

for step in range(5000):
    # real painting from artist
    artist_painting = artist_works()
    G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)
    G_paintings, pa0, D1 = sess.run([
        G_out, prob_artist0, D_loss, train_D, train_G],
        {G_in:G_ideas, real_art: artist_painting}
    )[:3]
    if step % 50 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % pa0.mean(), fontdict={'size': 15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D1, fontdict={'size': 15})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=12)
        plt.draw()
        plt.pause(0.1)

plt.ioff()
plt.show()