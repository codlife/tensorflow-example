import tensorflow as tf

var = tf.Variable(0)
add_operation = tf.add(var,1)
update_operation = tf.assign(var, add_operation)

a = tf.placeholder(shape=[1,2],dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        # sess.run(update_operation)
        # print(sess.run(var))
        print(sess.run(add_operation))
        print((sess.run(var)))
        print(add_operation)
        print(var)
        print(update_operation)
        print(a)