import tensorflow as tf
import numpy as np
# save to file

# restore 提取数据的时候需要定义相同的 dtype 和 shape
W = tf.Variable([[1,2,3],[3,4,5]],dtype = tf.float32, name = "weights")
b = tf.Variable([1,2,3],dtype = tf.float32, name = "biases")

init = tf.global_variables_initializer()

#保存时，首先建一个tf.train.Saver() 用来保存，提取变量，再创建一个名为my_net的文件夹
#用这个saver来保存变量到这个目录“my_net/save_net.ckpt”

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess,"my_net/save_net.ckpt")
    print("Save to path:", save_path)
    
# 提取数据
# 建立 w,b的容器
W = tf.Variable(np.arange(6).reshape(2,3),dtype = tf.float32, name = "weights")
b = tf.Variable(np.arange(3).reshape(1,3),dtype = tf.float32, name = "biases")

# 这里不需要初始化步骤 init= tf.initialize_all_variables()

# saver = tf.train.Saver()
with tf.Session() as sess:
    #提取变量
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights", sess.run(W))
    print("biases", sess.run(b))
