import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
print(Weights.read_value())
print(loss)

init = tf.global_variables_initializer()

# sess 的使用
sess = tf.Session()
sess.run(init)
# print(sess.run(Weights))
# sess.run(init)

# for step in range(201):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(Weights), sess.run(biases))

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1,matrix2)
result = sess.run(product)
print(result)
sess.close()
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

# variable的使用，在tf中，定义了某字符串是变量，
#它才是变量，这一点有与Python是不同的。

state = tf.Variable(0, name='counter')
# 定义常量 one
one = tf.constant(1)

# 定义加法步骤
new_value = tf.add(state, one)

# 将State更新成 new_value
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(state)
        print(sess.run(state))

#Placeholder传入值,tf 如果想要从外部传入data，那就需要用到tf.placeholder()
# 然后传输数据 sess.run(***,feed_dict={input:""}

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.],input2:[2.]}))

# 定义层
def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    layer_name = 'layer%d'%n_layer
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name = "W")
            tf.summary.histogram(layer_name+'/weights', Weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name = "b")
            tf.summary.histogram(layer_name+"/biases",biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name+"/output",outputs)
    return outputs

import numpy as np
#导入数据
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 利用占位符定义我们所需恩要的神经网络的输入，tf.placeholder()就代表占位符，
# 这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以是1

xs = tf.placeholder(tf.float32, [None,1],name = "x_in")
ys = tf.placeholder(tf.float32, [None,1],name = "y_in")

with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32,[None,1])
    ys = tf.placeholder(tf.float32,[None,1])
#搭建网络
#隐藏层
l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
#输出层
prediction = add_layer(l1,10,1,n_layer=2,activation_function=None)
# 计算预测值prediction和真实值之间的误差
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
    tf.summary.scalar("loss",loss)
#训练
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#使用变量时，进行初始化，这是必不可少的
init = tf.global_variables_initializer()
#定义 Session，并用Session来执行init初始化步骤，注意在tensorflow中，只有
#session.run()才会执行我们定义的运算。
sess = tf.Session()
sess.run(init)
merged = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("logs/",sess.graph)
#结果可视化
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
# # plt.ion()
# plt.show()

#下面让机器学习开始训练
# 每间隔50次训练刷新一次图形，用红心，宽度为5的线来显示我们的预测数据和输入之间的关系
# 并且暂停0.1秒
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        re = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(re,i)
        print(sess.run(loss,feed_dict={xs: x_data,ys: y_data}))
        # try:
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)
        # plt.show()
plt.show()

# xs = tf.placeholder(tf.float32,[None,1])
# ys = tf.placeholder(tf.float32,[None,1])

# xs = tf.placeholder(tf.float32,[None,1],name = "x_in")
# ys = tf.placeholder(tf.float32,[None,1],name = "y_in")

# with tf.name_scope("inputs"):
#     xs = tf.placeholder(tf.float32,[None,1])
#     ys = tf.placeholder(tf.float32,[None,1])

#编辑layer，