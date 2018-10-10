import tensorflow as tf

# tensorflow的核心是：构建计算图，运行计算图

node1 = tf.constant(3.0,dtype = tf.float32)
node2 = tf.constant(4.0)# also tf.float32 implicitly
print(node1,node2)

#要想打印最终结果，必须使用session，session封装了tensorflow运行时的状态可控制

sess = tf.Session()
print(sess.run([node1,node2]))

# 组合tensor节点操作，构造更加复杂的计算
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):",sess.run(node3))


#6tensorflow 提供了一个统一可视化展示工具的称为tensorBoard

#一个计算图可以参数化的接收外部的输入，作为一个placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b # + provides a shortcut for tf.add(a,b)
print(adder_node)
#通过feed_dict 参数传递到具体的值到run方法，来进行多个输入
print(sess.run(adder_node,{a:4,b:5}))
print(sess.run(adder_node,{a:[1,2],b:[2,4]}))


add_and_triple = adder_node * 3
print(sess.run(add_and_triple,{a:3,b:4.5}))

#9在机器学习中，我们通常想让一个模型接收任意多个输入，比如大于1个，好让这个模型可以被
#训练，在不改变输入的情况下，我们需要改变这个计算图去获得一个新的输入，
#比那辆允许我们增加可计算的参数到这个计算图中，他们被构造成有一个类型和初始值
# 这也成为，符号化编程

W = tf.Variable([.3],dtype = tf.float32)
b = tf.Variable([-.3],dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
# 10 ,当你调用tf.constant时常量被初始化，它们的值是不可以改变的，而放你调用
#tf.Variable时，没有被初始化，在，tensorflow程序中想要初始化这些变量，必须
#调用global_variables_initializer(),否则只有调用run的时候才会初始化

init = tf.global_variables_initializer()
sess.run(init)
#11
print(sess.run(linear_model,{x:[1,2,3,4]}))

#定义一个损失函数

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

# 使用tf.assign 对 tf.Variable 进行修改
fixw = tf.assign(W,[-1])
fixb = tf.assign(b,[1])
sess.run([fixw,fixb])
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

# 14 tf.train
# tf.train 优化损失函数
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults
for i in range(1000):
    sess.run(train, {x:[1,2,3,4],y:[0,-1,-2,-3]})

print(sess.run([W,b]))
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

# tf.estimator 是一个更加高级别的tensorflow库，简化了机械式的机器学习，
#包含以下几个方面
# running training loops
# running evaluation loops
# managing data sets

# tf.setimator 定义了很多相同的模型

# 
