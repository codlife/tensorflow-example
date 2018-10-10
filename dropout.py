import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

xs = tf.placeholder(tf.float32,[None,64])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)


def add_layer(inputs,input_size,output_size,activation_function=None):
    Weights = tf.Variable(tf.random_uniform([input_size,output_size]))
    biased = tf.Variable(tf.zeros([1,output_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs,Weights) + biased
    wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob)
    if activation_function is None:
        return wx_plus_b
    else:
        return activation_function(wx_plus_b)
#准备数据
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=.3)

l1 = add_layer(xs, 64, 50, activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, activation_function=tf.nn.softmax)
with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),1))
    tf.summary.scalar("loss",cross_entropy)
    
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#训练
merged = tf.summary.merge_all()

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={l1: v_xs})
    correct_prection = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prection,tf.float32))
    result = sess.run(accuracy, feed_dict = {xs: v_xs, ys: v_ys })
    return result
prob = 0.4
writer = tf.summary.FileWriter("logs",sess.graph)
for i in range(500):
    sess.run(train_step, feed_dict = {xs: x_train, ys:y_train,keep_prob:prob })
    if i % 50 == 0:
        # print(type(x_train))
        # print(sess.run(compute_accuracy(x_test,y_test)))
        loss = sess.run(merged, feed_dict = {xs: x_train, ys:y_train,keep_prob:prob})
        writer.add_summary(loss,i)
        print("first", sess.run(cross_entropy, feed_dict = {xs: x_test, ys:y_test,keep_prob:1}))
        print("second",sess.run(cross_entropy, feed_dict={xs: x_test, ys: y_test, keep_prob: 0.1}))

        