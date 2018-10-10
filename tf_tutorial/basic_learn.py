import tensorflow as tf
import pandas as pd

data = pd.read_csv('./123.csv')
# data['label']
# y = data['label']
# x = data[1:]
# print(data)
# print(x)
# print(y)

# for index in range(len(data)):
#     item = data.iloc[[index]]
#     print(item)
#     a
#     print("*"*10)

print(data.values)
print(type(data.values))

# label = data.values[0,axis]
# print(label)
import numpy as np
y = []
x = []
for item in data.values:
    y.append(item[0])
    print(item[0])
    x.append(item[1:].reshape((1,1,2,2)))

print(x[0].shape)
print(y)
import numpy
A = numpy.array([[[1.0],[2.0],[3.0]],[[1.0],[2.0],[3.0]]])
print(A.shape)
with tf.Session() as sess:
    print(sess.run(tf.nn.softmax(A, dim = -1)))