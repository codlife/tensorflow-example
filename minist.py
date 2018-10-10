from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# our application logic will be added here
def cnn_model_fn(features, labels, mode):
    """MOdel fun for CNN"""
    #Input layer
    with tf.name_scope("input") as scope:
        input_layer = tf.reshape(features["x"],[-1, 28, 28, 1])

    #Convontional Layer #1
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5,5],
        padding = "same",
        activation = tf.nn.relu)

    #Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs= conv1, pool_size =2, strides=2)

    # Con layer2 and pooling layer 2
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = 5,
        padding = "same",
        activation = tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides = 2)

    # Dense layer
    pool2_flat = tf.reshape(pool2,[-1, 7*7*64])
    dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs = dense, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # logits layer
    logits = tf.layers.dense(inputs = dropout, units = 10)

    predictions = {
        # Generate predictions (for predict and eval mode)
        "classes": tf.argmax(input = logits, axis = 1),
        # Add 'softmax_tensor' to the graph. It is used for predict and by the 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    # calculate loss(for both train and eval modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

    # Configure the Training Op(for Train mode)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    # Add evaluation metrics(for eval mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels = labels, predictions = predictions['classes'])
    }
    # tf.summary.scalar('loss', loss)

    return tf.estimator.EstimatorSpec(mode = mode, loss =loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    #load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images # return np.array
    train_labels = np.asarray(mnist.train.labels, dtype = np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels,dtype=np.int32)
    print(train_data.shape)

    # create the estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn= cnn_model_fn, model_dir = "..mnist_convnet_model"
    )
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 50)

    # train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_data},
        y = train_labels,
        batch_size = 1,
        num_epochs = None,
        shuffle = True
    )
    mnist_classifier.train(input_fn = train_input_fn,
                           steps = 2000,
                           hooks = [logging_hook])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":eval_data},
        y = eval_labels,
        num_epochs = 1,
        shuffle = False
    )
    eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)
    print(eval_results)



if __name__ == "__main__":
    tf.app.run()