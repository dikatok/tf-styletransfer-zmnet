from io import BytesIO

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io


def create_loss_model_fn(weights_path, data_format="channels_last"):
    """Higher order function to create vgg16 net

    :param weights_path: Path to vgg numpy weights
    :param data_format: Either 'channels_last' or 'channels_first'
    :return: vgg16 function
    """

    weights = np.load(BytesIO(file_io.read_file_to_string(weights_path, binary_mode=True)))
    nn_data_format = "NHWC" if data_format == "channels_last" else "NCHW"

    def vgg16(inputs):
        net = {}

        with tf.variable_scope("vgg16", reuse=tf.AUTO_REUSE):
            mean_shape = [1, 1, 1, 3] if data_format == "channels_last" else [1, 3, 1, 1]
            mean = tf.constant(
                [123.68, 116.779, 103.939],
                dtype=tf.float32,
                shape=mean_shape,
                name='imagenet_mean')
            inputs = inputs - mean

            with tf.name_scope('conv1_1') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv1_1_W"]),
                    trainable=False,
                    name='conv1_1_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv1_1_b"]),
                    trainable=False,
                    name='conv1_1_b')
                conv1_1 = tf.nn.conv2d(
                    inputs,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv1_1 = tf.nn.bias_add(conv1_1, biases, data_format=nn_data_format)
                net["conv1_1"] = conv1_1 = tf.nn.relu(conv1_1, name=scope)

            with tf.name_scope('conv1_2') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv1_2_W"]),
                    trainable=False,
                    name='conv1_2_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv1_2_b"]),
                    trainable=False,
                    name='conv1_2_b')
                conv1_2 = tf.nn.conv2d(
                    conv1_1,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv1_2 = tf.nn.bias_add(conv1_2, biases, data_format=nn_data_format)
                net["conv1_2"] = conv1_2 = tf.nn.relu(conv1_2, name=scope)

            pool1 = tf.nn.avg_pool(
                conv1_2,
                ksize=[1, 2, 2, 1] if nn_data_format == "NHWC" else [1, 1, 2, 2],
                strides=[1, 2, 2, 1] if nn_data_format == "NHWC" else [1, 1, 2, 2],
                padding='SAME',
                data_format=nn_data_format,
                name='pool1')

            with tf.name_scope('conv2_1') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv2_1_W"]),
                    trainable=False,
                    name='conv2_1_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv2_1_b"]),
                    trainable=False,
                    name='conv2_1_b')
                conv2_1 = tf.nn.conv2d(
                    pool1,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv2_1 = tf.nn.bias_add(conv2_1, biases, data_format=nn_data_format)
                net["conv2_1"] = conv2_1 = tf.nn.relu(conv2_1, name=scope)

            with tf.name_scope('conv2_2') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv2_2_W"]),
                    trainable=False,
                    name='conv2_2_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv2_2_b"]),
                    trainable=False,
                    name='conv2_2_b')
                conv2_2 = tf.nn.conv2d(
                    conv2_1,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv2_2 = tf.nn.bias_add(conv2_2, biases, data_format=nn_data_format)
                net["conv2_2"] = conv2_2 = tf.nn.relu(conv2_2, name=scope)

            pool2 = tf.nn.avg_pool(
                conv2_2,
                ksize=[1, 2, 2, 1] if nn_data_format == "NHWC" else [1, 1, 2, 2],
                strides=[1, 2, 2, 1] if nn_data_format == "NHWC" else [1, 1, 2, 2],
                padding='SAME',
                data_format=nn_data_format,
                name='pool2')

            with tf.name_scope('conv3_1') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv3_1_W"]),
                    trainable=False,
                    name='conv3_1_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv3_1_b"]),
                    trainable=False,
                    name='conv3_1_b')
                conv3_1 = tf.nn.conv2d(
                    pool2,
                    kernel,
                    [1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv3_1 = tf.nn.bias_add(conv3_1, biases, data_format=nn_data_format)
                net["conv3_1"] = conv3_1 = tf.nn.relu(conv3_1, name=scope)

            with tf.name_scope('conv3_2') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv3_2_W"]),
                    trainable=False,
                    name='conv3_2_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv3_2_b"]),
                    trainable=False,
                    name='conv3_2_b')
                conv3_2 = tf.nn.conv2d(
                    conv3_1,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv3_2 = tf.nn.bias_add(conv3_2, biases, data_format=nn_data_format)
                net["conv3_2"] = conv3_2 = tf.nn.relu(conv3_2, name=scope)

            with tf.name_scope('conv3_3') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv3_3_W"]),
                    trainable=False,
                    name='conv3_3_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv3_3_b"]),
                    trainable=False,
                    name='conv3_3_b')
                conv3_3 = tf.nn.conv2d(
                    conv3_2,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv3_3 = tf.nn.bias_add(conv3_3, biases, data_format=nn_data_format)
                net["conv3_3"] = conv3_3 = tf.nn.relu(conv3_3, name=scope)

            pool3 = tf.nn.avg_pool(
                conv3_3,
                ksize=[1, 2, 2, 1] if nn_data_format == "NHWC" else [1, 1, 2, 2],
                strides=[1, 2, 2, 1] if nn_data_format == "NHWC" else [1, 1, 2, 2],
                padding='SAME',
                data_format=nn_data_format,
                name='pool3')

            with tf.name_scope('conv4_1') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv4_1_W"]),
                    trainable=False,
                    name='conv4_1_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv4_1_b"]),
                    trainable=False,
                    name='conv4_1_b')
                conv4_1 = tf.nn.conv2d(
                    pool3,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv4_1 = tf.nn.bias_add(conv4_1, biases, data_format=nn_data_format)
                net["conv4_1"] = conv4_1 = tf.nn.relu(conv4_1, name=scope)

            with tf.name_scope('conv4_2') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv4_2_W"]),
                    trainable=False,
                    name='conv4_2_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv4_2_b"]),
                    trainable=False,
                    name='conv4_2_b')
                conv4_2 = tf.nn.conv2d(
                    conv4_1,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv4_2 = tf.nn.bias_add(conv4_2, biases, data_format=nn_data_format)
                net["conv4_2"] = conv4_2 = tf.nn.relu(conv4_2, name=scope)

            with tf.name_scope('conv4_3') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv4_3_W"]),
                    trainable=False,
                    name='conv4_3_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv4_3_b"]),
                    trainable=False,
                    name='conv4_3_b')
                conv4_3 = tf.nn.conv2d(
                    conv4_2,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv4_3 = tf.nn.bias_add(conv4_3, biases, data_format=nn_data_format)
                net["conv4_3"] = conv4_3 = tf.nn.relu(conv4_3, name=scope)

            pool4 = tf.nn.avg_pool(
                conv4_3,
                ksize=[1, 2, 2, 1] if nn_data_format == "NHWC" else [1, 1, 2, 2],
                strides=[1, 2, 2, 1] if nn_data_format == "NHWC" else [1, 1, 2, 2],
                padding='SAME',
                data_format=nn_data_format,
                name='pool4')

            with tf.name_scope('conv5_1') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv5_1_W"]),
                    trainable=False,
                    name='conv5_1_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv5_1_b"]),
                    trainable=False,
                    name='conv5_1_b')
                conv5_1 = tf.nn.conv2d(
                    pool4,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv5_1 = tf.nn.bias_add(conv5_1, biases, data_format=nn_data_format)
                net["conv5_1"] = conv5_1 = tf.nn.relu(conv5_1, name=scope)

            with tf.name_scope('conv5_2') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv5_2_W"]),
                    trainable=False,
                    name='conv5_2_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv5_2_b"]),
                    trainable=False,
                    name='conv5_2_b')
                conv5_2 = tf.nn.conv2d(
                    conv5_1,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv5_2 = tf.nn.bias_add(conv5_2, biases, data_format=nn_data_format)
                net["conv5_2"] = conv5_2 = tf.nn.relu(conv5_2, name=scope)

            with tf.name_scope('conv5_3') as scope:
                kernel = tf.get_variable(
                    initializer=tf.constant(weights["conv5_3_W"]),
                    trainable=False,
                    name='conv5_3_W')
                biases = tf.get_variable(
                    initializer=tf.constant(weights["conv5_3_b"]),
                    trainable=False,
                    name='conv5_3_b')
                conv5_3 = tf.nn.conv2d(
                    conv5_2,
                    filter=kernel,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format=nn_data_format)
                conv5_3 = tf.nn.bias_add(conv5_3, biases, data_format=nn_data_format)
                net["conv5_3"] = tf.nn.relu(conv5_3, name=scope)

        return net

    return vgg16


def gram_matrix(x, data_format="channels_last"):
    """Calculate gram matrix

    :param x: Input tensor
    :param data_format: Either 'channels_last' or 'channels_first'
    :return: Gram matrix (self inner product of x_reshape)
    """

    inferred_shape = tf.shape(x)
    if data_format == "channels_last":
        transpose_a = True
        transpose_b = False
        reshape_shape = tf.stack([inferred_shape[0], inferred_shape[1] * inferred_shape[2], inferred_shape[3]])
    else:
        transpose_a = False
        transpose_b = True
        reshape_shape = tf.stack([inferred_shape[0], inferred_shape[1], inferred_shape[2] * inferred_shape[3]])
    x = tf.reshape(x, reshape_shape)
    return tf.matmul(x, x, transpose_a=transpose_a, transpose_b=transpose_b)


def create_loss_fn(data_format="channels_last"):
    """Higher order function to create loss_fn (w/o total variation loss)

    :param data_format: Either 'channels_last' or 'channels_first'
    :return: loss_fn with signature as follows:
                target_content_features
                target_style_features
                transferred_content_features
                transferred_style_features
                content_loss_weight
                style_loss_weight
    """

    def loss_fn(target_content_features,
                target_style_features,
                transferred_content_features,
                transferred_style_features,
                content_loss_weight,
                style_loss_weight):

        assert len(target_content_features) == len(transferred_content_features)
        assert len(target_style_features) == len(transferred_style_features)

        content_loss = 0
        for i in range(len(transferred_content_features)):
            content_loss = content_loss \
                           + 2 * tf.nn.l2_loss(target_content_features[i] - transferred_content_features[i])

        style_loss = 0
        for i in range(len(transferred_style_features)):
            _, w, h, ch = target_style_features[i].shape.as_list()
            gram_target = gram_matrix(target_style_features[i], data_format=data_format)
            gram_transferred = gram_matrix(transferred_style_features[i], data_format=data_format)
            style_loss = style_loss \
                + (2 * tf.nn.l2_loss(gram_target - gram_transferred)) / (4 * ch ** 2 * (w * h) ** 2)

        return content_loss_weight * content_loss + style_loss_weight * style_loss

    return loss_fn
