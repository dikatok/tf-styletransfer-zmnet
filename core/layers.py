import tensorflow as tf


def instance_norm(inputs,
                  name,
                  epsilon=1e-5,
                  data_format="channels_last",
                  gamma=None,
                  beta=None):
    gamma_beta_shape = [1, 1, inputs.shape[3]] if data_format == "channels_last" else [inputs.shape[1], 1, 1]
    spatial_axes = [1, 2] if data_format == "channels_last" else [2, 3]
    with tf.variable_scope(name):
        if gamma is None:
            gamma = tf.get_variable(shape=gamma_beta_shape, name="gamma")
        if beta is None:
            beta = tf.get_variable(shape=gamma_beta_shape, name="beta")
        mean, var = tf.nn.moments(inputs, axes=spatial_axes, keep_dims=True)
        outputs = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon, name="norm")
    return outputs


def conv(inputs,
         name,
         filters,
         kernel_size,
         strides,
         with_bn=True,
         with_relu=True,
         data_format="channels_last",
         bn_gamma=None,
         bn_beta=None):
    """Convolution

    :param inputs: Input tensor
    :param name: Variable scope
    :param data_format: Either 'channels_last' or 'channels_first'
    :param filters: Filter size
    :param kernel_size: Kernel size
    :param strides: Strides
    :param with_bn: Either use instance normalization before non-linearity or not
    :param with_relu: Either use relu on output or not
    :return: Output tensor
    """

    padding = kernel_size // 2
    with tf.variable_scope(name):
        if data_format == "channels_last":
            paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]]
        else:
            paddings = [[0, 0], [0, 0], [padding, padding], [padding, padding]]
        outputs = tf.pad(
            inputs,
            paddings=paddings,
            mode="REFLECT")
        outputs = tf.layers.conv2d(
            outputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            name="conv",
            data_format=data_format)
        if with_bn:
            outputs = instance_norm(
                outputs,
                name="inorm",
                data_format=data_format,
                gamma=bn_gamma,
                beta=bn_beta)
        if with_relu:
            outputs = tf.nn.relu(outputs, name="act")
    return outputs


def residual_block(inputs,
                   name,
                   filters,
                   kernel_size,
                   data_format="channels_last",
                   bn_gamma1=None,
                   bn_gamma2=None,
                   bn_beta1=None,
                   bn_beta2=None):
    """Residual block

    :param inputs: Input tensor
    :param name: Variable scope
    :param data_format: Either 'channels_last' or 'channels_first'
    :param filters: Filter size
    :param kernel_size: Kernel size
    :return: Output tensor
    """

    with tf.variable_scope(name):
        residual = inputs
        outputs = conv(
            inputs,
            name="conv1",
            data_format=data_format,
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            bn_gamma=bn_gamma1,
            bn_beta=bn_beta1)
        outputs = conv(
            outputs,
            name="conv2",
            data_format=data_format,
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            with_relu=False,
            bn_gamma=bn_gamma2,
            bn_beta=bn_beta2)
        outputs = tf.nn.relu(outputs + residual)
    return outputs


def upsample(inputs,
             name,
             data_format,
             filters,
             kernel_size,
             strides,
             bn_gamma=None,
             bn_beta=None):
    """Upsample (resize -> conv) instead of transposed_conv or conv w/ 0.5 strides

    :param inputs: Input tensor
    :param name: Variable scope
    :param data_format: Either 'channels_last' or 'channels_first'
    :param filters: Filter size
    :param kernel_size: Kernel size
    :param strides: Strides
    :return: Output tensor
    """

    shape = inputs.shape.as_list()
    inferred_shape = tf.shape(inputs)

    spatial_axis = [1, 2] if data_format == "channels_last" else [2, 3]
    w, h = shape[spatial_axis[0]], shape[spatial_axis[1]]
    if w is None:
        w, h = inferred_shape[spatial_axis[0]], inferred_shape[spatial_axis[1]]

    with tf.variable_scope(name):
        if data_format == "channels_first":
            inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
        outputs = tf.image.resize_images(inputs, size=[w * strides, h * strides])
        if data_format == "channels_first":
            outputs = tf.transpose(outputs, perm=[0, 3, 1, 2])
        outputs = conv(
            outputs,
            name="conv",
            data_format=data_format,
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            bn_gamma=bn_gamma,
            bn_beta=bn_beta)
    return outputs
