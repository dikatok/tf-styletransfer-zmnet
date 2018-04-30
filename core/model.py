import tensorflow as tf

from core.layers import conv, residual_block, upsample


def _dense(inputs, kernel_size, name):
    shape = inputs.shape.as_list()
    with tf.variable_scope(name):
        outputs = conv(inputs, filters=shape[3], kernel_size=kernel_size, strides=1, with_bn=False, with_relu=False, name="conv")
        outputs = tf.reduce_mean(outputs, axis=[1, 2], keepdims=True)
    return outputs


def _pnet_fc(pnet_outs):
    gammas = {}
    betas = {}

    # param_kernels = zip(["conv1", "conv2", "conv3", "up1", "up2"],
    #                     [256, 128, 64, 128, 256])

    param_kernels = zip(["conv1", "conv2", "conv3", "up1", "up2"],
                        [3, 3, 3, 3, 3])

    for param_id, kernel_size in param_kernels:
        gamma_name = param_id + "_gamma_dense"
        beta_name = param_id + "_beta_dense"
        gammas[param_id] = _dense(pnet_outs[param_id], kernel_size=kernel_size, name=gamma_name)
        betas[param_id] = _dense(pnet_outs[param_id], kernel_size=kernel_size, name=beta_name)

    for res_id, conv_id in list(zip(list(range(1, 6, 1)) * 2,
                                    [1] * 5 + [2] * 5)):
        param_id = "res{res_id}_{conv_id}".format(res_id=res_id, conv_id=conv_id)
        tensor_name = "pnet/res{res_id}/conv{conv_id}/inorm/norm/add_1:0".format(res_id=res_id, conv_id=conv_id)
        gamma_name = param_id + "_gamma_dense"
        beta_name = param_id + "_beta_dense"
        gammas[param_id] = _dense(tf.get_default_graph().get_tensor_by_name(tensor_name), kernel_size=3, name=gamma_name)
        betas[param_id] = _dense(tf.get_default_graph().get_tensor_by_name(tensor_name), kernel_size=3, name=beta_name)

    return gammas, betas


def _pnet(styles, data_format="channels_last"):
    with tf.variable_scope("pnet", reuse=tf.AUTO_REUSE):
        pnet_outs = {}

        inputs = ((styles / 255.) - 0.5) * 2
        pnet_outs["conv1"] = \
            conv1 = conv(inputs, name="conv1", data_format=data_format, filters=32, kernel_size=9, strides=1)
        pnet_outs["conv2"] = \
            conv2 = conv(conv1, name="conv2", data_format=data_format, filters=64, kernel_size=3, strides=2)
        pnet_outs["conv3"] = \
            conv3 = conv(conv2, name="conv3", data_format=data_format, filters=128, kernel_size=3, strides=2)
        res1 = residual_block(conv3, name="res1", data_format=data_format, filters=128, kernel_size=3)
        res2 = residual_block(res1, name="res2", data_format=data_format, filters=128, kernel_size=3)
        res3 = residual_block(res2, name="res3", data_format=data_format, filters=128, kernel_size=3)
        res4 = residual_block(res3, name="res4", data_format=data_format, filters=128, kernel_size=3)
        res5 = residual_block(res4, name="res5", data_format=data_format, filters=128, kernel_size=3)
        pnet_outs["up1"] \
            = up1 = upsample(res5, name="up1", data_format=data_format,  filters=64, kernel_size=3, strides=2)
        pnet_outs["up2"] \
            = _ = upsample(up1, name="up2", data_format=data_format, filters=32, kernel_size=3, strides=2)

        gammas, betas = _pnet_fc(pnet_outs)

    return gammas, betas


def _tnet(contents, gammas, betas, data_format="channels_last"):
    with tf.variable_scope("tnet", reuse=tf.AUTO_REUSE):
        inputs = ((contents / 255.) - 0.5) * 2
        conv1 = conv(
            inputs,
            name="conv1",
            data_format=data_format,
            filters=32,
            kernel_size=9,
            strides=1,
            bn_gamma=gammas["conv1"],
            bn_beta=betas["conv1"])
        conv2 = conv(
            conv1,
            name="conv2",
            data_format=data_format,
            filters=64,
            kernel_size=3,
            strides=2,
            bn_gamma=gammas["conv2"],
            bn_beta=betas["conv2"])
        conv3 = conv(
            conv2,
            name="conv3",
            data_format=data_format,
            filters=128,
            kernel_size=3,
            strides=2,
            bn_gamma=gammas["conv3"],
            bn_beta=betas["conv3"])
        res1 = residual_block(
            conv3,
            name="res1",
            data_format=data_format,
            filters=128,
            kernel_size=3,
            bn_gamma1=gammas["res1_1"],
            bn_beta1=betas["res1_1"],
            bn_gamma2=gammas["res1_2"],
            bn_beta2=betas["res1_2"])
        res2 = residual_block(
            res1,
            name="res2",
            data_format=data_format,
            filters=128,
            kernel_size=3,
            bn_gamma1=gammas["res2_1"],
            bn_beta1=betas["res2_1"],
            bn_gamma2=gammas["res2_2"],
            bn_beta2=betas["res2_2"])
        res3 = residual_block(
            res2,
            name="res3",
            data_format=data_format,
            filters=128,
            kernel_size=3,
            bn_gamma1=gammas["res3_1"],
            bn_beta1=betas["res3_1"],
            bn_gamma2=gammas["res3_2"],
            bn_beta2=betas["res3_2"])
        res4 = residual_block(
            res3,
            name="res4",
            data_format=data_format,
            filters=128,
            kernel_size=3,
            bn_gamma1=gammas["res4_1"],
            bn_beta1=betas["res4_1"],
            bn_gamma2=gammas["res4_2"],
            bn_beta2=betas["res4_2"])
        res5 = residual_block(
            res4,
            name="res5",
            data_format=data_format,
            filters=128,
            kernel_size=3,
            bn_gamma1=gammas["res5_1"],
            bn_beta1=betas["res5_1"],
            bn_gamma2=gammas["res5_2"],
            bn_beta2=betas["res5_2"])
        up1 = upsample(
            res5,
            name="up1",
            data_format=data_format,
            filters=64,
            kernel_size=3,
            strides=2,
            bn_gamma=gammas["up1"],
            bn_beta=betas["up1"])
        up2 = upsample(
            up1,
            name="up2",
            data_format=data_format,
            filters=32,
            kernel_size=3,
            strides=2,
            bn_gamma=gammas["up2"],
            bn_beta=betas["up2"])
        conv4 = conv(
            up2,
            name="conv4",
            data_format=data_format,
            filters=3,
            kernel_size=9,
            strides=1,
            with_bn=False,
            with_relu=False)
        out = tf.clip_by_value(conv4, 0., 255.)
    return out


def create_model_fn(data_format="channels_last"):
    """Higher order function to create model_fn

    :param data_format: Either 'channels_last' or 'channels_first'
    :return: model_fn with signature as follows: inputs
    """

    def model_fn(contents, styles):
        gammas, betas = _pnet(styles, data_format=data_format)

        out = _tnet(contents, data_format=data_format, gammas=gammas, betas=betas)

        return out

    return model_fn
