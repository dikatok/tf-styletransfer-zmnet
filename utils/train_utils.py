import tensorflow as tf


def create_train_op(loss,
                    learning_rate):
    """

    :param loss: Loss tensor
    :param learning_rate: Initial learning rate
    :return: Operation
    """

    global_step = tf.train.get_global_step()

    with tf.variable_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)

    return train_op


def create_estimator_fn(model_fn, loss_model_fn, loss_fn, data_format="channels_last"):
    """Higher order function to create estimator_fn

    :param model_fn: Fn to create model network with signature as follows:
            inputs: Input image tensor
    :param loss_model_fn: Fn to create model used in loss calculation with signature as follows:
            inputs: Input image tensor
    :param loss_fn: Fn to create loss tensor with signature as follows:
            target_content_features
            target_style_features
            transferred_content_features
            transferred_style_features
            content_loss_weight
            style_loss_weight
    :param data_format: Either 'channels_last' or 'channels_first'
    :return: EstimatorSpec
    """

    def estimator_fn(features, labels, mode, params):
        contents = tf.identity(features["contents"], name="contents")
        styles = tf.identity(features["styles"], name="styles")

        if data_format == "channels_first":
            contents = tf.transpose(contents, perm=[0, 3, 1, 2])
            styles = tf.transpose(styles, perm=[0, 3, 1, 2])

        transferred = tf.identity(model_fn(contents, styles), name="transferred")

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=transferred,
                export_outputs={
                    "transferred": tf.estimator.export.PredictOutput(transferred)
                })

        content_net = loss_model_fn(contents)
        style_net = loss_model_fn(styles)
        transferred_net = loss_model_fn(transferred)

        target_content_features = [content_net[layer] for layer in params.content_features]
        target_style_features = [style_net[layer] for layer in params.style_features]

        transferred_content_features = [transferred_net[layer] for layer in params.content_features]
        transferred_style_features = [transferred_net[layer] for layer in params.style_features]

        loss = loss_fn(
            target_content_features=target_content_features,
            target_style_features=target_style_features,
            transferred_content_features=transferred_content_features,
            transferred_style_features=transferred_style_features,
            content_loss_weight=params.content_weight,
            style_loss_weight=params.style_weight)

        if data_format == "channels_first":
            styles = tf.transpose(styles, perm=[0, 2, 3, 1])
            contents = tf.transpose(contents, perm=[0, 2, 3, 1])
            transferred = tf.transpose(transferred, perm=[0, 2, 3, 1])

        tf.summary.image("style_summary", styles)

        tf.summary.image("content_and_output_summary", tf.concat([contents, transferred], axis=2))

        tf.summary.scalar("loss", loss)

        train_op = create_train_op(loss, params.learning_rate)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=transferred,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={})

    return estimator_fn
