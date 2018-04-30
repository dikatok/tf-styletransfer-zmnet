from io import BytesIO

import tensorflow as tf
from tensorflow.python.lib.io import file_io


def _create_parse_record_fn(image_size: (int, int)):
    """Higher order function to create parse_record_fn (since images in batch must have the same shape)

    :param image_size: standard input size
    :return: parse_record_fn with signature as follows: example
    """

    def parse_record_fn(example: tf.train.Example):
        features = {
            "image": tf.FixedLenFeature((), tf.string),
            "image_filename": tf.FixedLenFeature((), tf.string),
        }

        parsed_example = tf.parse_single_example(example, features)

        image = tf.cast(tf.image.decode_jpeg(parsed_example["image"], channels=3), dtype=tf.float32)

        image = tf.image.resize_images(image, image_size)

        return image

    return parse_record_fn


def _create_one_shot_iterator(tfrecord_filenames: [str],
                              num_epochs: int,
                              batch_size: int,
                              image_size: (int, int),
                              shuffle_buffer_size: int):
    """Create one shot iterator

    :param tfrecord_filenames: List of tfrecord files
    :param num_epochs: Number of epochs
    :param batch_size: Batch size
    :param image_size: Input image size
    :param shuffle_buffer_size: Buffer size to shuffle (higher means better shuffle but consumes more resources)
    :return: Iterator
    """

    dataset = tf.data.TFRecordDataset(tfrecord_filenames)

    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.map(_create_parse_record_fn(image_size))

    dataset = dataset.prefetch(batch_size)

    dataset = dataset.repeat(num_epochs)

    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator()


def create_inputs_fn(content_tfrecords: [str],
                     style_tfrecords: [str],
                     content_size: [int, int],
                     style_size: [int, int],
                     content_epochs: int,
                     style_epochs: int,
                     batch_size: int,
                     shuffle_buffer_size: int,
                     scope: str):
    """Higher order function to create inputs_fn

    :param content_tfrecords: List of tfrecords files
    :param style_img: Style image to use
    :param content_size: Content image size
    :param style_size: Style image size
    :param num_epochs: Number of epochs
    :param batch_size: Batch size
    :param shuffle_buffer_size: Buffer size to shuffle (higher means better shuffle but consumes more resources)
    :param scope: Variable scope
    :return: inputs_fn
    """

    def inputs_fn():

        with tf.variable_scope(scope), tf.device("/cpu:0"):
            style_iterator = _create_one_shot_iterator(
                tfrecord_filenames=style_tfrecords,
                num_epochs=style_epochs,
                batch_size=1,
                image_size=style_size,
                shuffle_buffer_size=shuffle_buffer_size)

            content_iterator = _create_one_shot_iterator(
                tfrecord_filenames=content_tfrecords,
                num_epochs=content_epochs,
                batch_size=batch_size,
                image_size=content_size,
                shuffle_buffer_size=shuffle_buffer_size)

            styles = style_iterator.get_next()
            contents = content_iterator.get_next()

        return {"contents": contents, "styles": styles}

    return inputs_fn
