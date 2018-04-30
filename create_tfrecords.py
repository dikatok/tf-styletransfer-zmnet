import argparse
import os
import random
import sys
import threading
from datetime import datetime
from glob import glob
from queue import Queue

import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    """

    :param value:
    :return:
    """

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(
        image_buffer,
        image_filename):
    """

    :param image_buffer:
    :param image_filename:
    :return:
    """

    image_filename = os.path.split(image_filename)[-1]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image_buffer),
        "image_filename": _bytes_feature(bytes(image_filename, encoding="UTF-8")),
    }))
    return example


def _process_image_files_batch(thread_index,
                               batch_data,
                               shards,
                               total_shards,
                               output_dir,
                               output_name,
                               error_queue):
    """

    :param thread_index:
    :param batch_data:
    :param shards:
    :param total_shards:
    :param output_dir:
    :param output_name:
    :param error_queue:
    :return:
    """

    batch_size = len(batch_data)
    batch_per_shard = batch_size // len(shards)

    counter = 0
    error_counter = 0
    for s in range(len(shards)):
        shard = shards[s]
        output_filename = '%s-%.5d-of-%.5d' % (output_name, shard, total_shards)
        output_file = os.path.join(output_dir, output_filename)

        writer = tf.python_io.TFRecordWriter(output_file)
        shard_counter = 0
        shard_range = (s * batch_per_shard, min(batch_per_shard, batch_size - (s * batch_per_shard)))
        files_in_shard = np.arange(shard_range[0], shard_range[1], dtype=int)
        for i in files_in_shard:
            image_filename = data[i]
            try:
                image_buffer = tf.gfile.FastGFile(image_filename, 'rb').read()
                example = _convert_to_example(image_buffer, image_filename)
                writer.write(example.SerializeToString())
                shard_counter += 1
                counter += 1
            except StopIteration as e:
                error_counter += 1
                error_msg = repr(e)
                error_queue.put(error_msg)

        print('%s [thread %d]: Wrote %d images to %s, with %d errors.' %
              (datetime.now(), thread_index, shard_counter, output_file, error_counter))
        sys.stdout.flush()

    print('%s [thread %d]: Wrote %d images to %d shards, with %d errors.' %
          (datetime.now(), thread_index, counter, len(shards), error_counter))
    sys.stdout.flush()


def create(data,
           output_dir,
           output_name,
           num_shards,
           num_threads):
    """

    :param data:
    :param output_name:
    :param output_dir:
    :param num_shards:
    :param num_threads:
    :return:
    """

    num_data_per_thread = len(data) // num_threads
    num_shard_per_thread = num_shards // num_threads
    batch_data_ranges = [(i * num_data_per_thread, min(num_data_per_thread, len(data) - i * num_data_per_thread))
                         for i in range(num_threads)]

    coord = tf.train.Coordinator()

    error_queue = Queue()

    threads = []
    for thread_index in range(1, num_threads + 1):
        batch_ranges = batch_data_ranges[thread_index - 1]
        batch_data = data[batch_ranges[0]:batch_ranges[1]]
        shards = [thread_index + (thread_index - 1) * (num_shard_per_thread - 1) + shard
                  for shard in range(num_shard_per_thread)]
        args = (
            thread_index,
            batch_data,
            shards,
            num_shards,
            output_dir,
            output_name,
            error_queue)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)

    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())
    print('%d examples failed.' % (len(errors),))


def parse_args():
    parser = argparse.ArgumentParser(description='Perceptual Losses for Real-Time Style Transfer and Super-Resolution')
    parser.add_argument('--image_dir', help='Directory of the images.', type=str, required=True)
    parser.add_argument('--image_suffix', help='Extension of the images.', type=str, default=".jpg")
    parser.add_argument('--output_dir', help='Directory to save tfrecords.', type=str, default="./")
    parser.add_argument('--output_prefix', help='Prefix to save tfrecords.', type=str, default="")
    parser.add_argument('--shards', help='Number of shards to make.', type=int, default=1)
    parser.add_argument('--threads', help='Number of threads to use.', type=int, default=1)
    parser.add_argument('--shuffle', help='Shuffle the samples.', action='store_true', default=True)
    parsed_args = parser.parse_args()
    return parsed_args


def assert_args(args):
    assert os.path.exists(args.image_dir), "Images directory does not exist"
    assert args.shards > 0, "Number of shards must be > 0"
    assert args.threads > 0, "Number of threads must be > 0"
    assert args.shards % args.threads == 0


if __name__ == '__main__':
    args = parse_args()
    assert_args(args)

    data = glob(os.path.join(args.image_dir, "*" + args.image_suffix))

    if args.shuffle:
        random.shuffle(data)

    create(data, args.output_dir, args.output_prefix + "train", args.shards, args.threads)
