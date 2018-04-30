import argparse
import shutil

import tensorflow as tf
from tensorflow.contrib import training as tf_training
from tensorflow.python.lib.io import file_io

from core.loss import create_loss_fn, create_loss_model_fn
from core.model import create_model_fn
from utils.data_utils import create_inputs_fn
from utils.train_utils import create_estimator_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Perceptual Losses for Real-Time Style Transfer and Super-Resolution')
    parser.add_argument('--restart_training', default=False, help='Learning rate.', action='store_true')
    parser.add_argument(
        '--content_files',
        default=['train-00001-of-00001'],
        help='List of content tfrecords.',
        type=str,
        nargs='+')
    parser.add_argument(
        '--style_files',
        default=['styletrain-00001-of-00001'],
        help='List of style tfrecords.',
        type=str,
        nargs='+')
    parser.add_argument('--job-dir', default='ckpts', help='Path to vgg weights.', type=str)
    parser.add_argument('--vgg_path', default='vgg_small.npz', help='Path to vgg weights.', type=str)
    parser.add_argument('--data_format', default='channels_last', help='Path to vgg weights.', type=str)
    parser.add_argument('--num_epochs', default=2, help='Number of iterations.', type=int)
    parser.add_argument('--batch_size', default=8, help='Learning rate.', type=int)
    parser.add_argument('--shuffle_buffer_size', default=32, help='Learning rate.', type=int)
    parser.add_argument('--learning_rate', default=1e-3, help='Learning rate.', type=int)
    parser.add_argument('--image_size', default=256, help='Learning rate.', type=int)
    parser.add_argument(
        '--content_features',
        default=['conv2_2'],
        help='List of features map to be used as content representation.',
        type=str,
        nargs='+')
    parser.add_argument(
        '--style_features',
        default=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3'],
        help='List of features map to be used as style representation.',
        type=str,
        nargs='+')
    parser.add_argument('--content_weight', default=1, help='Content loss weight.', type=float)
    parser.add_argument('--style_weight', default=1e4, help='Style loss weight.', type=float)
    parser.add_argument('--log_iter', default=50, help='Log interval.', type=int)
    parser.add_argument('--checkpoint_iter', default=50, help='Log interval.', type=int)
    return parser.parse_args()


def run_experiment(args):
    if args.restart_training:
        shutil.rmtree(args.job_dir, ignore_errors=True)

    content_size = style_size = (args.image_size, args.image_size)

    num_content_samples = sum(1 for f in file_io.get_matching_files(args.content_files)
                            for _ in tf.python_io.tf_record_iterator(f))

    num_style_samples = sum(1 for f in file_io.get_matching_files(args.style_files)
                            for _ in tf.python_io.tf_record_iterator(f))

    print("Number of training content samples: " + str(num_content_samples))
    print("Number of training style samples: " + str(num_style_samples))

    model_fn = create_model_fn(data_format=args.data_format)

    loss_model_fn = create_loss_model_fn(weights_path=args.vgg_path, data_format=args.data_format)

    loss_fn = create_loss_fn(data_format=args.data_format)

    estimator_fn = create_estimator_fn(
        model_fn=model_fn,
        loss_model_fn=loss_model_fn,
        loss_fn=loss_fn,
        data_format=args.data_format)

    config = tf.estimator.RunConfig(
        tf_random_seed=42,
        save_summary_steps=args.log_iter,
        save_checkpoints_steps=args.checkpoint_iter,
        log_step_count_steps=args.log_iter,
        model_dir=args.job_dir)

    params = tf_training.HParams(
        learning_rate=args.learning_rate,
        content_features=args.content_features,
        style_features=args.style_features,
        content_weight=args.content_weight,
        style_weight=args.style_weight)

    estimator = tf.estimator.Estimator(
        model_fn=estimator_fn,
        params=params,
        config=config)

    style_epochs = args.num_epochs * num_content_samples // args.batch_size

    train_inputs_fn = create_inputs_fn(
        content_tfrecords=args.content_files,
        style_tfrecords=args.style_files,
        content_size=content_size,
        style_size=style_size,
        content_epochs=args.num_epochs,
        style_epochs=style_epochs,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        scope="train_inputs")

    estimator.train(input_fn=train_inputs_fn)
