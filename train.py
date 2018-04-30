import tensorflow as tf

from trainer.train import parse_args, run_experiment


def main(_):
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    tf.app.run(main=main)
