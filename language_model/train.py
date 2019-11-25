#!/usr/bin/env python
import argparse
import time
import os

from six.moves import cPickle

import tensorflow as tf

from utils import TextLoader
from char_rnn_lm import CharRnnLm as Model


def train(args):

    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    # Check compatibility if training is continued from previously saved model
    if args.init_from is not None:

        # Check if all necessary files exist
        assert os.path.isdir(args.init_from), "{} must be a a path".format(args.init_from)
        assert os.path.isfile(os.path.join(args.init_from, "config.pkl")), "config.pkl file does not exist in path " + args.init_from
        assert os.path.isfile(os.path.join(args.init_from, "chars_vocab.pkl")), "chars_vocab.pkl.pkl file does not exist in path " + args.init_from

        ckpt = tf.train.latest_checkpoint(args.init_from)
        assert ckpt, "No checkpoint found"

        # Open old config and check if models are compatible
        with open(os.path.join(args.init_from, "config.pkl"), "rb") as f:
            saved_model_args = cPickle.load(f)

        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]

        for checkme in need_be_same:
            err = f"Command line argument and saved model disagree on '{checkme}'"
            assert vars(saved_model_args)[checkme] == vars(args)[checkme], err

        # Open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, "chars_vocab.pkl"), "rb") as f:
            saved_chars, saved_vocab = cPickle.load(f)

        assert saved_chars == data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab == data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, "config.pkl"), "wb") as f:
        cPickle.dump(args, f)

    with open(os.path.join(args.save_dir, "chars_vocab.pkl"), "wb") as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args, training=True)

    with tf.Session() as sess:

        # Instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt)

        for e in range(args.num_epochs):

            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)

            for b in range(data_loader.num_batches):

                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}

                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h

                # instrument for tensorboard
                summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                writer.add_summary(summ, e * data_loader.num_batches + b)

                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(e * data_loader.num_batches + b, args.num_epochs * data_loader.num_batches, e, train_loss, end - start,))

                is_checkpoint = (e * data_loader.num_batches + b) % args.save_every == 0
                is_last = e == args.num_epochs - 1 and b == data_loader.num_batches - 1
                to_save = is_checkpoint or is_last

                if to_save:
                    checkpoint_path = os.path.join(args.save_dir, "model.ckpt")
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data and model checkpoints directories
    parser.add_argument("--data-dir", type=str, required=True, help="data directory containing input.txt with training examples")
    parser.add_argument("--save-dir", type=str, default="output/language-model", help="directory to store checkpointed models")
    parser.add_argument("--log-dir", type=str, default="output/logs", help="directory to store tensorboard logs")
    parser.add_argument("--save-every", type=int, default=1000, help="Save frequency. Number of passes between checkpoints of the model.")
    parser.add_argument("--init-from", type=str, default=None,
	help="""Continue training from saved model at this path (usually "save").
		Path must contain files saved by previous training process:
		'config.pkl'        : configuration;
		'chars_vocab.pkl'   : vocabulary definitions;
		'checkpoint'        : paths to model file(s) (created by tf).
				      Note: this file contains absolute paths, be careful when moving files around;
		'model.ckpt-*'      : file(s) with model definition (created by tf)
		 Model params must be the same between multiple runs (model, rnn_size, num_layers and seq_length).
		""")

    # Model params
    parser.add_argument("--rnn-size", type=int, default=128, help="size of RNN hidden state")
    parser.add_argument("--num-layers", type=int, default=2, help="number of layers in the RNN")

    # Optimization
    parser.add_argument("--seq-length", type=int, default=50, help="RNN sequence length. Number of timesteps to unroll for.")
    parser.add_argument("--batch-size", type=int, default=50, help="""Minibatch size. Number of sequences propagated through the network in parallel.  Pick batch-sizes to fully leverage the GPU (e.g. until the memory is filled up) commonly in the range 10-500.""")
    parser.add_argument("--num-epochs", type=int, default=50, help="number of epochs. Number of full passes through the training examples.")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="clip gradients at this value")
    parser.add_argument("--learning-rate", type=float, default=0.002, help="learning rate")
    parser.add_argument("--decay-rate", type=float, default=0.97, help="decay rate for rmsprop")
    parser.add_argument("--output-keep-prob", type=float, default=1.0, help="probability of keeping weights in the hidden layer")
    parser.add_argument("--input-keep-prob", type=float, default=1.0, help="probability of keeping weights in the input layer")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
