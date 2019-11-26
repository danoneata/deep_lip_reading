import os
import pdb
import time

import tensorflow as tf

from main import (
    load_args,
    load_checkpoints,
    setup_generators,
    TransformerTrainGraph,
)


CONFIG = load_args()
LOG_DIR = "output/tensorboard"
MODEL_DIR = "output/model"
NUM_EPOCHS = 4


def init_models_and_data():

    gen, epoch_size = setup_generators()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.gpu_id)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=sess_config)

    shapes_and_types = TransformerTrainGraph.get_model_input_target_shapes_and_types()
    (shapes_in, dtypes_in), (shapes_out, dtypes_out) = shapes_and_types

    go_idx = gen.label_vectorizer.char_indices[gen.label_vectorizer.go_token]
    chars = gen.label_vectorizer.chars

    x = tf.placeholder(dtypes_in[0], shape=shapes_in[0])
    y = tf.placeholder(dtypes_out[0], shape=shapes_out[0])

    g = TransformerTrainGraph(
        x,
        y,
        is_training=True,
        reuse=tf.AUTO_REUSE,
        go_token_index=go_idx,
        chars=chars,
    )
    print("→  Graph loaded")

    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())

    load_checkpoints(sess)

    return g, epoch_size, chars, sess, gen

g, epoch_size, chars, sess, gen = init_models_and_data()

# Instrument for tensorboard
summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter(os.path.join(LOG_DIR, time.strftime("%Y-%m-%d-%H-%M-%S")))
# writer.add_graph(sess.graph)

saver = tf.train.Saver(tf.global_variables())
global_step = 0

for e in range(NUM_EPOCHS):
    print(e)
    while True:
        try:
            x, y, _ = gen.next()
        except:
            break
        feed_dict = {g.x: x[0], g.y: y[0]}
        summ, cer, _ = sess.run([summaries, g.cer, g.train_op], feed_dict)
        writer.add_summary(summ, global_step)
        global_step += 1
        print('.', end=' ')
    print()
    gen, _ = setup_generators()

checkpoint_path = os.path.join(MODEL_DIR, "model.ckpt")
saver.save(sess, checkpoint_path)
print("model saved to {}".format(checkpoint_path))
