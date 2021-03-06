import tensorflow as tf
import argparse
from tensorflow.python import pywrap_tensorflow
import matplotlib.cm as cm
import numpy as np

from model import build_model
from datasets.dataset import build_dataset, num_classes
import helpers
from preprocess import apply_preprocessing
from loss import build_loss

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, required=True, help="Where to save checkpoints and summaries")
parser.add_argument("--pre_train", type=str, default="pre_train/mobilenet_v1_1.0_224.ckpt", help="checkpoint to fine tune from")
parser.add_argument("--datasets", nargs="+", required=True, help="list of datasets to use")

args = parser.parse_args()

def batch_acc(logits, labels):
    prediction = tf.nn.softmax(logits)
    are_equal = tf.equal(tf.argmax(prediction, -1), labels)
    float_equal = tf.to_float(are_equal)
    accuracy = tf.reduce_mean(float_equal)
    return accuracy

datasets = set(args.datasets)
data_files = []
if "meld" in datasets:
    # data_files = ["data/meld_train.tfrecord-" + str(i) for i in range(36)]
    data_files += ["data/meld_train.tfrecord"]
if "tess" in datasets:
    data_files += ["data/tess_train.tfrecord-" + str(i) for i in range(2)]
if "savee" in datasets:
    data_files += ["data/savee_train.tfrecord-" + str(i) for i in range(1)]
if "crema" in datasets:
    data_files += ["data/crema_train.tfrecord-" + str(i) for i in range(1)]

print(data_files)

#make dataset and get elements
batch = 64
dataset = build_dataset(data_files, num_epoch=-1, num_readers=20, shuffle=True, shuffle_queue=125)
dataset = apply_preprocessing(dataset)
dataset = dataset.batch(batch)
iterator = dataset.make_one_shot_iterator()
inputs = iterator.get_next()

#make the model
emo_logits, sent_logits = build_model(inputs[helpers.DATA_KEY], 1.0, num_classes, True)

#make loss function and optimize
global_step = tf.train.get_or_create_global_step()
emo_loss, sent_loss, weight_loss = build_loss(emo_logits, sent_logits, inputs)
total_loss = emo_loss + 0.01 * weight_loss #+ sent_loss
lr = tf.train.linear_cosine_decay(0.01,global_step,args.num_steps)
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss, global_step=global_step)

#compute batch accuracy
emo_acc = batch_acc(emo_logits, inputs[helpers.EMO_KEY])
sent_acc = batch_acc(sent_logits, inputs[helpers.SENT_KEY])

#store summaries
min_val = tf.reduce_min(inputs[helpers.DATA_KEY], [1,2,3], keepdims=True)
log_spect = tf.log(inputs[helpers.DATA_KEY] - min_val + 1e-5)
colour_spect = helpers.colour_map_op(inputs[helpers.DATA_KEY] - min_val)
tf.summary.scalar("loss/total_loss", total_loss)
tf.summary.scalar("loss/emo_loss", emo_loss)
tf.summary.scalar("loss/sent_loss", sent_loss)
tf.summary.scalar("loss/weight_loss", weight_loss)
tf.summary.scalar("lr", lr)
tf.summary.scalar("accuracy/emo_acc", emo_acc)
tf.summary.scalar("accuracy/sent_acc", sent_acc)
tf.summary.image("spectrogram", colour_spect)
summary_op = tf.summary.merge_all()

#use saved checkpoints
checkpoint = tf.train.latest_checkpoint(args.log_dir)
to_load = None
if checkpoint is None:
    #there is no checkpoint
    checkpoint = args.pre_train

    #only load vars that are in the checkpoint file
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint)
    vars_in_check = reader.get_variable_to_shape_map()
    vars_in_graph = tf.trainable_variables()

    to_load = []
    for v in vars_in_graph:
        cur_name = v.name.split(":")[0]
        if cur_name in vars_in_check and tuple(vars_in_check[cur_name]) == tuple(v.shape):
            to_load.append(v)

restore_saver = tf.train.Saver(to_load)
init_ops = [tf.global_variables_initializer(), tf.local_variables_initializer()]
def init_fn(scaffold, session):
    session.run(init_ops)
    if to_load is not None:
        #checkpoint was not None after `latest_checkpoint`
        print("restoring from ", checkpoint)
        restore_saver.restore(session, checkpoint)

#build session
scaff = tf.train.Scaffold(
    init_fn=init_fn,
    local_init_op=tf.local_variables_initializer(),
    summary_op=summary_op,
    saver=tf.train.Saver())

hooks = [
    tf.train.StopAtStepHook(last_step=args.num_steps),
    tf.train.LoggingTensorHook({"loss" : total_loss, "emo_acc": emo_acc, "sent_acc": sent_acc, "global_step" : global_step}, every_n_iter=10),
]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.train.MonitoredTrainingSession(
        checkpoint_dir=args.log_dir,
        scaffold=scaff,
        config=config,
        hooks=hooks,
        save_summaries_secs=10,
        summary_dir=args.log_dir) as sess:

    while not sess.should_stop():
        step = tf.train.global_step(sess, global_step)
        sess.run(train_step)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

# with tf.Session(config=config) as sess:

#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

#     while True:
#         sess.run(train_step)
#         cur_loss = sess.run(loss)
#         print("cur:", cur_loss, "      ", end="\r")