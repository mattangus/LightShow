import tensorflow as tf
from model import build_model
from dataset import build_dataset, num_classes
from loss import build_loss

batch = 4

dataset = build_dataset(batch)
dataset = dataset.apply(tf.data.experimental.ignore_errors())
iterator = dataset.make_one_shot_iterator()
inputs = iterator.get_next()

with tf.device("gpu:0"):
    logits = build_model(inputs["spectrogram"], 1.0, num_classes, True)

loss = build_loss(logits, inputs["label"])

train_step = tf.train.AdamOptimizer().minimize(loss)

with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True)) as sess:
    
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    sess.run(train_step)
    # while True:
    #     sess.run(train_step)
    #     cur_loss = sess.run(loss)
    #     print("cur:", cur_loss, "      ")