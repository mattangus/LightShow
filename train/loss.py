import tensorflow as tf

def build_loss(logits, labels):
    return tf.losses.softmax_cross_entropy(
    onehot_labels=labels,
    logits=logits)