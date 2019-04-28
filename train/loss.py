import tensorflow as tf

import helpers

def _get_loss(logits, labels, num_class, weights):
    one_hot = tf.one_hot(labels, num_class)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot, logits=logits, weights=weights)

    return loss

def build_loss(emo_logits, sent_logits, inputs):
    with tf.variable_scope("Loss"):
        emo_loss = _get_loss(emo_logits, inputs[helpers.EMO_KEY], helpers.num_emotions, inputs[helpers.EMO_WEIGHT_KEY])
        sent_loss = _get_loss(sent_logits, inputs[helpers.SENT_KEY], helpers.num_sentiments, inputs[helpers.SENT_WEIGHT_KEY])

        with tf.variable_scope("Weight_Decay"):
            all_vars = tf.trainable_variables()
            weight_loss = tf.add_n([tf.nn.l2_loss(v) for v in all_vars])

    return emo_loss, sent_loss, weight_loss