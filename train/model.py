import tensorflow as tf
from mobilenet import mobilenet_v2
import helpers

def build_model(inputs, depth_multiplier, num_classes, is_training):
    with tf.variable_scope("Model"):
        logits, _ = mobilenet_v2.mobilenet(
            inputs,
            is_training=is_training,
            depth_multiplier=depth_multiplier,
            num_classes=num_classes)
            #global_pool=True)
        
        emo_logits = logits[:,:helpers.num_emotions]
        sent_logits = logits[:,helpers.num_emotions:helpers.num_emotions+helpers.num_sentiments]

    return emo_logits, sent_logits
    