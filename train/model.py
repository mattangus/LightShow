import tensorflow as tf
from mobilenet_v1 import mobilenet_v1

def build_model(inputs, depth_multiplier, num_classes, is_training):
    logits, _ = mobilenet_v1(
            inputs,
            is_training=is_training,
            depth_multiplier=depth_multiplier,
            num_classes=num_classes,
            global_pool=True)
    
    return logits
    