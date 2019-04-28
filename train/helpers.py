import tensorflow as tf
import matplotlib.cm as cm
import numpy as np

#ps = pleasant surprise
emotion_to_id = {
    "angry": 0,
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "joy": 3,
    "neutral": 4,
    "ps": 5,
    "surprise": 5,
    "sad": 6,
    "sadness": 6,
}

sentiment_to_id = {
    "positive": 0,
    "neutral": 1,
    "negative": 2,
}

num_emotions = 7
num_sentiments = 3

EMO_KEY = "emotion_id"
SENT_KEY = "sent_id"
EMO_WEIGHT_KEY = "emotion_weight"
SENT_WEIGHT_KEY = "sent_weight"
DATA_KEY = "wav_data"


def _apply_colour_map(imgs):
    ret_imgs = []
    for img in imgs:
        cm_img = cm.viridis(img[...,0])[...,:-1]
        ret_imgs.append(cm_img)
    return np.stack(ret_imgs, 0).astype(np.float32)

def colour_map_op(imgs):
    imgs_shape = tf.shape(imgs)
    shape = [-1] + [imgs_shape[1], imgs_shape[2]] + [3]
    imgs = tf.py_func(_apply_colour_map, [imgs], tf.float32, stateful=False, name="colour_map")
    imgs = tf.reshape(imgs, shape)
    return imgs