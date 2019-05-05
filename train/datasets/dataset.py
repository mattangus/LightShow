import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops
import random
import librosa
import numpy as np
import functools

import helpers
from datasets import dataset_handler as dh

mean_val_dict = {
    "meld": 0.22431603955600235, #MELD
    "tess": 0.396262021789898, #TESS
    "crema": 0.7988494136567209, #CREMA
    "savee": 5.596532604346669, #SAVEE
}

mean_val_dict = {
    "meld": 0.0, #MELD
    "tess": 0.0, #TESS
    "crema": 0.0, #CREMA
    "savee": 0.0, #SAVEE
}

num_classes = helpers.num_emotions + helpers.num_sentiments

_DATASET_SHUFFLE_SEED = random.randint(-(2**32-1),2**32-1)#7

def _spect_fn(waveform, sample_rate):
    spect = librosa.feature.melspectrogram(y=waveform, sr=sample_rate).astype(np.float32)
    return spect

def _spect_op(waveform_obj):
    ret = tf.py_func(_spect_fn, [waveform_obj.audio[:,0], waveform_obj.sample_rate], tf.float32, stateful=False, name="spectrogram")
    ret = tf.reshape(ret, [128, -1])
    return ret

def _create_tf_example_decoder():

    def decode_example(ex_proto, name):
        keys_to_features = {
            helpers.DATA_KEY: tf.FixedLenFeature((), tf.string),
            helpers.EMO_KEY: tf.FixedLenFeature((), tf.int64),
            helpers.SENT_KEY: tf.FixedLenFeature((), tf.int64),
            helpers.EMO_WEIGHT_KEY: tf.FixedLenFeature((), tf.float32),
            helpers.SENT_WEIGHT_KEY: tf.FixedLenFeature((), tf.float32)
        }

        decoded = tf.parse_single_example(ex_proto, keys_to_features)

        wav_binary = decoded[helpers.DATA_KEY]
        waveform_obj = audio_ops.decode_wav(wav_binary, desired_channels=1)
        spectrogram = _spect_op(waveform_obj)
        # spectrogram = audio_ops.audio_spectrogram(
        #     waveform.audio,
        #     window_size=1024,
        #     stride=32)
        # mfcc = audio_ops.mfcc(spectrogram, waveform.sample_rate, dct_coefficient_count=26)
        # spectrogram = mfcc
        
        # Tensorflow spectrogram has time along y axis and frequencies along x axis
        # spectrogram = tf.image.flip_left_right(spectrogram)
        # spectrogram = tf.transpose(spectrogram, [0, 2, 1])
        spectrogram = tf.expand_dims(spectrogram, -1) #add color channel
        # spectrogram = tf.image.resize_bilinear(spectrogram,
        #     (spectrogram.shape[1], spectrogram.shape[1]))
        # spectrogram = tf.squeeze(spectrogram, 0)
        mean_value = -1
        for n in mean_val_dict:
            if n in name:
                mean_value = mean_val_dict[n]
                break
        if mean_value == -1:
            raise RuntimeError("couldn't find mean value for " + name)
        spectrogram = spectrogram - mean_value

        items_to_handlers = decoded
        items_to_handlers[helpers.DATA_KEY] = spectrogram

        return items_to_handlers

    return decode_example

def _make_dataset(record_file, num_readers, decoder):
    dataset = tf.data.TFRecordDataset(record_file, "GZIP", num_readers)
    dataset = dataset.map(functools.partial(decoder, name=record_file), num_parallel_calls=num_readers*3)

    return dataset

def build_dataset(record_files, num_epoch, num_readers, shuffle, shuffle_queue=75):
    with tf.variable_scope("Dataset"):
        decoder = _create_tf_example_decoder()

        if len(record_files) == 1:
            dataset = _make_dataset(record_files[0], num_readers, decoder)
        else:
            datasets = list(map(lambda x: _make_dataset(x, num_readers, decoder), record_files))
            #TODO: fix this for annoying warning
            datasets = [ds.repeat() for ds in datasets]
            dataset = tf.data.experimental.sample_from_datasets(datasets)

        epochs = num_epoch if num_epoch > 0 else None

        if shuffle:
            sar_fn = tf.data.experimental.shuffle_and_repeat
            sar = sar_fn(shuffle_queue,
                            epochs, seed=_DATASET_SHUFFLE_SEED)
            dataset = dataset.apply(sar)
            print("shuffle seed:", _DATASET_SHUFFLE_SEED)
        else:
            dataset = dataset.repeat(epochs)
        
        dataset = dataset.prefetch(10)

        dataset = dataset.apply(tf.data.experimental.ignore_errors())

    return dataset
