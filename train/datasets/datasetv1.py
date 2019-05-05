import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops

_classes = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

_split_mapping = {"train": None}

num_classes = len(_classes)

mean_value = 0.14715156193625425

def _make_spect(file_name):
    audio_binary = tf.read_file(file_name)
    waveform = audio_ops.decode_wav(audio_binary, desired_channels=1)

    spectrogram = audio_ops.audio_spectrogram(
        waveform.audio,
        window_size=1024,
        stride=64)

    # Tensorflow spectrogram has time along y axis and frequencies along x axis
    # flip them
    spectrogram = tf.image.flip_left_right(spectrogram)
    spectrogram = tf.transpose(spectrogram, [0, 2, 1])
    spectrogram = tf.expand_dims(spectrogram, -1) #add color channel
    spectrogram = tf.image.resize_bilinear(spectrogram,
        (spectrogram.shape[1], spectrogram.shape[1]))
    spectrogram = tf.squeeze(spectrogram, 0)
    spectrogram = spectrogram - mean_value

    one_hot = []

    for c in _classes:
        match = tf.strings.regex_full_match(file_name, ".*" + c + ".*", name="find_" + c)
        one_hot.append(match)
    
    one_hot = tf.cast(tf.stack(one_hot), tf.int32)

    return {"spectrogram": spectrogram, "label": one_hot}

def build_dataset(split="train", batch=None, epochs=None):

    assert split in _split_mapping, split + " is not a valid split. Options are: " + list(_split_mapping.keys())

    split_name = _split_mapping[split]
    
    if split_name is not None:
        pattern = os.path.join("data/Emotion Dataset/", split_name, "*.wav")
    else:
        pattern = "data/Emotion Dataset/*.wav"

    dataset = tf.data.Dataset.list_files(pattern, shuffle=True)
    dataset = dataset.map(_make_spect, 4)

    dataset = dataset.repeat(epochs)
    if batch is not None and batch >= 1:
        dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)

    return dataset
