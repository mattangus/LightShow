import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from datasets.dataset import build_dataset
import helpers 

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", type=str, required=True)

args = parser.parse_args()

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

dataset = build_dataset([data_files], 1, 10, False)
iterator = dataset.make_one_shot_iterator()
next_item = iterator.get_next()

spectrogram = next_item[helpers.DATA_KEY]
mean_val = tf.reduce_mean(spectrogram)
length = tf.shape(spectrogram)[1]

cur_mean = 0
all_lens = []
i = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    while True:
        try:
            x,l = sess.run([mean_val, length])
        except tf.errors.OutOfRangeError:
            break
        i+=1
        cur_mean += (x - cur_mean)/i
        all_lens.append(l)

        print("{}: {}          ".format(i, cur_mean), end="\r")
    
    print("{}: {}          ".format(i, cur_mean))

_, bins, _ = plt.hist(all_lens)
print(bins)
plt.show()