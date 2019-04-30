import tensorflow as tf
import numpy as np
import csv
import argparse
import sys
import tensorflow as tf
import os
sys.path.append(".")

import helpers
from datasets import dataset_handler as dh

name_to_handler = {
    "tess": dh.TessHandler,
    "meld": dh.MeldHandler,
    "savee": dh.SaveeHandler,
    "crema": dh.CremaHandler,
}

parser = argparse.ArgumentParser()
parser.add_argument("--base_folder", type=str, required=True, help="base folder where audio files are stored")
parser.add_argument("--name", type=str, required=True, help="name of dataset")
parser.add_argument("--split", type=str, required=True, help="'train', 'val', or 'test'")
parser.add_argument("--output_folder", type=str, required=True, help="folder to write tfrecord to")

args = parser.parse_args()

byte_limit = 1500*1024*1024

def _bytes_feature(values):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[values]))

def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def create_tf_example(example_data):
    data = example_data[helpers.DATA_KEY]
    emo_id = example_data[helpers.EMO_KEY]
    sent_id = example_data[helpers.SENT_KEY]
    emo_weight = example_data[helpers.EMO_WEIGHT_KEY]
    sent_weight = example_data[helpers.SENT_WEIGHT_KEY]
    
    feature_dict = {
        helpers.DATA_KEY: _bytes_feature(data),
        helpers.EMO_KEY: _int64_feature(emo_id),
        helpers.SENT_KEY: _int64_feature(sent_id),
        helpers.EMO_WEIGHT_KEY: _float_feature(emo_weight),
        helpers.SENT_WEIGHT_KEY: _float_feature(emo_weight),
    }
    
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return example

def _make_new_writer(output_path, shard_num):
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(output_path + "-" + str(shard_num), options=options)
    return writer

def _create_tf_record(handler, output_path):
    shard_num = 0
    num_bytes = 0
    writer = _make_new_writer(output_path, shard_num)
    num_elem = len(handler)
    for idx, item in enumerate(handler):
        if idx % 100 == 0:
            print("On image {} of {}".format(idx, num_elem))
        tf_example = create_tf_example(item)
        str_out = tf_example.SerializeToString()
        num_bytes += len(str_out)
        if num_bytes > byte_limit:
            writer.close()
            shard_num += 1
            num_bytes = 0
            writer = _make_new_writer(output_path, shard_num)
        writer.write(str_out)
    writer.close()
    print("Finished writing!")

if __name__ == "__main__":        
    assert args.name in name_to_handler, "name must be one of: " + str(list(name_to_handler.keys()))

    handler = name_to_handler[args.name](args.base_folder, args.split)
    output_path = os.path.join(args.output_folder, args.name + "_" + args.split + ".tfrecord")

    _create_tf_record(handler, output_path)