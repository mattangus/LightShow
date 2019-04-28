import tensorflow as tf
import argparse
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.freeze_graph import freeze_graph
import matplotlib.cm as cm
import numpy as np
import os

from model import build_model
from datasets.dataset import num_classes

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, required=True, help="Where to save checkpoints and summaries")
parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint to fine tune from")
parser.add_argument("--quantize", action="store_true", default=False, help="toggle if the model is quantized")

parser.add_argument("--frozen_name", type=str, default="frozen.pb", help="checkpoint to fine tune from")

args = parser.parse_args()

inputs = tf.placeholder(tf.float32, [None, 128, 150, 1], name="inputs")
# inputs = tf.placeholder(tf.float32, [None, 128, None, 1], name="inputs")

#make the model
print("Building Model")
emo_logits, _ = build_model(inputs, 1.0, num_classes, False)

g = tf.get_default_graph()
if args.quantize:
    print("Quantizing Graph")
    tf.contrib.quantize.create_eval_graph(input_graph=g)

# Save the checkpoint and eval graph proto to disk for freezing
# and providing to TFLite.
os.makedirs(args.log_dir, exist_ok=True)

graph_def_file = os.path.join(args.log_dir, "graph_def.pb")
frozen_graph_file = os.path.join(args.log_dir, args.frozen_name)

print("Writing Graph Def")
with open(graph_def_file, "w") as f:
    f.write(str(g.as_graph_def()))

input_names = "inputs"
output_names = emo_logits.name.replace(":0","")

#freeze graph
print("Freezing Graph")
freeze_graph(input_graph=graph_def_file,
            input_saver="",
            input_binary=False,
            input_checkpoint=args.checkpoint,
            output_node_names=output_names,
            restore_op_name="save/restore_all",
            filename_tensor_name="",
            output_graph=frozen_graph_file,
            clear_devices=True,
            initializer_nodes="",
            input_saved_model_dir="")

#tf lite optimizer
input_arrays = ["inputs"]
output_arrays = [output_names]

print("Converting to tflite")
converter = tf.lite.TFLiteConverter.from_frozen_graph(
  frozen_graph_file, input_arrays, output_arrays)
tflite_model = converter.convert()
tflite_output = os.path.join(args.log_dir, "frozen.tflite")
with open(tflite_output, "wb") as f:
    f.write(tflite_model)