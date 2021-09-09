# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Saves out a GraphDef containing the architecture of the model.

To use it, run something like this, with a model name defined by slim:

bazel build tensorflow_models/research/slim:export_inference_graph
bazel-bin/tensorflow_models/research/slim/export_inference_graph \
--model_name=inception_v3 --output_file=/tmp/inception_v3_inf_graph.pb

If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/tmp/inception_v3_inf_graph.pb \
--checkpoint_path=/tmp/checkpoints/inception_v3.ckpt \
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
--output_node_names=InceptionV3/Predictions/Reshape_1

The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:

bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=/tmp/inception_v3_inf_graph.pb

To run the resulting graph in C++, you can look at the label_image sample code:

bazel build tensorflow/examples/label_image:label_image
bazel-bin/tensorflow/examples/label_image/label_image \
--image=${HOME}/Pictures/flowers.jpg \
--input_layer=input \
--output_layer=InceptionV3/Predictions/Reshape_1 \
--graph=/tmp/frozen_inception_v3.pb \
--labels=/tmp/imagenet_slim_labels.txt \
--input_mean=0 \
--input_std=255

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import slim as contrib_slim
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef

from tensorflow.python.platform import gfile
from datasets import dataset_factory
from datasets import dataset_utils
from nets import nets_factory
from freeze_graph import freeze_graph, export_inference_graph

import os
import argparse

slim = contrib_slim

parser = argparse.ArgumentParser()

parser.add_argument("--project_dir", type=str, default='./project_dir/' , help='Directory where the results are saved to.')

parser.add_argument('--project_name', type=str, default=None, help= 'Must supply a project name examples: flower_classifier, component_classifier')

# parser.add_argument("--dataset_name", type=str, default=None, help='The name of the dataset to load.')

parser.add_argument('--experiment_name', type=str, default=None, help= ' If None the highest experiment number (The number of experiment folders) is selected. ')

parser.add_argument(
    '--model_name',
    type=str,
    default='mobilenet_v1',
    help='The name of the architecture to save.')

parser.add_argument(
    '--is_training', type=bool, default=False, help='Whether to save out a training-focused version of the model.')

parser.add_argument(
    '--image_size', type=int, default=None, help='The image size to use, otherwise use the model default_image_size.')

parser.add_argument(
    '--batch_size', type=int, default=None, help='Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

parser.add_argument('--dataset_name', type=str, default="", help='The name of the dataset to use with the model.')

parser.add_argument(
    '--labels_offset', type=int, default=0,
    help='An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

parser.add_argument(
    '--output_file', type=str, default="", help='Where to save the resulting file to.')

parser.add_argument(
    '--dataset_dir', type=str, default="", help='Directory to save intermediate dataset files to')

parser.add_argument(
    '--quantize', type=bool, default=False, help='whether to use quantized graph or not.')

parser.add_argument(
    '--is_video_model', type=bool, default=False, help='whether to use 5-D inputs for video model.')

parser.add_argument(
    '--num_frames', type=int, default=None,
    help='The number of frames to use. Only used if is_video_model is True.')

parser.add_argument('--write_text_graphdef', type=bool, default=False, help='Whether to write a text version of graphdef.')

parser.add_argument('--use_grayscale', type=bool, default=False, help='Whether to convert input images to grayscale.')

parser.add_argument(
    '--final_endpoint', type=str, default=None, help='Specifies the endpoint to construct the network up to.'
    'By default, None would be the last layer before Logits.') # this argument was added for modbilenet_v1.py

parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--input_graph",
    type=str,
    default="",
    help="TensorFlow \'GraphDef\' file to load.")
parser.add_argument(
    "--input_saver",
    type=str,
    default="",
    help="TensorFlow saver file to load.")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="",
    help="TensorFlow variables file to load.")
parser.add_argument(
    "--checkpoint_version",
    type=int,
    default=2,
    help="Tensorflow variable file format")
parser.add_argument(
    "--output_graph",
    type=str,
    default="",
    help="Output \'GraphDef\' file name.")
parser.add_argument(
    "--input_binary",
    nargs="?",
    const=True,
    type="bool",
    default=True,
    help="Whether the input files are in binary format.")
parser.add_argument(
    "--output_node_names",
    type=str,
    default="MobilenetV1/Predictions/Reshape_1",
    help="The name of the output nodes, comma separated.")
parser.add_argument(
    "--restore_op_name",
    type=str,
    default="save/restore_all",
    help="""\
    The name of the master restore operator. Deprecated, unused by updated \
    loading code.
    """)
parser.add_argument(
    "--filename_tensor_name",
    type=str,
    default="save/Const:0",
    help="""\
    The name of the tensor holding the save path. Deprecated, unused by \
    updated loading code.
    """)
parser.add_argument(
    "--clear_devices",
    nargs="?",
    const=True,
    type="bool",
    default=True,
    help="Whether to remove device specifications.")
parser.add_argument(
    "--initializer_nodes",
    type=str,
    default="",
    help="Comma separated list of initializer nodes to run before freezing.")
parser.add_argument(
    "--variable_names_whitelist",
    type=str,
    default="",
    help="""\
    Comma separated list of variables to convert to constants. If specified, \
    only those variables will be converted to constants.\
    """)
parser.add_argument(
    "--variable_names_blacklist",
    type=str,
    default="",
    help="""\
    Comma separated list of variables to skip converting to constants.\
    """)
parser.add_argument(
    "--input_meta_graph",
    type=str,
    default="",
    help="TensorFlow \'MetaGraphDef\' file to load.")
parser.add_argument(
    "--input_saved_model_dir",
    type=str,
    default="",
    help="Path to the dir with TensorFlow \'SavedModel\' file and variables.")
parser.add_argument(
    "--saved_model_tags",
    type=str,
    default="serve",
    help="""\
    Group of tag(s) of the MetaGraphDef to load, in string format,\
    separated by \',\'. For tag-set contains multiple tags, all tags \
    must be passed in.\
    """)

# FLAGS = tf.app.flags.FLAGS
FLAGS = parser.parse_args()
# _IMAGE_DIR = '/home/docker/ahmed/datasets/blocks_cleaned_photos_tfrecord/blocks_20'

def export_inference_graph(dataset_name, dataset_dir,  model_name, labels_offset, is_training, final_endpoint, image_size, use_grayscale, is_video_model, batch_size, num_frames, quantize, write_text_graphdef, output_file):
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as graph:
        dataset = dataset_factory.get_dataset(dataset_name, 'train', dataset_dir)
        network_fn = nets_factory.get_network_fn(
            model_name,
            num_classes=(dataset.num_classes - labels_offset),
            is_training=is_training,
            final_endpoint=final_endpoint)
        image_size = image_size or network_fn.default_image_size
        num_channels = 1 if use_grayscale else 3
        if is_video_model:
          input_shape = [
              batch_size, num_frames, image_size, image_size,
              num_channels
          ]
        else:
          input_shape = [batch_size, image_size, image_size, num_channels]
        placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                     shape=input_shape)
        network_fn(placeholder)

        if quantize:
          contrib_quantize.create_eval_graph()

        graph_def = graph.as_graph_def()
        if write_text_graphdef:
          tf.io.write_graph(
              graph_def,
              os.path.dirname(output_file),
              os.path.basename(output_file),
              as_text=True)
        else:
          with gfile.GFile(output_file, 'wb') as f:
            f.write(graph_def.SerializeToString())


def main():

    # if not FLAGS.output_file:
    #     raise ValueError('You must supply the path to save to with --output_file')
    if FLAGS.is_video_model and not FLAGS.num_frames:
        raise ValueError(
        'Number of frames must be specified for video models with --num_frames')
    if not FLAGS.checkpoint_path:
      # checkpoint_path = experiment_dir
        checkpoint_path = os.path.join(experiment_dir, 'train')
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        print('#####2', checkpoint_path)
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    if FLAGS.checkpoint_version == 1:
        checkpoint_version = saver_pb2.SaverDef.V1
    elif FLAGS.checkpoint_version == 2:
        checkpoint_version = saver_pb2.SaverDef.V2
    else:
        print("Invalid checkpoint version (must be '1' or '2'): %d" % FLAGS.checkpoint_version)
        return -1

    export_inference_graph(FLAGS.dataset_name, dataset_dir,  FLAGS.model_name, FLAGS.labels_offset, FLAGS.is_training, FLAGS.final_endpoint, FLAGS.image_size, FLAGS.use_grayscale, FLAGS.is_video_model, FLAGS.batch_size, FLAGS.num_frames, FLAGS.quantize, FLAGS.write_text_graphdef, output_file)

    if not os.path.isfile(output_file):
        raise ValueError('graph not found')
    freeze_graph(output_file, FLAGS.input_saver, FLAGS.input_binary,
    checkpoint_path, FLAGS.output_node_names,
    FLAGS.restore_op_name, FLAGS.filename_tensor_name,
    FLAGS.output_graph, FLAGS.clear_devices, FLAGS.initializer_nodes,
    FLAGS.variable_names_whitelist, FLAGS.variable_names_blacklist,
    FLAGS.input_meta_graph, FLAGS.input_saved_model_dir,
    FLAGS.saved_model_tags, checkpoint_version)

if __name__ == '__main__':
  # check required input arguments
  if not FLAGS.project_name:
    raise ValueError('You must supply a dataset name with --project_name')
  if not FLAGS.dataset_name:
    raise ValueError('You must supply a dataset name with --dataset_name')

  project_dir = os.path.join(FLAGS.project_dir, FLAGS.project_name)
  if not FLAGS.experiment_name:
    # list only directories that are names experiment_
      # experiment_dir = create_new_experiment_dir(project_dir)
      experiment_dir = dataset_utils.select_latest_experiment_dir(project_dir)
  else:
      experiment_dir = os.path.join(os.path.join(project_dir, 'experiments'), FLAGS.experiment_name)
      if not os.path.exists(experiment_dir):
          raise ValueError('Experiment directory {} does not exist.'.format(experiment_dir))

  train_dir = os.path.join(experiment_dir, 'train')
  # if not os.path.exists(train_dir):
  #     os.makedirs(train_dir)
  output_file = os.path.join(train_dir, FLAGS.model_name + '_graph.pb')
  # print('#############',output_file)
  # set and check dataset directory
  if FLAGS.dataset_dir:
      dataset_dir = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name)
  else:
      dataset_dir = os.path.join(os.path.join(project_dir, 'datasets'), FLAGS.dataset_name)
  if not os.path.isdir(dataset_dir):
      raise ValueError(f'Can not find tfrecord dataset directory {dataset_dir}')

  main()
