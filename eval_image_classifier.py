# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import slim as contrib_slim

from datasets import dataset_factory
from datasets import dataset_utils
from nets import nets_factory
from preprocessing import preprocessing_factory

import os
import argparse

slim = contrib_slim

p = argparse.ArgumentParser()

p.add_argument("--batch_size", type=int, default=16, help='The number of samples in each batch.')

p.add_argument("--max_num_batches", type=int, default=None, help='Max number of batches to evaluate by default use all.')

p.add_argument("--master", type=str, default='', help='The address of the TensorFlow master to use.')

p.add_argument("--checkpoint_path", type=str, default=None, help='The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

p.add_argument("--project_dir", type=str, default='./project_dir/' , help='Directory where the results are saved to.')

p.add_argument('--project_name', type=str, default=None, help= 'Must supply a project name examples: flower_classifier, component_classifier')

p.add_argument("--num_preprocessing_threads", type=int, default=4, help='The number of threads used to create the batches.')

p.add_argument("--dataset_name", type=str, default=None, help='The name of the dataset to load.')

p.add_argument("--dataset_split_name", type=str, default='validation', help='The name of the train/validation/test split.')

p.add_argument("--dataset_dir", type=str, default=None, help='The directory where the dataset files are stored.')

p.add_argument("--labels_offset", type=int, default=0, help='An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background class for the ImageNet dataset.')

p.add_argument("--model_name", type=str, default='mobilenet_v1', help='The name of the architecture to evaluate.')

p.add_argument("--preprocessing_name", type=str, default='custom_1_preprocessing_pipline', help='The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

p.add_argument("--moving_average_decay", type=float, default=None, help='The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

p.add_argument("--eval_image_size", type=int, default=None, help='Eval image size')

p.add_argument("--quantize", type=bool, default=False, help='whether to use quantized graph or not.')

p.add_argument("--use_grayscale", type=bool, default=False, help='Whether to convert input images to grayscale.')

p.add_argument("--final_endpoint", type=str, default=None, help='Specifies the endpoint to construct the network up to.'
    'By default, None would be the last layer before Logits.') # this argument was added for modbilenet_v1.py

p.add_argument("--verbose_placement", type=bool, default=False, help='Shows detailed information about device placement.')

p.add_argument("--hard_placement", type=bool, default=False, help='Uses hard constraints for device placement on tensorflow sessions.')

p.add_argument("--fixed_memory", type=bool, default=False, help='Allocates the entire memory at once.')

p.add_argument('--experiment_name', type=str, default=None, help= ' If None the highest experiment number (The number of experiment folders) is selected. ')

p.add_argument("--eval_interval_secs", type=int, default=20, help='The frequency with which the model is evaluated')

p.add_argument("--eval_timeout_secs", type=int, default=None, help='The maximum amount of time to wait between checkpoints. If left as None, then the process will wait for double the eval_interval_secs.')

p.add_argument("--experiment_number", type=int, default=0, help='Only needs to be specified if running script with guild')

#######################
# Preprocessing Flags #
#######################

p.add_argument("--add_image_summaries", type=bool, default=True, help='Enable image summaries.')

p.add_argument("--roi", type=str, default=None, help='Specifies the coordinates of an ROI for cropping the input images.'
    ' Expects six integers in the order of roi_y_min, roi_x_min, roi_height, roi_width, image_height, image_width.')

FLAGS = p.parse_args()

def _parse_roi():
    if FLAGS.roi is None:
      return FLAGS.roi
    else:
      roi_array_string = FLAGS.roi.split(',')
      roi_array = []
      for i in roi_array_string:
        roi_array.append(int(i))
      return roi_array

def main():
  # check required input arguments
  if not FLAGS.project_name:
    raise ValueError('You must supply a project name with --project_name')
  if not FLAGS.dataset_name:
    raise ValueError('You must supply a dataset name with --dataset_name')
  # set and check project_dir and experiment_dir.
  project_dir = os.path.join(FLAGS.project_dir, FLAGS.project_name)
  if not FLAGS.experiment_name:
    # list only directories that are names experiment_
    experiment_dir = dataset_utils.select_latest_experiment_dir(project_dir)
  else:
    experiment_dir = os.path.join(os.path.join(project_dir, 'experiments'), FLAGS.experiment_name)
    if not os.path.exists(experiment_dir):
        raise ValueError('Experiment directory {} does not exist.'.format(experiment_dir))

  eval_dir = os.path.join(experiment_dir, FLAGS.dataset_split_name)
  if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

  if FLAGS.dataset_dir:
      dataset_dir = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name)
  else:
      dataset_dir = os.path.join(os.path.join(project_dir, 'datasets'), FLAGS.dataset_name)
  if not os.path.isdir(dataset_dir):
    raise ValueError(f'Can not find tfrecord dataset directory {dataset_dir}')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    # tf_global_step = slim.get_or_create_global_step()
    tf_global_step = tf.train.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False,
        final_endpoint=FLAGS.final_endpoint)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        num_epochs=None,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False,
        use_grayscale=FLAGS.use_grayscale)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size,
        add_image_summaries=FLAGS.add_image_summaries,
        roi=_parse_roi())

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if FLAGS.quantize:
      contrib_quantize.create_eval_graph()

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, dataset.num_classes), logits)

    #############################
    ## Calculation of metrics ##
    #############################
    accuracy, accuracy_op = tf.metrics.accuracy(tf.squeeze(labels), tf.argmax(logits, 1))
    precision, precision_op = tf.metrics.average_precision_at_k(tf.squeeze(labels), logits, 1)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops.append([accuracy_op, precision_op])

    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('accuracy_op', accuracy_op)
    tf.add_to_collection('precision', precision)
    tf.add_to_collection('precision_op', precision_op)

    for class_id in range(dataset.num_classes):
        precision_at_k, precision_at_k_op = tf.metrics.precision_at_k(tf.squeeze(labels), logits, k=1, class_id=class_id)
        recall_at_k, recall_at_k_op = tf.metrics.recall_at_k(tf.squeeze(labels),    logits, k=1, class_id=class_id)
        update_ops.append([precision_at_k_op, recall_at_k_op])

        tf.add_to_collection(f'precision_at_{class_id}', precision_at_k)
        tf.add_to_collection(f'precision_at_{class_id}_op', precision_at_k_op)
        tf.add_to_collection(f'recall_at_{class_id}', recall_at_k)
        tf.add_to_collection(f'recall_at_{class_id}_op', recall_at_k_op)

    #############################
    ## Add summaries ##
    #############################
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
      summaries.add(tf.summary.scalar('Losses/clone_loss', loss))

    summaries.add(tf.summary.scalar('Metrics/accuracy', accuracy))
    summaries.add(tf.summary.scalar('op/accuracy_op', accuracy_op))

    summaries.add(tf.summary.scalar('Metrics/average_precision', precision))
    summaries.add(tf.summary.scalar('op/average_precision_op', precision_op))

    for class_id in range(dataset.num_classes):
        precision_at_k = tf.get_collection(f'precision_at_{class_id}')
        precision_at_k_op = tf.get_collection(f'precision_at_{class_id}_op')
        recall_at_k = tf.get_collection(f'recall_at_{class_id}')
        recall_at_k_op = tf.get_collection(f'recall_at_{class_id}_op')

        precision_at_k = tf.reshape(precision_at_k, [])
        precision_at_k_op = tf.reshape(precision_at_k_op, [])
        recall_at_k = tf.reshape(recall_at_k, [])
        recall_at_k_op = tf.reshape(recall_at_k_op, [])

        summaries.add(tf.summary.scalar(f'Metrics/class_{class_id}_precision', precision_at_k))
        summaries.add(tf.summary.scalar(f'op/class_{class_id}_precision_op', precision_at_k_op))
        summaries.add(tf.summary.scalar(f'Metrics/class_{class_id}_recall', recall_at_k))
        summaries.add(tf.summary.scalar(f'op/class_{class_id}_recall_op', recall_at_k_op))

    # set batch size if none to
    # number_of_samples_in_dataset / batch_size
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    # if checkpoint_path flag is none, look for checkpoint in experiment train directory
    if FLAGS.checkpoint_path is None:
        checkpoint_path = os.path.join(experiment_dir, 'train')
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating checkpoint: %s' % checkpoint_path)
    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      total_loss = tf.identity(loss, name='total_loss')

    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    # configure session
    session_config = tf.ConfigProto(
        log_device_placement = FLAGS.verbose_placement,
        allow_soft_placement = not FLAGS.hard_placement)
    if not FLAGS.fixed_memory :
      session_config.gpu_options.allow_growth=True
    # set evaluation interval
    if not FLAGS.eval_timeout_secs:
        eval_timeout_secs = FLAGS.eval_interval_secs * 2
    else:
        eval_timeout_secs = FLAGS.eval_timeout_secs

    # Evaluate every 1 minute:
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=checkpoint_path,
        logdir=eval_dir,
        num_evals=num_batches,
        eval_op=update_ops,
        summary_op=summary_op,
        eval_interval_secs=FLAGS.eval_interval_secs,
        timeout=eval_timeout_secs,
        session_config=session_config)

if __name__ == '__main__':
  main()
