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
from nets import nets_factory
from preprocessing import preprocessing_factory

import os

slim = contrib_slim

tf.app.flags.DEFINE_integer(
    'batch_size', 256, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', None, 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

tf.app.flags.DEFINE_bool('use_grayscale', False,
                         'Whether to convert input images to grayscale.')

tf.app.flags.DEFINE_string(
    'final_endpoint', None,
    'Specifies the endpoint to construct the network up to.'
    'By default, None would be the last layer before Logits.') # this argument was added for modbilenet_v1.py
tf.app.flags.DEFINE_bool(
    'verbose_placement', False,
    'Shows detailed information about device placement.')

tf.app.flags.DEFINE_bool(
    'hard_placement', False,
    'Uses hard constraints for device placement on tensorflow sessions.')

tf.app.flags.DEFINE_bool(
    'fixed_memory', False,
    'Allocates the entire memory at once.')

#######################
# Preprocessing Flags #
#######################

tf.app.flags.DEFINE_string(
    'roi', None, 
    'Specifies the coordinates of an ROI for cropping the input images.'
    'Expects four integers in the order of roi_y_min, roi_x_min, roi_height, roi_width, image_height, image_width.')

FLAGS = tf.app.flags.FLAGS
EVAL_DIR = os.path.join(FLAGS.eval_dir, FLAGS.dataset_split_name)

def _parse_roi():
    # parse roi
    if FLAGS.roi is None:
      return FLAGS.roi
    else: 
      roi_array_string = FLAGS.roi.split(',')
      roi_array = []
      for i in roi_array_string:
        roi_array.append(int(i))
      return roi_array

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  
  if not FLAGS.eval_dir:
    raise ValueError('You must supply an eval directory with --eval_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    # tf_global_step = slim.get_or_create_global_step()
    tf_global_step = tf.train.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

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
        num_epochs=1,
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
        use_grayscale=FLAGS.use_grayscale,
        roi=_parse_roi())

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

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


    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.floor(dataset.num_samples / float(FLAGS.batch_size)) - 1

    # print('####################2', FLAGS.batch_size, num_batches, dataset.num_samples)

    # if checkpoint_path flag not set, look for checkpoint in train
    if FLAGS.checkpoint_path is None:
        checkpoint_path = os.path.join(FLAGS.eval_dir, 'train')
    else:
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
          checkpoint_path = FLAGS.checkpoint_path
        

    tf.logging.info('#####Evaluating %s' % checkpoint_path)
    # evaluate for 1000 batches:
    # num_evals = 5
    
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    update_op = tf.group(*update_ops)
    # print('################', update_op)
    with tf.control_dependencies([update_op]):
      total_loss = tf.identity(loss, name='total_loss')
      # summaries.add(tf.summary.scalar('total_loss_1', total_loss))



    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    # session_config = tf.ConfigProto()
    # session_config.gpu_options.allow_growth = True
    session_config = tf.ConfigProto(
        log_device_placement = FLAGS.verbose_placement, 
        allow_soft_placement = not FLAGS.hard_placement)
    if not FLAGS.fixed_memory :
      session_config.gpu_options.allow_growth=True

    # Evaluate every 1 minute:
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=checkpoint_path,
        logdir=EVAL_DIR,
        num_evals=num_batches,
        eval_op=update_ops,
        summary_op=summary_op,
        eval_interval_secs=60,
        session_config=session_config) 
    # How often to run the evaluation
    # slim.evaluation.evaluate_once(
    #     master=FLAGS.master,
    #     checkpoint_path=checkpoint_path,
    #     logdir=FLAGS.eval_dir,
    #     num_evals=num_batches,
    #     eval_op=list(names_to_updates.values()),
    #     variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
