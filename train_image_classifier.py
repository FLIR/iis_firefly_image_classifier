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
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import slim as contrib_slim
from tensorflow.python.training import saver as tf_saver

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

import os
import datetime
import signal
import time

slim = contrib_slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy. Note For '
                            'historical reasons loss from all clones averaged '
                            'out and learning rate decay happen per clone '
                            'epochs')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 20,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 20,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

tf.app.flags.DEFINE_bool(
    'verbose_placement', False,
    'Shows detailed information about device placement.')

tf.app.flags.DEFINE_bool(
    'hard_placement', False,
    'Uses hard constraints for device placement on tensorflow sessions.')

tf.app.flags.DEFINE_bool(
    'fixed_memory', False,
    'Allocates the entire memory at once.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'adam',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1e-08, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

tf.app.flags.DEFINE_integer(
    'quantize_delay', -1,
    'Number of steps to start quantized training. Set to -1 would disable '
    'quantized training.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v1', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'custom_1_preprocessing_pipline', 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 64, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 224, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', 50000,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_bool('use_grayscale', False,
                         'Whether to convert input images to grayscale.')

tf.app.flags.DEFINE_bool('balance_classes', False,
                         'apply class weight to loss function .')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_bool(
    'feature_extraction', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', './checkpoints/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'MobilenetV1/Logits',
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.'
    'By default, only the Logits layer is excluded')

tf.app.flags.DEFINE_string(
    'trainable_scopes', 'MobilenetV1/Logits',
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, only the Logits layer is trained. None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')

tf.app.flags.DEFINE_string(
    'final_endpoint', None,
    'Specifies the endpoint to construct the network up to.'
    'By default, None would be the last layer before Logits.') # this argument was added for modbilenet_v1.py

#######################
# Experiment Details #
#######################

tf.app.flags.DEFINE_string(
    'experiment_tag', '', 'Internal tag for experiment')

tf.app.flags.DEFINE_string(
    'experiment_file', None, 'File to output experiment metadata')

#######################
# Preprocessing Flags #
#######################


tf.app.flags.DEFINE_bool(
    'add_image_summaries', True,
    'Enable image summaries.')

tf.app.flags.DEFINE_bool(
    'apply_image_augmentation', True,
    'Enable random image augmentation during preprocessing for training.')

tf.app.flags.DEFINE_bool(
    'random_image_crop', True,
    'Enable random cropping of images. Only Enabled if apply_image_augmentation flag is also enabled')

tf.app.flags.DEFINE_float(
    'min_object_covered', 0.8,
    'The remaining cropped image must contain at least this fraction of the whole image. Only Enabled if apply_image_augmentation flag is also enabled')

tf.app.flags.DEFINE_bool(
    'random_image_rotation', True,
    'Enable random image rotation counter-clockwise by 90, 180, 270, or 360 degrees. Only Enabled if apply_image_augmentation flag is also enabled')

tf.app.flags.DEFINE_bool(
    'random_image_flip', False,
    'Enable random image flip (horizontally). Only Enabled if apply_image_augmentation flag is also enabled')

tf.app.flags.DEFINE_string(
    'roi', None,
    'Specifies the coordinates of an ROI for cropping the input images.'
    'Expects four integers in the order of roi_y_min, roi_x_min, roi_height, roi_width, image_height, image_width. Only applicable to mobilenet_preprocessing pipeline ')


FLAGS = tf.app.flags.FLAGS

if FLAGS.train_dir:
    TRAIN_DIR = os.path.join(FLAGS.train_dir, FLAGS.dataset_split_name)
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
else:
    raise ValueError('You must supply train directory with --train_dir.')



def _parse_roi():
    # parse roi
    # roi="650,950,224,224"
    if FLAGS.roi is None:
      return FLAGS.roi
    else:
      # print("##################################### roi", FLAGS.roi)
      roi_array_string = FLAGS.roi.split(',')
      roi_array = []
      for i in roi_array_string:
        roi_array.append(int(i))
      # print("##################################### roi parsed", roi_array)
      return roi_array


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  # Note: when num_clones is > 1, this will actually have each clone to go
  # over each epoch FLAGS.num_epochs_per_decay times. This is different
  # behavior from sync replicas and is expected to produce different results.
  decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay /
                    FLAGS.batch_size)

  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)
  return optimizer


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  # if tf.train.latest_checkpoint(TRAIN_DIR):
  #   tf.logging.info(
  #       'Ignoring --checkpoint_path because a checkpoint already exists in %s'
  #       % TRAIN_DIR)
  #   return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        break
    else:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path
  if FLAGS.feature_extraction:
  	tf.logging.info('Feature-extraction from %s' % checkpoint_path)
  else:
  	tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
     checkpoint_path,
     variables_to_restore,
     ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  # print('######## All avaialable Trainable variables from name scope \n', tf.trainable_variables())
  # fine-tune setting will add all batchnorm layers to variables to train, if no batchnorm layer is included in the trainable_scope flag.
  if not FLAGS.feature_extraction:
    if 'BatchNorm' not in FLAGS.trainable_scopes:
      scopes.append('BatchNorm')

  print('##############', scopes)
  for scope in scopes:
  	variables = []
  	for variable in tf.trainable_variables():
  		if scope in variable.name:
  			variables.append(variable)
  	variables_to_train.extend(variables)
  	# variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

  	print('######## Trainable variables from name scope', scope, '\n', variables)
  # print('######## List of all Trainable Variables ########### \n', list(set(variables_to_train)))
  return list(set(variables_to_train))


def main(_):
  if not os.path.isdir(FLAGS.dataset_dir):
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  DATASET_DIR = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name+'_tfrecord')
  if not os.path.isdir(DATASET_DIR):
    raise ValueError(f'Can not find tfrecord dataset directory {DATASET_DIR}')
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, DATASET_DIR)

    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=not FLAGS.feature_extraction,
        final_endpoint=FLAGS.final_endpoint)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=FLAGS.apply_image_augmentation,
        use_grayscale=FLAGS.use_grayscale)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=FLAGS.num_readers,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
      [image, label] = provider.get(['image', 'label'])
      label -= FLAGS.labels_offset

      train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      image = image_preprocessing_fn(image,
        train_image_size, train_image_size,
        add_image_summaries=FLAGS.add_image_summaries,
        crop_image=FLAGS.random_image_crop,
        min_object_covered=FLAGS.min_object_covered,
        rotate_image=FLAGS.random_image_rotation,
        random_flip=FLAGS.random_image_flip,
        roi=FLAGS.roi)

      images, labels = tf.train.batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
      labels = slim.one_hot_encoding(
          labels, dataset.num_classes - FLAGS.labels_offset)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)

    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)

      #############################
      # Specify the loss function #
      #############################
      if 'AuxLogits' in end_points:
        slim.losses.softmax_cross_entropy(
            end_points['AuxLogits'], labels,
            label_smoothing=FLAGS.label_smoothing, weights=0.4,
            scope='aux_loss')


      if FLAGS.balance_classes:
          for label_name in bdataset.num_samples_per_class:
              class_name = dataset.labels_to_names[label_name]

          class_weights = tf.constant([41.25, 0.51])
          sample_weights = tf.reduce_sum(tf.multiply(labels, class_weights), 1)
          slim.losses.softmax_cross_entropy(
                    logits, labels, label_smoothing=FLAGS.label_smoothing, weights=sample_weights)
      else:
          slim.losses.softmax_cross_entropy(
                    logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
      #############################
      ## Calculation of metrics ##
      #############################

      # print('###########1',logits, labels)
      accuracy, accuracy_op = tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(logits, 1))
      precision, precision_op = tf.metrics.average_precision_at_k(tf.argmax(labels, 1), logits, 1)

      with tf.device('/device:CPU:0'):
        for class_id in range(dataset.num_classes):
          precision_at_k, precision_at_k_op = tf.metrics.precision_at_k(tf.argmax(labels, 1), logits, k=1, class_id=class_id)
          recall_at_k, recall_at_k_op = tf.metrics.recall_at_k(tf.argmax(labels, 1), logits, k=1, class_id=class_id)
          tf.add_to_collection(f'precision_at_{class_id}', precision_at_k)
          tf.add_to_collection(f'precision_at_{class_id}_op', precision_at_k_op)
          tf.add_to_collection(f'recall_at_{class_id}', recall_at_k)
          tf.add_to_collection(f'recall_at_{class_id}_op', recall_at_k_op)

      # print('###########',precision, recall)

      tf.add_to_collection('accuracy', accuracy)
      tf.add_to_collection('accuracy_op', accuracy_op)
      tf.add_to_collection('precision', precision)
      tf.add_to_collection('precision_op', precision_op)


      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    #########################################################
    ## Calculation of metrics for all clones ##
    #########################################################

    # Metrics for all clones.
    accuracy = tf.get_collection('accuracy')
    accuracy_op = tf.get_collection('accuracy_op')
    precision = tf.get_collection('precision')
    precision_op = tf.get_collection('precision_op')
    # accuracy_op = tf.reshape(accuracy_op, [])


    # Stack and take the mean.
    accuracy = tf.reduce_mean(tf.stack(accuracy, axis=0))
    accuracy_op = tf.reduce_mean(tf.stack(accuracy_op, axis=0))
    precision = tf.reduce_mean(tf.stack(precision, axis=0))
    precision_op = tf.reduce_mean(tf.stack(precision_op, axis=0))

    # Add metric summaries.
    summaries.add(tf.summary.scalar('Metrics/accuracy', accuracy))
    summaries.add(tf.summary.scalar('op/accuracy_op', accuracy_op))
    summaries.add(tf.summary.scalar('Metrics/average_precision', precision))
    summaries.add(tf.summary.scalar('op/average_precision_op', precision_op))

    # Add precision/recall at each class to summary
    for class_id in range(dataset.num_classes):
      precision_at_k = tf.get_collection(f'precision_at_{class_id}')
      precision_at_k_op = tf.get_collection(f'precision_at_{class_id}_op')
      recall_at_k = tf.get_collection(f'recall_at_{class_id}')
      recall_at_k_op = tf.get_collection(f'recall_at_{class_id}_op')

      precision_at_k = tf.reduce_mean(tf.stack(precision_at_k, axis=0))
      precision_at_k_op = tf.reduce_mean(tf.stack(precision_at_k_op, axis=0))
      recall_at_k = tf.reduce_mean(tf.stack(recall_at_k, axis=0))
      recall_at_k_op = tf.reduce_mean(tf.stack(recall_at_k_op, axis=0))

      summaries.add(tf.summary.scalar(f'Metrics/class_{class_id}_precision', precision_at_k))
      summaries.add(tf.summary.scalar(f'op/class_{class_id}_precision_op', precision_at_k_op))
      summaries.add(tf.summary.scalar(f'Metrics/class_{class_id}_recall', recall_at_k))
      summaries.add(tf.summary.scalar(f'op/class_{class_id}_recall_op', recall_at_k_op))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    if FLAGS.quantize_delay >= 0:
      contrib_quantize.create_training_graph(quant_delay=FLAGS.quantize_delay)

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar('Losses/learning_rate', learning_rate))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.worker_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    # print('############# operations', update_op)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    session_config = tf.ConfigProto(
        log_device_placement = FLAGS.verbose_placement,
        allow_soft_placement = not FLAGS.hard_placement)
    if not FLAGS.fixed_memory :
      session_config.gpu_options.allow_growth=True

    ###########################
    # Kicks off the training. #
    ###########################
    def train_step_fn(sess, train_op, global_step, train_step_kwargs):
      """Function that takes a gradient step and specifies whether to stop.
      Args:
        sess: The current session.
        train_op: An `Operation` that evaluates the gradients and returns the total
          loss.
        global_step: A `Tensor` representing the global training step.
        train_step_kwargs: A dictionary of keyword arguments.
      Returns:
        The total loss and a boolean indicating whether or not to stop training.
      Raises:
        ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
      """
      start_time = time.time()

      trace_run_options = None
      run_metadata = None
      if 'should_trace' in train_step_kwargs:
        if 'logdir' not in train_step_kwargs:
          raise ValueError('logdir must be present in train_step_kwargs when '
                           'should_trace is present')
        if sess.run(train_step_kwargs['should_trace']):
          trace_run_options = config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE)
          run_metadata = config_pb2.RunMetadata()

      total_loss, np_global_step = sess.run([train_op, global_step],
                                            options=trace_run_options,
                                            run_metadata=run_metadata)

      time_elapsed = time.time() - start_time

      if run_metadata is not None:
        tl = timeline.Timeline(run_metadata.step_stats)
        trace = tl.generate_chrome_trace_format()
        trace_filename = os.path.join(train_step_kwargs['logdir'],
                                      'tf_trace-%d.json' % np_global_step)
        logging.info('Writing trace to %s', trace_filename)
        file_io.write_string_to_file(trace_filename, trace)
        if 'summary_writer' in train_step_kwargs:
          train_step_kwargs['summary_writer'].add_run_metadata(
              run_metadata, 'run_metadata-%d' % np_global_step)

      if 'should_log' in train_step_kwargs:
        if sess.run(train_step_kwargs['should_log']):
          logging.info('global step %d: loss = %.4f (%.3f sec/step)', np_global_step, total_loss, time_elapsed)
          # print(logits, labels)

      if 'should_stop' in train_step_kwargs:
        should_stop = sess.run(train_step_kwargs['should_stop'])
      else:
        should_stop = False

      return total_loss, should_stop or train_step_fn.should_stop


    train_step_fn.should_stop = False

    def exit_gracefully(signum, frame) :
      interrupted = datetime.datetime.utcnow()
      if not FLAGS.experiment_file is None :
        print('Interrupted on (UTC): ', interrupted, sep='', file=experiment_file)
        experiment_file.flush()
      train_step_fn.should_stop = True
      print('Interrupted on (UTC): ', interrupted, sep='')

    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    start = datetime.datetime.utcnow()
    print('Started on (UTC): ', start, sep='')
    if not FLAGS.experiment_file is None :

      experiment_file = open(os.path.join(TRAIN_DIR, FLAGS.experiment_file), 'w')
      print('Experiment metadata file:', file=experiment_file)
      print(FLAGS.experiment_file, file=experiment_file)
      print('========================', file=experiment_file)
      print('All command-line flags:', file=experiment_file)
      print(FLAGS.experiment_file, file=experiment_file)
      for flag_key in sorted(FLAGS.__flags.keys()) :
        print(flag_key, ' : ', FLAGS.__flags[flag_key].value, sep='', file=experiment_file)
      print('========================', file=experiment_file)
      print('Started on (UTC): ', start, sep='', file=experiment_file)
      experiment_file.flush()

    slim.learning.train(
        train_tensor,
        train_step_fn=train_step_fn,
        logdir=TRAIN_DIR,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None,
        session_config=session_config)

    finish = datetime.datetime.utcnow()
    if not FLAGS.experiment_file is None :
      print('Finished on (UTC): ', finish, sep='', file=experiment_file)
      print('Elapsed: ', finish-start, sep='', file=experiment_file)
      experiment_file.flush()

if __name__ == '__main__':
  tf.app.run()
