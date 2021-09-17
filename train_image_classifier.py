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
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef

from datasets import dataset_factory
from datasets import convert_dataset
from datasets import dataset_utils
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from freeze_graph import freeze_graph, export_inference_graph

import os
import datetime
import signal
import time
import argparse
import urllib.request
import tarfile

slim = contrib_slim

p = argparse.ArgumentParser()
p.add_argument(
    '--master', type=str, default='', help='The address of the TensorFlow master to use.')

p.add_argument('--project_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'), help='Directory where checkpoints and event logs are written to.')

p.add_argument('--project_name', type=str, default='classifier', help= 'Must supply project name examples: flower_classifier, component_classifier')

p.add_argument('--num_clones', type=int, default=1, help='Number of model clones to deploy. Note For '
                            'historical reasons loss from all clones averaged '
                            'out and learning rate decay happen per clone '
                            'epochs')

p.add_argument('--clone_on_cpu', type=bool, default=False, help='Use CPUs to deploy clones.')

p.add_argument('--worker_replicas', type=int, default=1, help='Number of worker replicas.')

p.add_argument('--num_ps_tasks', type=int, default=0, help='The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

p.add_argument('--num_readers', type=int, default=4, help='The number of parallel readers that read data from the dataset.')

p.add_argument('--num_preprocessing_threads', type=int, default=4, help='The number of threads used to create the batches.')

p.add_argument('--log_every_n_steps', type=int, default=10, help='The frequency with which logs are print.')

p.add_argument('--save_summaries_secs', type=int, default=20, help='The frequency with which summaries are saved, in seconds.')

p.add_argument('--save_interval_secs', type=int, default=20, help='The frequency with which the model is saved, in seconds.')

p.add_argument('--task', type=int, default=0, help='Task id of the replica running the training.')

p.add_argument('--verbose_placement', type=bool, default=False, help='Shows detailed information about device placement.')

p.add_argument('--hard_placement', type=bool, default=False, help='Uses hard constraints for device placement on tensorflow sessions.')

p.add_argument('--fixed_memory', type=bool, default=False, help='Allocates the entire memory at once.')

######################
# Optimization Flags #
######################

p.add_argument('--weight_decay', type=float, default=0.00004, help='The weight decay on the model weights.')

p.add_argument('--optimizer', type=str, default='adam', help='The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

p.add_argument('--adadelta_rho', type=float, default=0.95, help='The decay rate for adadelta.')

p.add_argument('--adagrad_initial_accumulator_value', type=float, default=0.1, help='Starting value for the AdaGrad accumulators.')

p.add_argument('--adam_beta1', type=float, default=0.9, help='The exponential decay rate for the 1st moment estimates.')

p.add_argument('--adam_beta2', type=float, default=0.999, help='The exponential decay rate for the 2nd moment estimates.')

p.add_argument('--opt_epsilon', type=float, default=1e-08, help='Epsilon term for the optimizer.')

p.add_argument('--ftrl_learning_rate_power', type=float, default=-0.5, help='The learning rate power.')

p.add_argument('--ftrl_initial_accumulator_value', type=float, default=0.1, help='Starting value for the FTRL accumulators.')

p.add_argument('--ftrl_l1', type=float, default=0.0, help='The FTRL l1 regularization strength.')

p.add_argument('--ftrl_l2', type=float, default=0.0, help='The FTRL l2 regularization strength.')

p.add_argument('--momentum', type=float, default=0.9, help='The momentum for the MomentumOptimizer and RMSPropOptimizer.')

p.add_argument('--rmsprop_momentum', type=float, default=0.9, help='Momentum.')

p.add_argument('--rmsprop_decay', type=float, default=0.9, help='Decay term for RMSProp.')

p.add_argument('--quantize_delay', type=int, default=-1, help='Number of steps to start quantized training. Set to -1 would disable '
    'quantized training.')

#######################
# Learning Rate Flags #
#######################

p.add_argument('--learning_rate_decay_type', type=str, default='exponential', help='Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

p.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')

p.add_argument('--end_learning_rate', type=float, default=0.0001, help='The minimal end learning rate used by a polynomial decay learning rate.')

p.add_argument('--label_smoothing', type=float, default=0.0, help='The amount of label smoothing.')

p.add_argument('--learning_rate_decay_factor', type=float, default=0.94, help='Learning rate decay factor.')

p.add_argument('--num_epochs_per_decay', type=float, default=2.0, help='Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')

p.add_argument('--sync_replicas', type=bool, default=False, help='Whether or not to synchronize the replicas during training.')

p.add_argument('--replicas_to_aggregate', type=int, default=1, help='The Number of gradients to collect before updating params.')

p.add_argument('--moving_average_decay', type=float, default=None, help='The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

p.add_argument('--image_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'), help='The directory where the input images are saved.')

p.add_argument('--output_graph', type=str, default=os.environ.get('SM_MODEL_DIR'), help='local path where the training job writes the model artificats to.')

p.add_argument('--dataset_name', type=str, default='images', help='The name of the dataset to load.')

p.add_argument('--dataset_split_name', type=str, default='train', help='The name of the train/test split.')

p.add_argument('--dataset_dir', type=str, default="", help='The directory where the dataset files are stored.')

p.add_argument('--train_percentage', type=int, default=80, help='What percentage of images to use as a train set.')

p.add_argument('--validation_percentage', type=int, default=10, help='What percentage of images to use as a validation set.')

p.add_argument('--test_percentage', type=int, default=10, help='What percentage of images to use as a test set.')

p.add_argument('--labels_offset', type=int, default=0, help='An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

p.add_argument('--model_name', type=str, default='mobilenet_v1', help='The name of the architecture to train.')

p.add_argument('--preprocessing_name', type=str, default='custom_1_preprocessing_pipline', help='The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

p.add_argument('--batch_size', type=int, default=16, help='The number of samples in each batch.')

p.add_argument('--train_image_size', type=int, default=224, help='Train image size')

p.add_argument('--max_number_of_steps', type=int, default=80000, help='The maximum number of training steps.')

p.add_argument('--use_grayscale', type=bool, default=False, help='Whether to convert input images to grayscale.')

p.add_argument('--imbalance_correction', type=bool, default=False, help='apply class weight to loss function .')

#####################
# Fine-Tuning Flags #
#####################

p.add_argument('--feature_extraction', type=bool, default=False, help='Whether or not to synchronize the replicas during training.')

p.add_argument('--checkpoint_path',  type=str, default=None,
    help='The path to a checkpoint from which to fine-tune.')

p.add_argument('--checkpoint_exclude_scopes',  type=str, default=None,
    help='Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.'
    'By default, only the Logits layer is excluded')

p.add_argument('--trainable_scopes', type=str, default=None,
    help='Comma-separated list of scopes to filter the set of variables to train.'
    'By default, only the Logits layer is trained. None would train all the variables.')

p.add_argument('--ignore_missing_vars', type=bool, default=True, help='When restoring a checkpoint would ignore missing variables.')

p.add_argument('--final_endpoint', type=str, default=None, help='Specifies the endpoint to construct the network up to.'
    'By default, None would be the last layer before Logits.') # this argument was added for modbilenet_v1.py

#######################
# Experiment Details #
#######################

p.add_argument('--experiment_tag', type=str, default='', help='Internal tag for experiment')

p.add_argument('--experiment_name', type=str, default=None, help= ' If None a new experiment folder is created. Naming convension experiment_number')

#######################
# Preprocessing Flags #
#######################


p.add_argument('--add_image_summaries', type=bool, default=True, help='Enable image summaries.')

p.add_argument('--apply_image_augmentation', type=bool, default=True, help='Enable random image augmentation during preprocessing for training.')

p.add_argument('--random_image_crop', type=bool, default=True, help='Enable random cropping of images. Only Enabled if apply_image_augmentation flag is also enabled')

p.add_argument('--min_object_covered', type=float, default=0.9, help='The remaining cropped image must contain at least this fraction of the whole image. Only Enabled if apply_image_augmentation flag is also enabled')

p.add_argument('--random_image_rotation', type=bool, default=True, help='Enable random image rotation counter-clockwise by 90, 180, 270, or 360 degrees. Only Enabled if apply_image_augmentation flag is also enabled')

p.add_argument('--random_image_flip', type=bool, default=True, help='Enable random image flip (horizontally). Only Enabled if apply_image_augmentation flag is also enabled')

p.add_argument('--roi', type=str, default=None, help='Specifies the coordinates of an ROI for cropping the input images.Expects four integers in the order of roi_y_min, roi_x_min, roi_height, roi_width, image_height, image_width. Only applicable to mobilenet_preprocessing pipeline ')

model_name_to_variables = {
  'mobilenet_v1_025':'MobilenetV1',
  'mobilenet_v1_050':'MobilenetV1',
  'mobilenet_v1_075':'MobilenetV1',
  'mobilenet_v1':'MobilenetV1',
  'inception_v1':'InceptionV1'
  }
FLAGS = p.parse_args()

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

def download_and_extract_file(checkpoint_path, url):
    checkpoint_dir = checkpoint_path.split('/')[0:3]
    checkpoint_dir = '/'.join(checkpoint_dir)
    # print(url, checkpoint_dir)
    ftpstream = urllib.request.urlopen(url)
    thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
    thetarfile.extractall(path=checkpoint_dir)

def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if not FLAGS.checkpoint_path:
      # download imagenet pre-trained model weights
      if FLAGS.model_name == 'inception_v1':
          checkpoint_path = './imagenet_checkpoints/inception_v1_224/inception_v1.ckpt'
          if not os.path.isfile(checkpoint_path):
              url = 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz'
              download_and_extract_file(checkpoint_path, url)
          # exclusions = ['InceptionV1/Logits']
      elif FLAGS.model_name == 'mobilenet_v1':
          checkpoint_path = './imagenet_checkpoints/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt'
          if not os.path.isfile(checkpoint_path):
              url = os.path.join('http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz')
              download_and_extract_file(checkpoint_path, url)
          # exclusions = ['MobilenetV1/Logits']
      elif FLAGS.model_name == 'mobilenet_v1_075':
          checkpoint_path = './imagenet_checkpoints/mobilenet_v1_0.75_224/mobilenet_v1_0.75_224.ckpt'
          if not os.path.isfile(checkpoint_path):
              url = os.path.join('http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_224.tgz')
              download_and_extract_file(checkpoint_path, url)
      elif FLAGS.model_name == 'mobilenet_v1_050':
          checkpoint_path = './imagenet_checkpoints/mobilenet_v1_0.5_224/mobilenet_v1_0.5_224.ckpt'
          if not os.path.isfile(checkpoint_path):
              url = os.path.join('http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_224.tgz')
              download_and_extract_file(checkpoint_path, url)
      elif FLAGS.model_name == 'mobilenet_v1_025':
          checkpoint_path = './imagenet_checkpoints/mobilenet_v1_0.25_224/mobilenet_v1_0.25_224.ckpt'
          if not os.path.isfile(checkpoint_path):
              url = os.path.join('http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_224.tgz')
              download_and_extract_file(checkpoint_path, url)
  else:
      checkpoint_path = FLAGS.checkpoint_path

  # exclude scope
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
  else:
    model_variables = model_name_to_variables.get(FLAGS.model_name)
    exclusions = [model_variables+'/Logits']

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        break
      else:
        variables_to_restore.append(var)

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
      print('################', FLAGS.model_name)
      if FLAGS.model_name.startswith('inception_v1'):
          scopes = ['InceptionV1/Logits', 'BatchNorm']
      elif FLAGS.model_name.startswith('mobilenet_v1'):
          scopes = ['MobilenetV1/Logits', 'BatchNorm']
  else:
      scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []

  for scope in scopes:
  	variables = []
  	for variable in tf.trainable_variables():
  		if scope in variable.name:
  			variables.append(variable)
  	variables_to_train.extend(variables)
  	print('######## Trainable variables from name scope', scope, '\n', variables)
  return list(set(variables_to_train))


def main():

  # check required input arguments
  if not FLAGS.project_name:
    raise ValueError('You must supply a project name with --project_name')
  if not FLAGS.dataset_name:
    raise ValueError('You must supply a dataset name with --dataset_name')
  if not FLAGS.model_name in model_name_to_variables:
    raise ValueError('Model name not supported name please select one of the following model architecture: mobilenet_v1, mobilenet_v1_075, mobilenet_v1_050, mobilenet_v1_025, inception_v1')

  # set and check project_dir and experiment_dir.
  project_dir = os.path.join(FLAGS.project_dir, FLAGS.project_name)
  if not FLAGS.experiment_name:
    # list only directories that are names experiment_
      experiment_dir = dataset_utils.create_new_experiment_dir(project_dir)
  else:
      experiment_dir = os.path.join(os.path.join(project_dir, 'experiments'), FLAGS.experiment_name)
      if not os.path.exists(experiment_dir):
          raise ValueError('Experiment directory {} does not exist.'.format(experiment_dir))

  train_dir = os.path.join(experiment_dir, FLAGS.dataset_split_name)
  if not os.path.exists(train_dir):
      os.makedirs(train_dir)

  # set and check dataset_dir
  # if FLAGS.image_dir:
  # prefix = '/opt/ml/'
  #
  # input_path = prefix + 'input/data'
  # print('image_dir ##########', FLAGS.image_dir, input_path)
  # for root, dirs, files in os.walk(input_path):
  #     print(root, dirs, files)

  # ={"additional_framework_parameters":{},"channel_input_dirs":{"train":"/opt/ml/input/data/train"}

  input_path = os.environ.get('SM_TRAINING_ENV')["channel_input_dirs"]["train"]
  print('input path#########', os.environ.get('SM_TRAINING_ENV'), input_path)

  dataset_dir = convert_dataset.convert_img_to_tfrecord(project_dir,
          FLAGS.dataset_name,
          FLAGS.dataset_dir,
          input_path,
          FLAGS.train_percentage,
          FLAGS.validation_percentage,
          FLAGS.test_percentage,
          FLAGS.train_image_size,
          FLAGS.train_image_size)
  # else:
  #     if os.path.isdir(FLAGS.dataset_dir):
  #         dataset_dir = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name)
  #     else:
  #         dataset_dir = os.path.join(os.path.join(project_dir, 'datasets'), FLAGS.dataset_name)
  # if not os.path.isdir(dataset_dir):
  #   raise ValueError('Can not find tfrecord dataset directory {}'. format(dataset_dir))
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
        FLAGS.dataset_name, FLAGS.dataset_split_name, dataset_dir)

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
      if FLAGS.imbalance_correction:
          # specify some class weightings
          class_weights = dataset.sorted_class_weights
          # deduce weights for batch samples based on their true label
          weights = tf.reduce_sum(tf.multiply(labels, class_weights), 1)

          slim.losses.softmax_cross_entropy(
                    logits, labels, label_smoothing=FLAGS.label_smoothing, weights=weights)
      else:
          if 'AuxLogits' in end_points:
            slim.losses.softmax_cross_entropy(
                end_points['AuxLogits'], labels,
                label_smoothing=FLAGS.label_smoothing, weights=0.4,
                scope='aux_loss')
          else:
            slim.losses.softmax_cross_entropy(
                    logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
      #############################
      ## Calculation of metrics ##
      #############################
      accuracy, accuracy_op = tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(logits, 1))
      precision, precision_op = tf.metrics.average_precision_at_k(tf.argmax(labels, 1), logits, 1)

      with tf.device('/device:CPU:0'):
        for class_id in range(dataset.num_classes):
          precision_at_k, precision_at_k_op = tf.metrics.precision_at_k(tf.argmax(labels, 1), logits, k=1, class_id=class_id)
          recall_at_k, recall_at_k_op = tf.metrics.recall_at_k(tf.argmax(labels, 1), logits, k=1, class_id=class_id)
          tf.add_to_collection('precision_at_{}'.format(class_id), precision_at_k)
          tf.add_to_collection('precision_at_{}_op'.format(class_id), precision_at_k_op)
          tf.add_to_collection('recall_at_{}'.format(class_id), recall_at_k)
          tf.add_to_collection('recall_at_{}_op'.format(class_id), recall_at_k_op)

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
      precision_at_k = tf.get_collection('precision_at_{}'.format(class_id))
      precision_at_k_op = tf.get_collection('precision_at_{}_op'.format(class_id))
      recall_at_k = tf.get_collection('recall_at_{}'.format(class_id))
      recall_at_k_op = tf.get_collection('recall_at_{}_op'.format(class_id))

      precision_at_k = tf.reduce_mean(tf.stack(precision_at_k, axis=0))
      precision_at_k_op = tf.reduce_mean(tf.stack(precision_at_k_op, axis=0))
      recall_at_k = tf.reduce_mean(tf.stack(recall_at_k, axis=0))
      recall_at_k_op = tf.reduce_mean(tf.stack(recall_at_k_op, axis=0))

      summaries.add(tf.summary.scalar('Metrics/class_{}_precision'.format(class_id), precision_at_k))
      summaries.add(tf.summary.scalar('op/class_{}_precision_op'.format(class_id), precision_at_k_op))
      summaries.add(tf.summary.scalar('Metrics/class_{}_recall'.format(class_id), recall_at_k))
      summaries.add(tf.summary.scalar('op/class_{}_recall_op'.format(class_id), recall_at_k_op))

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
    loss = total_loss
    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)
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
            print('global step {:d}: loss = {:1.4f} ({:.3f} sec/step)'.format(np_global_step, total_loss, time_elapsed))

      if 'should_stop' in train_step_kwargs:
        should_stop = sess.run(train_step_kwargs['should_stop'])
      else:
        should_stop = False

      return total_loss, should_stop or train_step_fn.should_stop


    train_step_fn.should_stop = False
    # train_step_fn.accuracy = accuracy

    def exit_gracefully(signum, frame) :
      interrupted = datetime.datetime.utcnow()
      # if not experiment_file is None :
      print('Interrupted on (UTC): ', interrupted, sep='', file=experiment_file)
      experiment_file.flush()
      train_step_fn.should_stop = True
      print('Interrupted on (UTC): ', interrupted, sep='')

    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    start = datetime.datetime.utcnow()
    print('Started on (UTC): ', start, sep='')

    # record script flags (FLAGS). write to experiment file
    experiment_file_path = os.path.join(train_dir, 'experiment_setting.txt')
    experiment_file = open(experiment_file_path, 'w')
    print('Experiment metadata file:', file=experiment_file)
    print(experiment_file_path, file=experiment_file)
    print('========================', file=experiment_file)
    print('All command-line flags:', file=experiment_file)
    print(experiment_file_path, file=experiment_file)
    for key,value in vars(FLAGS).items():
      print(key, ' : ', value, sep='', file=experiment_file)
    print('========================', file=experiment_file)
    print('Started on (UTC): ', start, sep='', file=experiment_file)
    experiment_file.flush()

    slim.learning.train(
        train_tensor,
        train_step_fn=train_step_fn,
        logdir=train_dir,
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
    # generate and save graph (output file model_name_graph.pb)
    print('Generate frozen graph')
    # TODO: Simplify by loading checkpoint+graph and freezing together (no need to save graph)
    # genrate and save inference graph
    is_training = False
    is_video_model = False
    batch_size = None
    num_frames = None
    quantize = False
    write_text_graphdef = False
    output_file = os.path.join(train_dir, FLAGS.model_name + '_graph.pb')
    export_inference_graph(FLAGS.dataset_name, dataset_dir,  FLAGS.model_name, FLAGS.labels_offset, is_training, FLAGS.final_endpoint, FLAGS.train_image_size, FLAGS.use_grayscale, is_video_model, batch_size, num_frames, quantize, write_text_graphdef, output_file)
    # record training session end
    print('Finished on (UTC): ', finish, sep='', file=experiment_file)
    print('Elapsed: ', finish-start, sep='', file=experiment_file)
    experiment_file.flush()

if __name__ == '__main__':
  # tf.app.run()
  main()
  # TODO: Fix the mess below.

  # The code below generates a frozen graph from the last trained checkpoint and defined graph (export_inference_graph)
  # set and check project_dir and experiment_dir.
  project_dir = os.path.join(FLAGS.project_dir, FLAGS.project_name)
  if not FLAGS.experiment_name:
    # list only directories that are names experiment_
      # experiment_dir = create_new_experiment_dir(project_dir)
      experiment_dir = dataset_utils.select_latest_experiment_dir(project_dir)
  else:
      experiment_dir = os.path.join(os.path.join(project_dir, 'experiments'), FLAGS.experiment_name)
      if not os.path.exists(experiment_dir):
          raise ValueError('Experiment directory {} does not exist.'.format(experiment_dir))
  train_dir = os.path.join(experiment_dir, FLAGS.dataset_split_name)
  output_file = os.path.join(train_dir, FLAGS.model_name + '_graph.pb')
  if not os.path.isfile(output_file):
      raise ValueError('Graph file not found')
  # load last checkpoint and generate frozen weighted graph from export_inference_graph
  from tensorflow.core.protobuf import saver_pb2
  input_saver = ""
  input_binary=True
  checkpoint_path = tf.train.latest_checkpoint(train_dir)
  print('#######',checkpoint_path)
  if FLAGS.model_name.startswith('inception_v1'):
    output_node_names = "InceptionV1/Predictions/Reshape_1"
  elif FLAGS.model_name.startswith('mobilenet_v1'):
    output_node_names = "MobilenetV1/Predictions/Reshape_1"

  restore_op_name = "save/restore_all"
  filename_tensor_name = "save/Const:0"
  output_graph_name = "{}_{}_{}_frozen.pb".format(FLAGS.project_name, FLAGS.dataset_name, FLAGS.model_name)
  if FLAGS.output_graph:
      output_graph = os.path.join(FLAGS.output_graph, output_graph_name)
  else:
      output_graph = os.path.join(train_dir, output_graph_name)
  clear_devices = True
  initializer_nodes = ""
  variable_names_whitelist = ""
  variable_names_blacklist = ""
  input_meta_graph = ""
  input_saved_model_dir = ""
  saved_model_tags = "serve"
  checkpoint_version = saver_pb2.SaverDef.V2
  freeze_graph(output_file, input_saver, input_binary,
  checkpoint_path, output_node_names,
  restore_op_name, filename_tensor_name,
  output_graph, clear_devices, initializer_nodes,
  variable_names_whitelist, variable_names_blacklist,
  input_meta_graph, input_saved_model_dir,
  saved_model_tags, checkpoint_version)
