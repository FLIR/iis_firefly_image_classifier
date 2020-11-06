# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import shutil


import tensorflow as tf
from sklearn.model_selection import train_test_split

from datasets import dataset_utils
# import dataset_utils




# The URL where the Flowers data can be downloaded.
# _DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

# The number of images in the validation set.
# _NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5


class ImageReaderPNG(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB png data.
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)
    self._decode_png_float = tf.image.convert_image_dtype(self._decode_png, dtype=tf.float32, saturate=False)

    # self._image = tf.placeholder(dtype=tf.float32)
    self._image_height = tf.placeholder(dtype=tf.int16)
    self._image_width = tf.placeholder(dtype=tf.int16)
    self._resize_image = tf.image.resize_images(self._decode_png_float, [self._image_height, self._image_width])

    # self._encode_png_data = tf.placeholder(dtype=tf.uint8)
    self._resize_image = tf.image.convert_image_dtype(self._resize_image, dtype=tf.uint8, saturate=False)
    self._encode_png = tf.image.encode_png(self._resize_image)

  def read_image_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

  def resize_image(self, sess, image_data, image_height, image_width):
    image_data = sess.run(self._encode_png,
                      feed_dict={self._decode_png_data:image_data,
                                self._image_height:image_height,
                                self._image_width:image_width})
    return image_data

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png_float,
                     feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def encode_png(self, sess, image):
    image_data = sess.run(self._encode_png,
                     feed_dict={self._encode_png_data: image})
    return image_data



def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  # flower_root = os.path.join(dataset_dir, 'flower_photos')
  directories = []
  class_names = []
  for filename in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id, dataset_name):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      dataset_name, split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, dataset_name, image_height, image_width):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation', 'test']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
  # print('##############', filenames[0])
  num_samples_per_class = dict()

  with tf.Graph().as_default():
    image_reader = ImageReaderPNG()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id, dataset_name)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):

            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.GFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)
            # print('################# before', height, width)
            if image_height and image_width:
                image_data = image_reader.resize_image(sess, image_data, image_height, image_width)
                height, width = image_reader.read_image_dims(sess, image_data)
                # print('################# after', height, width)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            if class_name not in num_samples_per_class:
                num_samples_per_class[class_name] = 0
            num_samples_per_class[class_name] += 1
            class_id = class_names_to_ids[class_name]
            image_name = bytes(filenames[i], 'utf-8')
            # print('####',image_name,height, width, class_id )
            example = dataset_utils.image_to_tfexample(
                image_data, image_name, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()
  # num_samples_per_classl['class_weights'] = dict{}
  # for num_samples in num_samples_per_class:
  return num_samples_per_class


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'flower_photos')
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir, dataset_name):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id, dataset_name)
      if not tf.gfile.Exists(output_filename):
        return False
  return True

def convert_img_to_tfrecord(project_dir,
        dataset_name,
        dataset_dir,
        image_dir,
        train_percentage,
        validation_percentage,
        test_percentage,
        image_height,
        image_width,
        **kwargs):
  """Runs conversion operation.

    Args:
      project_dir: (Default) Directory where the newly created dataset with tfrecord will be stored.
      dataset_name: The name of dataset that is created from input dataset.
      dataset_dir: (optional) Directory where the newly created dataset with tfrecord will be stored.
      image_dir: The dataset directory where the dataset is stored.
      train_percentage: train dataset
      validation_percentage: validation dataset
      test_percentage: test dataset
      image_height: Target image height for tfrecord.
      image_width: Target image width for tfrecord.
  """
  # print(dataset_dir)

  # if not os.listdir(image_dir):
  #   raise ValueError('No label folders found in image directory --image_dir')
  if not image_dir:
    raise ValueError('You must supply a image directory with --image_dir')

  if dataset_dir:
    dataset_dir = os.path.join(dataset_dir, dataset_name)
  else:
    # initialize default directories
    # project_dir = os.path.join(project_dir, project_name)
    dataset_dir = os.path.join(os.path.join(project_dir, 'datasets'), dataset_name)
  # delete dataset directory if it exists
  if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)
  # call convert dataset function
  if len(os.listdir(image_dir)):
    # run(dataset_name, image_dir, dataset_dir, train_percentage, validation_percentage, test_percentage, image_height, image_width)
    # create new dataset
    # dataset_dir = os.path.join(dataset_dir, dataset_name)
    if not tf.gfile.Exists(dataset_dir):
      tf.gfile.MakeDirs(dataset_dir)

    if train_percentage + validation_percentage + test_percentage > 100:
      raise ValueError('The sum of train, validation, and test percentages can not be greater than 100')
    # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
    photo_filenames, class_names = _get_filenames_and_classes(image_dir)
    # print('############', class_names)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Divide into train, validation and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    dataset_split = dict()
    training_filenames = photo_filenames[:]

    if train_percentage > 0:
      training_filenames, train_filenames = train_test_split(training_filenames, test_size=train_percentage/100, random_state=_RANDOM_SEED)
      train_size = len(train_filenames)
      print('Number of training images: ', train_size)
      num_samples_per_class = _convert_dataset('train', train_filenames, class_names_to_ids, dataset_dir, dataset_name, image_height, image_width)
      dataset_split['train'] = train_size
      dataset_split['train_per_class'] = num_samples_per_class

    if test_percentage > 0:
      training_filenames, test_filenames = train_test_split(training_filenames, test_size=test_percentage/100, random_state=_RANDOM_SEED)
      test_size = len(test_filenames)
      print('Number of test images: ', test_size)
      num_samples_per_class = _convert_dataset('test', test_filenames, class_names_to_ids,
                     dataset_dir, dataset_name, image_height, image_width)
      dataset_split['test'] = test_size
      dataset_split['test_per_class'] = num_samples_per_class

    if validation_percentage > 0:
      training_filenames, validation_filenames = train_test_split(training_filenames, test_size=validation_percentage/100, random_state=_RANDOM_SEED)
      validation_size = len(validation_filenames)
      print('Number of validation images: ', validation_size)
      num_samples_per_class = _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir, dataset_name, image_height, image_width)
      dataset_split['validation'] = validation_size
      dataset_split['validation_per_class'] = num_samples_per_class

    # Finally, write the label and dataset json files:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    dataset_utils.write_dataset_config_json(dataset_name,
                       dataset_dir, class_names,
                       dataset_split)

    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the ',dataset_name,' dataset! under the following directory', dataset_dir)
    return dataset_dir

  else:
    raise ValueError(
        'image directory --image_dir=[%s] is empty'.format(image_dir))

# def run(dataset_name, image_dir, dataset_dir, train_percentage, validation_percentage, test_percentage, image_height, image_width):
#   """Runs the download and conversion operation.
#
#   Args:
#
#     dataset_name: The name of dataset that is created from input dataset.
#     dataset_dir: Directory where the newly created dataset with tfrecord will be stored.
#     image_dataset_dir: The dataset directory where the dataset is stored.
#     validation_percentage: validation dataset
#     test_percentage: test dataset
#   """
#   # create new dataset
#   # dataset_dir = os.path.join(dataset_dir, dataset_name)
#   if not tf.gfile.Exists(dataset_dir):
#     tf.gfile.MakeDirs(dataset_dir)
#
#   if train_percentage + validation_percentage + test_percentage > 100:
#     raise ValueError('The sum of train, validation, and test percentages can not be greater than 100')
#   # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
#   photo_filenames, class_names = _get_filenames_and_classes(image_dir)
#   # print('############', class_names)
#   class_names_to_ids = dict(zip(class_names, range(len(class_names))))
#
#   # Divide into train, validation and test:
#   random.seed(_RANDOM_SEED)
#   random.shuffle(photo_filenames)
#   dataset_split = dict()
#   training_filenames = photo_filenames[:]
#
#   if train_percentage > 0:
#     training_filenames, train_filenames = train_test_split(training_filenames, test_size=train_percentage/100, random_state=_RANDOM_SEED)
#     train_size = len(train_filenames)
#     print('Number of training images: ', train_size)
#     num_samples_per_class = _convert_dataset('train', train_filenames, class_names_to_ids,
#                    dataset_dir, dataset_name, image_height, image_width)
#     dataset_split['train'] = train_size
#     dataset_split['train_per_class'] = num_samples_per_class
#
#   if test_percentage > 0:
#     training_filenames, test_filenames = train_test_split(training_filenames, test_size=test_percentage/100, random_state=_RANDOM_SEED)
#     test_size = len(test_filenames)
#     print('Number of test images: ', test_size)
#     num_samples_per_class = _convert_dataset('test', test_filenames, class_names_to_ids,
#                    dataset_dir, dataset_name, image_height, image_width)
#     dataset_split['test'] = test_size
#     dataset_split['test_per_class'] = num_samples_per_class
#
#   if validation_percentage > 0:
#     training_filenames, validation_filenames = train_test_split(training_filenames, test_size=validation_percentage/100, random_state=_RANDOM_SEED)
#     validation_size = len(validation_filenames)
#     print('Number of validation images: ', validation_size)
#     num_samples_per_class = _convert_dataset('validation', validation_filenames, class_names_to_ids,
#                    dataset_dir, dataset_name, image_height, image_width)
#     dataset_split['validation'] = validation_size
#     dataset_split['validation_per_class'] = num_samples_per_class
#
#   # Finally, write the label and dataset json files:
#   labels_to_class_names = dict(zip(range(len(class_names)), class_names))
#   dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
#   dataset_utils.write_dataset_config_json(dataset_name,
#                      dataset_dir, class_names,
#                      dataset_split)
#
#   # _clean_up_temporary_files(dataset_dir)
#   print('\nFinished converting the ',dataset_name,' dataset! under the following directory', dataset_dir)
