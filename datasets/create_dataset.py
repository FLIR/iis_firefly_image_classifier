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
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from datasets import dataset_utils

slim = contrib_slim

_FILE_PATTERN = '%s_%s_*.tfrecord'


def get_split(dataset_name, split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """

  num_classes, splits_to_sizes, items_to_descriptions = dataset_utils.read_dataset_config_json(dataset_name, dataset_dir)

  if split_name not in splits_to_sizes:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:

    file_pattern = _FILE_PATTERN % (dataset_name, split_name)
  file_pattern = os.path.join(dataset_dir, file_pattern)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/name': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/height': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/width': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
      'image_name': slim.tfexample_decoder.Tensor('image/name'),
      'image_height': slim.tfexample_decoder.Tensor('image/height'),
      'image_width': slim.tfexample_decoder.Tensor('image/width'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  def dataset_class_weight(splits_to_sizes, split_name, label_to_class_number):
    """
    Returns sorted list of class ratios
    """

    num_samples = splits_to_sizes[split_name]
    num_samples_per_class = splits_to_sizes[split_name+'_per_class']
    sorted_class_weights = list()
    sorted_label_name = sorted(label_to_class_number.keys(), key= lambda x: label_to_class_number[x])
    for label_name in sorted_label_name:
        if label_name in num_samples_per_class:
            norm_class_weight = num_samples / (num_samples_per_class[label_name] * num_classes)
            sorted_class_weights.append(norm_class_weight)
    return sorted_class_weights

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    class_to_label_name, label_to_class_number = dataset_utils.read_label_file(dataset_dir)

  num_samples=splits_to_sizes[split_name]
  # if split_name+'_per_class' in splits_to_sizes:
  sorted_class_weights = dataset_class_weight(splits_to_sizes, split_name, label_to_class_number)

  return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=items_to_descriptions,
        num_classes=num_classes,
        labels_to_names=label_to_class_number,
        sorted_class_weights=sorted_class_weights
        )

  # return slim.dataset.Dataset(
  #     data_sources=file_pattern,
  #     reader=reader,
  #     decoder=decoder,
  #     num_samples=num_samples,
  #     items_to_descriptions=items_to_descriptions,
  #     num_classes=num_classes,
  #     labels_to_names=label_to_class_number)
