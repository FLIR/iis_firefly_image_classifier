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
"""Converts a particular image dataset to tfrecord format

Usage:
```shell

$ python convert_image_to_tfrecord.py \
    --project_name=flower_classifier \
    --dataset_name=flowers \
    --image_dir=./tmp/flower_photes


```
Improvment notes:
1- Add condition to check for number of directories (number of classes) and print out
2- Add condition to check there is atleast 10 images per directory or atleast not empty

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import argparse

from datasets import convert_dataset
from datasets import dataset_utils
from datasets import download_and_convert_flowers

# The URL where the Flowers data can be downloaded.
FLOWERS_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
# FLOWERS_DATA_URL = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'

# FLAGS = tf.compat.v1.app.flags.FLAGS
p = argparse.ArgumentParser()

p.add_argument('--project_dir', type=str, default='./project_dir/', help='Directory where checkpoints and event logs are written to.')

p.add_argument('--project_name', type=str, default=None, help= 'Must supply project name examples: flower_classifier, component_classifier')

p.add_argument('--image_dir', type=str, default=None, help='The directory where the input images are saved.')

p.add_argument('--dataset_name', type=str, default='imagenet', help='The name of the dataset to load.')

p.add_argument('--dataset_dir', type=str, default=None, help='The directory where the dataset files are stored.')

p.add_argument('--train_percentage', type=int, default=70, help='What percentage of images to use as a train set.')

p.add_argument('--validation_percentage', type=int, default=10, help='What percentage of images to use as a validation set.')

p.add_argument('--test_percentage', type=int, default=20, help='What percentage of images to use as a test set.')

p.add_argument('--image_height', type=int, default=None, help='Target image height after resizing. If None original image height is used.')

p.add_argument('--image_width', type=int, default=None, help='Target image width after resizing. If None original image width is used.')

FLAGS = p.parse_args()

if __name__ == '__main__':
  # check required input arguments
  if not FLAGS.project_name:
    raise ValueError('You must supply a dataset name with --project_name')
  if not FLAGS.dataset_name:
    raise ValueError('You must supply a dataset name with --dataset_name')

  # check dataset name and image directory
  if FLAGS.dataset_name == 'flowers' and not FLAGS.image_dir:
    # download flowers dataset
    image_dir = os.path.join('./image_dir', 'flower_photos')
    dataset_utils.download_and_uncompress_tarball(FLOWERS_DATA_URL, './image_dir')
  else:
    if not FLAGS.image_dir:
      raise ValueError('You must supply an image directory with --image_dir')
    image_dir= FLAGS.image_dir

  # set project_dir and convert images to tfrecord
  project_dir = os.path.join(FLAGS.project_dir, FLAGS.project_name)
  convert_dataset.convert_img_to_tfrecord(project_dir,
                          FLAGS.dataset_name,
                          FLAGS.dataset_dir,
                          image_dir,
                          FLAGS.train_percentage,
                          FLAGS.validation_percentage,
                          FLAGS.test_percentage,
                          FLAGS.image_height,
                          FLAGS.image_width)
