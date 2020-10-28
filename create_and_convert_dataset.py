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
"""Create and converts a particular dataset.

Usage:
```shell

$ python create_and_convert_dataset.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/flower_photes
    --tfrecords_dataset_dir=/tmp/flowers
    --validation_percentage=10
    --test_percentage=10


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

from datasets import convert_dataset

FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset to convert, one of "flowers", "cifar10", "mnist", "visualwakewords"'
    )

tf.compat.v1.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the input images are saved.')

tf.compat.v1.app.flags.DEFINE_string(
    'tfrecords_dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')

tf.compat.v1.app.flags.DEFINE_integer(
    'validation_percentage',
    10,
    'What percentage of images to use as a test set.'
    )
tf.compat.v1.app.flags.DEFINE_integer(
    'test_percentage',
    10,
    'What percentage of images to use as a validation set.'
    )
tf.compat.v1.app.flags.DEFINE_integer(
    'image_height',
    224,
    'Target image height after resizing. If None original image size is kepts.'
    )
tf.compat.v1.app.flags.DEFINE_integer(
    'image_width',
    224,
    'Target image width after resizing. If None original image size is kepts.'
    )



def main(_):
  # print(FLAGS.dataset_dir)
  if not FLAGS.dataset_name:
    raise ValueError('You must supply a dataset name with --dataset_name')
  if not os.path.isdir(FLAGS.dataset_dir):
    raise ValueError('You must supply the dataset directory where you current image are stored with --dataset_dir')
  DATASET_DIR = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name)
  # if not FLAGS.tfrecords_dataset_dir:
  #   raise ValueError('You must supply a dataset directory to store the convert tfrecord dataset with --tfrecords_dataset_dir')
  if not FLAGS.tfrecords_dataset_dir:
    TFRECORDS_DATASET_DIR = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name+'_tfrecord')
    # if os.path.isdir(TFRECORDS_DATASET_DIR)
  if len(os.listdir(FLAGS.dataset_dir)):
    # print('#############',FLAGS.validation_percentage, FLAGS.test_percentage)
    convert_dataset.run(FLAGS.dataset_name, DATASET_DIR, TFRECORDS_DATASET_DIR, FLAGS.validation_percentage, FLAGS.test_percentage, FLAGS.image_height, FLAGS.image_width)

  else:
    raise ValueError(
        'dataset_dir [%s] is empty.' % FLAGS.dataset_dir)

if __name__ == '__main__':
  tf.compat.v1.app.run()
