#!/usr/bin/env python

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

from datasets import imagenet
from nets import inception
from nets import resnet_v1
from nets import inception_utils
from nets import resnet_utils
from preprocessing import inception_preprocessing
from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets import dataset_utils

import numpy as np
import os
import sys
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'test_dir', None, 'Directory where the output .txt file for prediction probabilities is saved .')

tf.app.flags.DEFINE_string(
    'model_name', None, 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The directory where the model was written to or an absolute path to a checkpoint file.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_dir',None,
    'The directory where the dataset files are stored. You can also specify .txt image file, one image filepath per line.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'custom_1_preprocessing_pipline', 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_boolean(
    'tfrecord',False, 'Input file is formatted as TFRecord.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Eval image size.')

tf.app.flags.DEFINE_integer(
    'num_classes', None, 'The number of classes.')

#######################
# Test evaluation Flags #
#######################

tf.app.flags.DEFINE_boolean(
    'test_with_groudtruth',False, 'Evaluate with groudtruth')

tf.app.flags.DEFINE_string(
    'label_file', None, 'Image file, one image per line.')

tf.app.flags.DEFINE_boolean(
    'print_misclassified_test_images',True, ' Whether to print out a list of all misclassified test images.')


FLAGS = tf.app.flags.FLAGS

if FLAGS.test_dir:
    TEST_DIR = os.path.join(FLAGS.test_dir, FLAGS.dataset_split_name)
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
else:
    raise ValueError('You must supply test directory with --test_dir.')


prediction_outfile = os.path.join(TEST_DIR, 'predictions.csv')

def read_label_file(filepath):
    """
    read label file and return two dictionaries. Label to class index and class index to label.

    Args:
        path: String. file path to label .txt file, where line index represents the class and the line string element represents the label
    Returns:
        labels_to_class_names: A map of (integer) labels to class names.
        class_to_label_names: A map of class names to (integer) labels.
    """

    if os.path.isfile(filepath):
        # create empty lists and dictionaries
        label_to_class_names = {}
        class_to_label_names = {}
        label_list = list()
        # itorate over the lines to create label/class dictionaries
        with open(filepath, 'r') as label_file:
            for label in label_file.readlines():
                class_label_index = label.split(':')[0].strip()
                class_label = label.split(':')[1].strip()
                # add class:label pairs to class_to_label_names dict
                if class_label_index not in class_to_label_names:
                    class_to_label_names[class_label_index] = class_label
                # add label:class pairs to label_to_class_names dict
                if class_label not in label_to_class_names:
                    label_to_class_names[class_label] = class_label_index

    else:
        raise ValueError('label file [%s] was not recognized' %
                         FLAGS.label_file)


    return class_to_label_names, label_to_class_names


def read_gt_from_filepath(filepath):
    """
    read groudtruth from filepath

    Args:
        path: String.
    Returns:
        A tuple. Image name string, and class label string.
    """

    class_label = filepath.split('/')[-2]
    image_name = filepath.split('/')[-1]

    return image_name, class_label

model_name_to_variables = {'mobilenet_v1_025':'MobilenetV1','mobilenet_v1':'MobilenetV1','inception_v3':'InceptionV3','inception_v4':'InceptionV4','resnet_v1_50':'resnet_v1_50','resnet_v1_152':'resnet_v1_152'}

preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
test_image_size = FLAGS.test_image_size

if FLAGS.tfrecord:
  fls = tf.python_io.tf_record_iterator(path=FLAGS.dataset_dir)
elif os.path.isfile(FLAGS.dataset_dir):
  fls = [s.strip() for s in open(FLAGS.dataset_dir)]
elif os.path.isdir(FLAGS.dataset_dir):
  fls = list()
  for root, dirs, files in os.walk(FLAGS.dataset_dir):
      for file in files:
          if file.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp']:
            fls.append(os.path.join(root,file))


model_variables = model_name_to_variables.get(FLAGS.model_name)
if model_variables is None:
  tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
  sys.exit(-1)

if FLAGS.tfrecord:
  tf.logging.warn('Image name is not available in TFRecord file.')

if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
  checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
else:
  checkpoint_path = FLAGS.checkpoint_path

# if evaluation with groudtruth is set true. read label file, set number of classes and create class-to-label/label-to-class dictionary
num_classes = FLAGS.num_classes
if FLAGS.test_with_groudtruth:
    if FLAGS.label_file:
        num_classes = len(open(FLAGS.label_file).readlines())
        class_to_label_dict, label_to_class_dict = read_label_file(FLAGS.label_file)
    else:
        raise ValueError('You must supply the label file path with --label_file.')

if not num_classes:
    raise ValueError('You must supply number of output classes with --num_classes.')
# print('####################', num_classes)

image_string = tf.placeholder(tf.string) # Entry to the computational graph, e.g. image_string = tf.gfile.FastGFile(image_file).read()

#image = tf.image.decode_image(image_string, channels=3)
image = tf.image.decode_jpeg(image_string, channels=3, try_recover_truncated=True, acceptable_fraction=0.3) ## To process corrupted image files

image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)

network_fn = nets_factory.get_network_fn(FLAGS.model_name, num_classes, is_training=False)

if FLAGS.test_image_size is None:
  test_image_size = network_fn.default_image_size

processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)

processed_images  = tf.expand_dims(processed_image, 0) # Or tf.reshape(processed_image, (1, test_image_size, test_image_size, 3))

logits, _ = network_fn(processed_images)

probabilities = tf.nn.softmax(logits)

init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))


if FLAGS.test_dir:
  with open(prediction_outfile, 'w') as fout:
      h = ['image']
      h.extend(['class%s' % i for i in range(num_classes)])
      h.append('predicted_class')
      fout.write(','.join(h) + '\n')

with tf.Session() as sess:

    counter = 0
    init_fn(sess)
    output_pred = list()
    output_gt = list()
    file_name = list()
    for fl in fls:
      image_name = None
      try:
        if FLAGS.tfrecord is False:
          # read image file
          img = open(fl, 'rb').read()
          # read groudtruth label for model evaluation.
          if FLAGS.test_with_groudtruth:
              image_name, gt_label = read_gt_from_filepath(fl)
              output_gt.append(gt_label)
              # print('image name', image_name, gt_label)
          else:
              image_name = os.path.basename(fl)

        else:
          example = tf.train.Example()
          example.ParseFromString(fl)

          # Note: The key of example.features.feature depends on how you generate tfrecord.
          img = example.features.feature['image/encoded'].bytes_list.value[0] # retrieve image string

          image_name = 'TFRecord'

        probs = sess.run(probabilities, feed_dict={image_string:img})
        #np_image, network_input, probs = sess.run([image, processed_image, probabilities], feed_dict={image_string:x})

      except Exception as e:
        tf.logging.warn('Cannot process image file %s' % fl)
        continue

      # check if groudtruth class label names match with class labels from label_file
      if FLAGS.test_with_groudtruth:
          if gt_label not in list(label_to_class_dict.keys()):
              raise ValueError(f'groundtruth label ({gt_label}) does not match class label in file --label_file. Check image file parent directory names and selected label_file')

      probs = probs[0, 0:]
      # a = [image_name]
      # file_name.append(image_name)
      a = [fl]
      file_name.append(fl)

      a.extend(probs)
      a.append(np.argmax(probs))
      # print(probs.shape)
      if FLAGS.test_with_groudtruth:
          pred_label = class_to_label_dict[str(a[-1])]
      else:
          pred_label = str(a[-1])
      if FLAGS.test_dir is not None:
        with open(prediction_outfile, 'a') as fout:
          fout.write(','.join([str(e) for e in a]))
          fout.write('\n')
      # print(f'image name: {a[0]},     class prediction: {pred_label}')
      counter += 1
      sys.stdout.write(f'\rProcessing images... {str(counter)}/{len(fls)}')
      sys.stdout.flush()
      output_pred.append(pred_label)

    fout.close()
    print(f'\n\nPredition results saved to >>>>>> {prediction_outfile}')

# sess.close()
# misclassified image
if FLAGS.test_with_groudtruth:
    if FLAGS.print_misclassified_test_images:
        print("\n\n\n==================== Misclassified Images ====================")
        count = 0
        for image_name, gt_label, pred_label in zip(file_name, output_gt, output_pred):
              if pred_label != gt_label:
                  count += 1
                  print(f'Image file {image_name} misclassified as {pred_label}. (groundtruth label {gt_label})')
        print(f'\n\nTotal misclassified images {count}/{len(file_name)}')
        print("==============================================================")


    y_true = output_gt
    y_pred = output_pred
    conf_mat_output = confusion_matrix(y_true, y_pred, labels=np.unique(output_gt))
    output_acc = accuracy_score(y_true, y_pred)
    output_precision = precision_score(y_true, y_pred, average='micro', labels=np.unique(output_gt))
    output_recall = recall_score(y_true, y_pred, average='micro', labels=np.unique(output_gt))
    output_f1 = f1_score(y_true, y_pred, average='micro', labels=np.unique(output_gt), zero_division=0)
    #print(conf_mat_output, output_acc, output_precision, output_recall)

    print("\n\n\n==================== Evaluation Result Summary ====================")
    print("Accuracy score : {}".format(output_acc))
    print("Precision score : {}".format(output_precision))
    print("Recall score: {}".format(output_recall))
    print("F1 score: {}".format(output_f1))
    print(classification_report(y_true, y_pred, digits=7, labels=np.unique(output_gt)))
    print("===================================================================")
