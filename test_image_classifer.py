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
    'experiment_dir', None, 'Directory where the output .txt file for prediction probabilities is saved .')

tf.app.flags.DEFINE_string('experiment_name', None, ' If None a new experiment folder is created. Naming convension experiment_number')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v1', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The directory where the model was written to or an absolute path to a checkpoint file.')

tf.app.flags.DEFINE_string(
    'final_endpoint', None,
    'Specifies the endpoint to construct the network up to.'
    'By default, None would be the last layer before Logits.') # this argument was added for modbilenet_v1.py

tf.app.flags.DEFINE_bool('use_grayscale', False,
                         'Whether to convert input images to grayscale.')
#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', None, 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_dir',None,
    'The directory where the dataset files are stored. You can also specify .txt image file, one image filepath per line.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'custom_1_preprocessing_pipline', 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_boolean(
    'tfrecord',True, 'Input file is formatted as TFRecord.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Eval image size.')

tf.app.flags.DEFINE_integer(
    'num_classes', None, 'The number of classes.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

#######################
# Test evaluation Flags #
#######################

tf.app.flags.DEFINE_boolean(
    'test_with_groudtruth',True, 'Evaluate with groudtruth')

tf.app.flags.DEFINE_string(
    'label_file', None, 'Image file, one image per line.')

tf.app.flags.DEFINE_boolean(
    'print_misclassified_test_images',True, ' Whether to print out a list of all misclassified test images.')


FLAGS = tf.app.flags.FLAGS

# set up experiment directory
if FLAGS.experiment_dir:
    experiment_dir = FLAGS.experiment_dir
    experiment_name = FLAGS.experiment_name
    # create a new experiment directory if experiment_name is none).
    if not FLAGS.experiment_name:
        # list only directories that are names experiment_
        output_dirs = [x[0] for x in os.walk(experiment_dir) if 'experiment_' in x[0].split('/')[-1]]
        if not output_dirs:
            raise ValueError('No experiment folders found. check evaluation directory with --experiment_dir and assign experiment name with --experiment_name.')
        experiment_name = 'experiment_'+ str(len(output_dirs))
    # exports experiment number to guild (guild compare)
    try:
        experiment_number = experiment_name.split('_')[-1]
        experiment_number = int(experiment_number)
        print('experiment number: {}'.format(experiment_number))
        
    except ValueError:
        pass  # it was a string, not an int.
    
    experiment_dir = os.path.join(os.path.join(experiment_dir, experiment_name), FLAGS.dataset_split_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
else:
    raise ValueError('You must supply train directory with --experiment_dir.')


# if FLAGS.experiment_dir:
#     experiment_dir = os.path.join(FLAGS.experiment_dir, FLAGS.dataset_split_name)
#     if not os.path.exists(experiment_dir):
#         os.makedirs(experiment_dir)
# else:
#     raise ValueError('You must supply test directory with --experiment_dir.')

PREDICTION_FILE = os.path.join(experiment_dir, 'predictions.csv')

model_name_to_variables = {'mobilenet_v1_025':'MobilenetV1','mobilenet_v1_050':'MobilenetV1','mobilenet_v1_075':'MobilenetV1','mobilenet_v1':'MobilenetV1','inception_v1':'InceptionV1'}

preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
test_image_size = FLAGS.test_image_size
fls = list()

####################
# create dataset list
####################

# Read tfrecord file
if FLAGS.tfrecord:
    if os.path.isfile(FLAGS.dataset_dir):
        fls = list(tf.python_io.tf_record_iterator(path=FLAGS.dataset_dir))
    else:
        if not os.path.isdir(FLAGS.dataset_dir):
          raise ValueError('You must supply the dataset directory with --dataset_dir')
        DATASET_DIR = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name+'_tfrecord')
        if not os.path.isdir(DATASET_DIR):
          raise ValueError(f'Can not find tfrecord dataset directory {DATASET_DIR}')
        file_pattern = '_'.join([FLAGS.dataset_name, FLAGS.dataset_split_name])
        for root, dirs, files in os.walk(DATASET_DIR):
            for file in files:
                if file.startswith(file_pattern):
                    if file.endswith('.tfrecord'):
                        # print(file)
                        fls_temp = list(tf.python_io.tf_record_iterator(path=os.path.join(root, file)))
                        fls.extend(fls_temp)
                    else:
                        raise ValueError(f'No .tfrecord files that start with {file_pattern}. Check --dataset_name, --dataset_dir, and --dataset_split_name flags')

        if not fls:
            raise ValueError('No data was found in .tfrecord file')

# create path list from file (.txt) with a list of path to image files.
elif os.path.isfile(FLAGS.dataset_dir):
  fls = [s.strip() for s in open(FLAGS.dataset_dir)]

# create path list from directory
elif os.path.isdir(FLAGS.dataset_dir):
  fls = list()
  for root, dirs, files in os.walk(FLAGS.dataset_dir):
      for file in files:
          if file.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp']:
            fls.append(os.path.join(root,file))
          else:
            print(f'Coud not process the following file {os.path.join(root,file)}')


model_variables = model_name_to_variables.get(FLAGS.model_name)
if model_variables is None:
  tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
  sys.exit(-1)

if FLAGS.tfrecord:
  tf.logging.warn('Image name is not available in TFRecord file.')

tf.logging.info('Evaluating checkpoint')#: %s' % checkpoint_path)
# if checkpoint_path flag is none, look for checkpoint in experiment train directory
if FLAGS.checkpoint_path is None:
    checkpoint_path = '/'.join(experiment_dir.split('/')[:-1])
    checkpoint_path = os.path.join(checkpoint_path, 'train')
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
else:
    # checkpoint_path = FLAGS.checkpoint_path
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

tf.logging.info('Evaluating checkpoint: %s' % checkpoint_path)


# if evaluation with groudtruth is set true. read label file, set number of classes and create class-to-label/label-to-class dictionary
num_classes = FLAGS.num_classes
if FLAGS.test_with_groudtruth:
    if FLAGS.label_file:
        class_to_label_dict, label_to_class_dict = dataset_utils.read_label_file(FLAGS.label_file)
        num_classes = len(class_to_label_dict.keys())
    elif FLAGS.tfrecord:
        class_to_label_dict, label_to_class_dict = dataset_utils.read_label_file(os.path.join(DATASET_DIR, 'labels.txt'))
        num_classes = len(class_to_label_dict.keys())
    else:
        raise ValueError('You must supply the label file path with --label_file.')

if not num_classes:
    raise ValueError('You must supply number of output classes with --num_classes.')
# print('####################', num_classes)

image_string = tf.placeholder(tf.string) # Entry to the computational graph, e.g. image_string = tf.gfile.FastGFile(image_file).read()

#image = tf.image.decode_image(image_string, channels=3)
image = tf.image.decode_png(image_string, channels=3)#, try_recover_truncated=True, acceptable_fraction=0.3) ## To process corrupted image files

####################
# Select the model #
####################

network_fn = nets_factory.get_network_fn(
    FLAGS.model_name,
    num_classes=(num_classes - FLAGS.labels_offset),
    is_training=False,
    final_endpoint=FLAGS.final_endpoint)


#####################################
# Select the preprocessing function #
#####################################
preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name,
    is_training=False,
    use_grayscale=FLAGS.use_grayscale)

test_image_size = FLAGS.test_image_size or network_fn.default_image_size

processed_image = image_preprocessing_fn(image, test_image_size, test_image_size) #,roi=_parse_roi())

processed_images  = tf.expand_dims(processed_image, 0) # Or tf.reshape(processed_image, (1, test_image_size, test_image_size, 3))

logits, _ = network_fn(processed_images)

probabilities = tf.nn.softmax(logits)

init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))


if FLAGS.experiment_dir:
  with open(PREDICTION_FILE, 'w') as fout:
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
      # try:
        if FLAGS.tfrecord is False:
          # read image file
          img = open(fl, 'rb').read()
          # read groudtruth label for model evaluation.
          if FLAGS.test_with_groudtruth:
              gt_label = fl.split('/')[-2]
              # image_name, gt_label = read_gt_from_filepath(fl)
              output_gt.append(gt_label)
              # print('image name', image_name, gt_label)
          # else:
          #     image_name = os.path.basename(fl)
          image_name = fl.split('/')[-1]
          a = [fl]
          file_name.append(fl)

        else:
          example = tf.train.Example()
          example.ParseFromString(fl)

          # Note: The key of example.features.feature depends on how you generate tfrecord.
          img = example.features.feature['image/encoded'].bytes_list.value # retrieve image string
          img = list(img)[0]
          # print('##########', type(img))
          image_file = example.features.feature['image/name'].bytes_list.value
          # print('##############', type(list(image_file)))
          # print('##############', list(image_file))
          image_file = list(image_file)[0].decode('utf-8')
          # print('##############', image_name)
          # image_height = example.features.feature['image/height'].int64_list.value
          # image_width = example.features.feature['image/width'].int64_list.value
          # print('###################', type(img), type(image_name))
          if FLAGS.test_with_groudtruth:
              gt_label = example.features.feature['image/class/label'].int64_list.value
              gt_label = list(gt_label)[0]
              gt_label = class_to_label_dict[str(gt_label)]
              output_gt.append(gt_label)
              # print('image name', image_name, gt_label)
          # else:
          #     image_name = image_name.split('/')[-1]
          # image_name = 'TFRecord'
          a = [image_file]
          file_name.append(image_file)
          image_name = image_file.split('/')[-1]



          probs = sess.run(probabilities, feed_dict={image_string:img})
        # output_1 = example.features.feature['image/class/label'].bytes_list.value[0]
        #np_image, network_input, probs = sess.run([image, processed_image, probabilities], feed_dict={image_string:x})

      # except Exception as e:
      #   # tf.logging.warn('Cannot process image file %s' % fl)
      #   continue

      # check if groudtruth class label names match with class labels from label_file
          if FLAGS.test_with_groudtruth:
              if gt_label not in list(label_to_class_dict.keys()):
                  raise ValueError(f'groundtruth label ({gt_label}) does not match class label in file --label_file. Check image file parent directory names and selected label_file')

          probs = probs[0, 0:]
          # a = [image_name]
          # file_name.append(image_name)

          a.extend(probs)
          a.append(np.argmax(probs))
          # print(probs.shape)
          if FLAGS.test_with_groudtruth:
              pred_label = class_to_label_dict[str(a[-1])]
          else:
              pred_label = str(a[-1])
          # if FLAGS.experiment_dir is not None:
          with open(PREDICTION_FILE, 'a') as fout:
            fout.write(','.join([str(e) for e in a]))
            fout.write('\n')
          # print(f'image name: {a[0]},     class prediction: {pred_label}')
          counter += 1
          sys.stdout.write(f'\rProcessing images... {str(counter)}/{len(fls)}')
          sys.stdout.flush()
          output_pred.append(pred_label)

    fout.close()
    print(f'\n\nPredition results saved to >>>>>> {PREDICTION_FILE}')

sess.close()
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
    # output_f1 = f1_score(y_true, y_pred, average='micro', labels=np.unique(output_gt), zero_division=0)
    #print(conf_mat_output, output_acc, output_precision, output_recall)

    print("\n\n\n==================== Evaluation Result Summary ====================")
    print("Accuracy score : {}".format(output_acc))
    print("Precision score : {}".format(output_precision))
    print("Recall score: {}".format(output_recall))
    # print("F1 score: {}".format(output_f1))
    print(classification_report(y_true, y_pred, digits=7, labels=np.unique(output_gt)))
    print("===================================================================")
