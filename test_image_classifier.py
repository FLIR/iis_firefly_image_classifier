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
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'project_dir', './project_dir', 'default project folder. all prject folder are stored.')

tf.app.flags.DEFINE_string('project_name', None, 'Must supply a project name examples: flower_classifier, component_classifier')

tf.app.flags.DEFINE_string('experiment_name', None, 'If None the highest experiment number (The number of experiment folders) is selected.')

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

tf.app.flags.DEFINE_string(
    'label_file', None, 'Image file, one image per line.')

tf.app.flags.DEFINE_boolean(
    'print_misclassified_test_images', True, ' Whether to print out a list of all misclassified test images.')

FLAGS = tf.app.flags.FLAGS

def select_latest_experiment_dir(project_dir):
    output_dirs = [x[0] for x in os.walk(project_dir) if 'experiment_' in x[0].split('/')[-1]]
    if not output_dirs:
        raise ValueError('No experiments found in project folder: {}. Check project folder or specify experiment name with --experiment_name flag'.format(project_dir))
    experiment_number = max([int(x.split('_')[-1]) for x in output_dirs])
    experiment_name = 'experiment_'+ str(experiment_number)

    print('experiment number: {}'.format(experiment_number))
    experiment_dir = os.path.join(os.path.join(project_dir, 'experiments'), experiment_name)

    return experiment_dir

def main(FLAGS):
  # check required input arguments
  # if not FLAGS.project_name:
  #   raise ValueError('You must supply a project name with --project_name')
  # if not FLAGS.dataset_name:
  #   raise ValueError('You must supply a dataset name with --dataset_name')
  # AWS environment variables
  aws_env_var = json.loads(os.environ.get('SM_TRAINING_ENV'))
  project_dir = os.path.join(FLAGS.project_dir, aws_env_var["job_name"])

  # set and check project_dir and experiment_dir.
  # project_dir = os.path.join(FLAGS.project_dir, FLAGS.project_name)
  if not FLAGS.experiment_name:
    # list only directories that are names experiment_
    experiment_dir = select_latest_experiment_dir(project_dir)
  else:
    experiment_dir = os.path.join(os.path.join(project_dir, 'experiments'), FLAGS.experiment_name)
    if not os.path.exists(experiment_dir):
        raise ValueError('Experiment directory {} does not exist.'.format(experiment_dir))

  test_dir = os.path.join(experiment_dir, FLAGS.dataset_split_name)
  if not os.path.exists(test_dir):
    # raise ValueError('Can not find evalulation directory {}'.format(eval_dir))
    os.makedirs(test_dir)

  # set and check dataset directory
  if FLAGS.dataset_dir:
      dataset_dir = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name)
  else:
      dataset_dir = os.path.join(os.path.join(project_dir, 'datasets'), FLAGS.dataset_name)
  if not os.path.isdir(dataset_dir):
    raise ValueError('Can not find tfrecord dataset directory {}'.format(dataset_dir))

  prediction_file = os.path.join(test_dir, 'predictions.csv')

  ####################
  # create dataset list
  ####################
  fls = list()
  file_pattern = '_'.join([FLAGS.dataset_name, FLAGS.dataset_split_name])
  for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file_pattern in file:
            if file.endswith('.tfrecord'):
                fls_temp = list(tf.python_io.tf_record_iterator(path=os.path.join(root, file)))
                fls.extend(fls_temp)
            else:
                raise ValueError('No .tfrecord files that start with {}. Check --dataset_name, --dataset_dir, and --dataset_split_name flags'.format(file_pattern))
  # raise error if no tfrecord files found in dataset directory
  if not fls:
    raise ValueError('No data was found in .tfrecord file')

  # set and check number of classes in dataset
  num_classes = FLAGS.num_classes
  # if FLAGS.tfrecord:
  class_to_label_dict, label_to_class_dict = dataset_utils.read_label_file(os.path.join(dataset_dir, 'labels.txt'))
  num_classes = len(class_to_label_dict.keys())
      # else:
      #     raise ValueError('You must supply the label file path with --label_file.')
  if not num_classes:
      raise ValueError('You must supply number of output classes with --num_classes.')

  # set checkpoint path
  if FLAGS.checkpoint_path is None:
      # checkpoint_path = '/'.join(project_dir.split('/')[:-1])
      checkpoint_path = os.path.join(experiment_dir, 'train')
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
  else:
      # checkpoint_path = FLAGS.checkpoint_path
      if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
      else:
          checkpoint_path = FLAGS.checkpoint_path

######################
# get model variables#
######################
  model_name_to_variables = {
    'mobilenet_v1_025':'MobilenetV1',
    'mobilenet_v1_050':'MobilenetV1',
    'mobilenet_v1_075':'MobilenetV1',
    'mobilenet_v1':'MobilenetV1',
    'inception_v1':'InceptionV1'
    }

#####################################
# Select the preprocessing function #
#####################################
  tf.reset_default_graph()
  model_variables = model_name_to_variables.get(FLAGS.model_name)
  if model_variables is None:
      tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
      sys.exit(-1)


  image_string = tf.placeholder(name='input', dtype=tf.string)
  # Entry to the computational graph, e.g.
  # image_string = tf.gfile.FastGFile(image_file).read()

  image = tf.image.decode_png(image_string, channels=3)
  # image = tf.image.decode_image(image_string, channels=3)

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

  test_image_size = network_fn.default_image_size

  processed_image = image_preprocessing_fn(image, test_image_size, test_image_size) #,roi=_parse_roi())

  processed_images  = tf.expand_dims(processed_image, 0, name='input_after_preprocessing')

  logits, _ = network_fn(processed_images)

  probabilities = tf.nn.softmax(logits)

  init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))


  # if FLAGS.project_dir:
  with open(prediction_file, 'w') as fout:
      h = ['image']
      h.extend(['class%s' % i for i in range(num_classes)])
      h.append('predicted_class')
      fout.write(','.join(h) + '\n')

  with tf.Session() as sess:
        sess = tf.Session()
        # fls = list()
        counter = 0
        print('\nLoading from checkpoint file {}\n'.format(checkpoint_path))
        init_fn(sess)
        # print([n.name for n in tf.get_default_graph().as_graph_def().node if 'input' in n.name])
        output_pred = list()
        output_gt = list()
        file_name = list()
        # print('##########', len(fls))
        for fl in fls:

              image_name = None
              # print('#############')
              example = tf.train.Example()
              example.ParseFromString(fl)
              # Note: The key of example.features.feature depends on how you generate tfrecord.
              # read image bytes
              img = example.features.feature['image/encoded'].bytes_list.value # retrieve image string
              img = list(img)[0]
              # read image file name
              image_file = example.features.feature['image/name'].bytes_list.value
              image_file = list(image_file)[0].decode('utf-8')

              # if FLAGS.test_with_groudtruth:
              gt_label = example.features.feature['image/class/label'].int64_list.value
              gt_label = list(gt_label)[0]
              gt_label = class_to_label_dict[str(gt_label)]
              output_gt.append(gt_label)
              a = [image_file]
              file_name.append(image_file)
              image_name = image_file.split('/')[-1]
              probs = sess.run(probabilities, feed_dict={image_string:img})

          # check if groudtruth class label names match with class labels from label_file
              if gt_label not in list(label_to_class_dict.keys()):
                  raise ValueError('groundtruth label ({}) does not match class label in file --label_file. Check image file parent directory names and selected label_file'.format(gt_label))

              probs = probs[0, 0:]
              a.extend(probs)
              a.append(np.argmax(probs))
              pred_label = class_to_label_dict[str(a[-1])]
              with open(prediction_file, 'a') as fout:
                fout.write(','.join([str(e) for e in a]))
                fout.write('\n')
              counter += 1
              sys.stdout.write('\rProcessing images... {}/{}'.format(str(counter), len(fls)))
              sys.stdout.flush()
              output_pred.append(pred_label)

        fout.close()
        print('\n\nPredition results saved to >>>>>> {}'.format(prediction_file))
  # misclassified image
  if FLAGS.print_misclassified_test_images:
    test_file = os.path.join(test_dir, 'results.txt')
    with open(test_file, 'w') as f:
        print("\n\n\n==================== Misclassified Images ====================", file=f)
        count = 0
        for image_name, gt_label, pred_label in zip(file_name, output_gt, output_pred):
              if pred_label != gt_label:
                  count += 1
                  print('Image file {} misclassified as {}. (groundtruth label {})'.format(image_name, pred_label, gt_label), file=f)
        print('\n\nTotal misclassified images {}/{}'.format(str(count), len(file_name)), file=f)
        print("==============================================================", file=f)
        y_true = output_gt
        y_pred = output_pred
        conf_mat_output = confusion_matrix(y_true, y_pred, labels=np.unique(output_gt))
        output_acc = accuracy_score(y_true, y_pred)
        output_precision = precision_score(y_true, y_pred, average='micro', labels=np.unique(output_gt))
        output_recall = recall_score(y_true, y_pred, average='micro', labels=np.unique(output_gt))
        print("\n\n\n==================== Evaluation Result Summary ====================", file=f)
        print("Accuracy score : {}".format(output_acc),  file=f)
        print("Precision score : {}".format(output_precision), file=f)
        print("Recall score: {}".format(output_recall), file=f)
        # print("F1 score: {}".format(output_f1))
        print(classification_report(y_true, y_pred, digits=7, labels=np.unique(output_gt)), file=f)
        print("===================================================================", file=f)

    print("\n\n\n==================== Misclassified Images ====================")
    count = 0
    for image_name, gt_label, pred_label in zip(file_name, output_gt, output_pred):
          if pred_label != gt_label:
              count += 1
              print('Image file {} misclassified as {}. (groundtruth label {})'.format(image_name, pred_label, gt_label))
    print('\n\nTotal misclassified images {}/{}'.format(str(count), len(file_name)))
    print("==============================================================")
    y_true = output_gt
    y_pred = output_pred
    conf_mat_output = confusion_matrix(y_true, y_pred, labels=np.unique(output_gt))
    output_acc = accuracy_score(y_true, y_pred)
    output_precision = precision_score(y_true, y_pred, average='micro', labels=np.unique(output_gt))
    output_recall = recall_score(y_true, y_pred, average='micro', labels=np.unique(output_gt))
    print("\n\n\n==================== Evaluation Result Summary ====================")
    print("Accuracy score : {}".format(output_acc))
    print("Precision score : {}".format(output_precision))
    print("Recall score: {}".format(output_recall))
    # print("F1 score: {}".format(output_f1))
    print(classification_report(y_true, y_pred, digits=7, labels=np.unique(output_gt)))
    print("===================================================================")
    print('Compression Output & Stats Follow')



if __name__ == '__main__':
  tf.app.run()
