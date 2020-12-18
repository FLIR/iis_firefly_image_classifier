#Compressing a "flowers" MobileNetV1 Classifier using Qualcomm's AIMET

#Imports
from __future__ import print_function
import sys
sys.path.append('/home/research/Public/RobertB/aimet_tests/aimet/TrainingExtensions/common/src/python')
sys.path.append('/home/research/Public/RobertB/aimet_tests/aimet/TrainingExtensions/tensorflow/src/python')

import warnings
warnings.filterwarnings("ignore")

from decimal import Decimal

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications import MobileNet

# Compression-related imports
from aimet_common.defs import GreedySelectionParameters
from aimet_common.defs import CostMetric, CompressionScheme
from aimet_tensorflow.defs import SpatialSvdParameters, ChannelPruningParameters, ModuleCompRatioPair
from aimet_tensorflow.compress import ModelCompressor

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

import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

slim = tf.contrib.slim

#Using the same flags as the test image classifier script
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

# check required input arguments
if not FLAGS.project_name:
    raise ValueError('You must supply a project name with --project_name')
if not FLAGS.dataset_name:
    raise ValueError('You must supply a dataset name with --dataset_name')

# set and check project_dir and experiment_dir.
project_dir = os.path.join(FLAGS.project_dir, FLAGS.project_name)
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

#In a departure from the test_image_classifier script, we perform all of the setup in a designated function, which returns items to be used later
def setup():
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
        'inception_v1':'InceptionV1'}

    #####################################
    # Select the preprocessing function #
    #####################################

    model_variables = model_name_to_variables.get(FLAGS.model_name)
    if model_variables is None:
        tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
        sys.exit(-1)


    image_string = tf.placeholder(name='input', dtype=tf.string)

    image = tf.image.decode_png(image_string, channels=3)

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

    sess = tf.Session()# as sess:
    
    return init_fn,label_to_class_dict,class_to_label_dict,fls,sess

#Here we define a function to generate an initialization function (to initialize all of the variables) for a given tensorflow session
#We have to do this symbolically, because during compression, the architecture will be different every time; so we can't hard-code an init_fn
def _get_init_fn(ckpt_path):
    mnv1_checkpoint_path = ckpt_path
    #checks how the checkpoint path was passed in
    if tf.gfile.IsDirectory(mnv1_checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(mnv1_checkpoint_path)
    else:
        checkpoint_path = mnv1_checkpoint_path
    #Makes a list of all of the model's variables to restore them
    variables_to_restore = []
    for var in slim.get_model_variables():
        variables_to_restore.append(var)
    #Returns a callable function to initialize all the variables
    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=False)

#This is the tough part of using AIMET with tensorflow. AIMET expects a function, evaluate_model, which will return the performance of *ANY* session it passes in.
#On the fly, we have to do the following:
#1) Temporarily save the session to a checkpoint file, for reference
#2) Use the session's graph to create the initialization function, with _get_init_fn() from earlier
#3) Begin a NEW session, that uses the graph from the session that's passed in.
#4) Initialize all of the session's variables
#5) Load the weights from the saved checkpoint file
#6) Finally, iterate through the test data to perform inference, and return the arbitrary session's performance on the test data.
def evaluate_model(gen_sess, eval_iterations, use_cuda):
    #Attempt to save the gen_sess
    #Checkpoint path - where the generic session will be temporarily stored
    ckpt_path = '/home/research/Public/RobertB/aimet_tests/aimet/comp-tests/iis_firefly_image_classifier/temp_ckpts/mid_comp.ckpt'
    #Save the session at the designated location
    tf_saver.save(gen_sess,ckpt_path)
    #Now we define the initialization function with the generic session's graph as default
    with gen_sess.graph.as_default():
        #specific_init_fn will be used below to initialize all of the session's variables
        specific_init_fn = _get_init_fn(ckpt_path)
    #Declaring a few variables that will be used during inference.
    counter = 0
    output_pred = list()
    output_gt = list()
    file_name = list()
    #Now we create a new tensorflow session that uses the generic session's graph, and call it "specific_sess"
    with tf.Session(graph=gen_sess.graph) as specific_sess:
        #Initialize all the variables using a tensorflow built-in function. This does not load the values, just makes it so that the variables exist.
        specific_sess.run(tf.initialize_all_variables())
        #We use the saver to restore the values of the specific_sess from the checkpoint we made out of gen_sess earlier
        tf_saver.restore(specific_sess,ckpt_path)
        #We call the specific_init_fn to initialize the quantities for the model
        specific_init_fn(specific_sess)
        #This is the same as in the testing script. We iterate through the test data files and have the session evaluate them one at a time.
        for fl in fls:
       	    image_name = None
       	    example = tf.train.Example()
       	    example.ParseFromString(fl)
       	    # Note: The key of example.features.feature depends on how you generate tfrecord
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
            #MODIFICATION from the test script. We don't necessarily know where the input and output operations are, so we access them symbolically.
            #The output op will always be MobilenetV1/Predictions/Softmax, and the input op will always be input.
            #From there, you just have to get the tensors associated with those operations, using .outputs[-1] or [0], shouldn't make a difference.
            probs = specific_sess.run([op for op in specific_sess.graph.get_operations() if op.name=='MobilenetV1/Predictions/Softmax'][0].outputs[-1], feed_dict={[op for op in specific_sess.graph.get_operations() if op.name=='input'][0].outputs[0]:img})
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
    output_acc = accuracy_score(output_gt, output_pred)
    #We can now return the accuracy score for the arbitrary session on the test data.
    print('\nthe output accuracy is {}'.format(output_acc))
    return float(output_acc)

#If evaluate_model breaks, you can use this simple function to ensure that everything else is working.
#We just return a float.
#def evaluate_model(sess: tf.Session, eval_iterations: int, use_cuda: bool) -> float:
#    return 0.5

#Function to perform spatial SVD (singular value decomposition) automatically using AIMET.
#Largely constructed using the examples on AIMET's API page for tensorflow
def spatial_svd_auto_mode():
    #You can choose to ignore, say, the first convolution layer in compression
    modules_to_ignore = None
    #This allows you to specify how you want AIMET to compress your model.
    #Here we're using 0.8, with three compression ratio candidates (33%,66%,and~100%) for each layer
    #At the end it will select a combination of individual compressions that reaches your target_comp_ratio, here we're aiming for 80% (0.8)
    greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                              num_comp_ratio_candidates=3,
                                              use_monotonic_fit=True,
                                              saved_eval_scores_dict=None)

    auto_params = SpatialSvdParameters.AutoModeParams(greedy_select_params=greedy_params,
                                                      modules_to_ignore=modules_to_ignore)
    #Has to match the input operation and output operations of your ORIGINAL UNCOMPRESSED MODEL
    params = SpatialSvdParameters(input_op_names=['input'], output_op_names=['MobilenetV1/Predictions/Reshape_1'],
                                  mode=SpatialSvdParameters.Mode.auto, params=auto_params, multiplicity=8)
    #Your model's input shape
    input_shape = (1, 3, 224, 224)
    # Single call to compress the model
    compr_model_sess, stats = ModelCompressor.compress_model(sess=sess,
                                                             working_dir=str('./'),
                                                             eval_callback=evaluate_model,
                                                             eval_iterations=10,
                                                             input_shape=input_shape,
                                                             compress_scheme=CompressionScheme.spatial_svd,
                                                             cost_metric=CostMetric.mac,
                                                             parameters=params,
                                                             trainer=None)
    #the compressed session is contained in compr_model_sess, so we'll return that.
    print(stats)    # Stats object can be pretty-printed easily
    return compr_model_sess

#We'll do setup() and return the items that are used elsewhere in the script so that they're globally accessible by other functions we might call.
init_fn,label_to_class_dict,class_to_label_dict,fls,sess = setup()
#We also have to initialize our uncompressed model session in order to compress it.
init_fn(sess)

#We make a new saver so that we can save the compressed model when we're done.
saver = tf.train.Saver()
tf_saver = tf.train.Saver(name="saver")
#We call the compression function here, and save its output to the name compressed_sess
compressed_sess = spatial_svd_auto_mode()
export_dir = './savedmodeltest'

from tensorflow.python.tools import freeze_graph

directory='./savedmodeltest'
filename='flower_comp_test_0'

#Define a saving function
def save(directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename + '.ckpt')
    saver.save(compressed_sess, filepath)
    return filepath

#def save_as_pb(self, directory, filename):

if not os.path.exists(directory):
    os.makedirs(directory)

# Save check point for graph frozen later
ckpt_filepath = save(directory=directory, filename=filename)
pbtxt_filename = 'flowers_comp_0_graph.pb'
pbtxt_filepath = os.path.join(directory, pbtxt_filename)
pb_filepath = os.path.join(directory, filename + '.pb')

#Not in use for now, but this code might be used to save as a .pb instead of a .ckpt.
# Freeze graph
# Method 1
#freeze_graph.freeze_graph(input_graph=compressed_sess.graph.as_graph_def(),
#                          input_saver='',
#                          input_binary=False,
#                          input_checkpoint=ckpt_filepath,
#                          output_node_names='MobilenetV1/Predictions/Reshape_1',
#                          restore_op_name='save/restore_all',
#                          filename_tensor_name='save/Const:0',
#                          output_graph='test.pb',#pb_filepath,
#                          clear_devices=True,
#                          initializer_nodes='')
