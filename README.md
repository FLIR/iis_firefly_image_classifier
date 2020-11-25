# Image classification using Tensorflow-Slim library
<!-- # Image classification model library -->

This repository is based on the [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim).
[TF-slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)
is a lightweight high-level API of TensorFlow (`tensorflow.contrib.slim`)
for defining, training and evaluating complex
models. This repository contains
code for training and evaluating several widely used Convolutional Neural
Network (CNN) models, which can easily be use to train an image classification model on your own datasets.
It also contains scripts that will allow
you to convert your image dataset to TensorFlow's native TFRecord format, train models from scratch or fine-tune them from pre-trained network
weights (Transfer Learning). In addition, we've included a
TAN document (To be added),
which provides working examples of how to use this repository.

## Features
- **Functionality**:
    - **Train an image classifier**:
        - Train an image classification model that can be deployed on FLIR's FireFly-DL camera.
        - Choose from several backbone model architectures based on your deployment requirements (inference time, model memory footprint, and accuracy).
        - Transfer-learn from ImageNet dataset, and fine-tune the model on your own image classification dataset.
        - Generate a training event-log, which can be visualized using TensorBoard.
    - **Evaluate your image classifier**:
        - Evaluate your classification model while training.
        - Generate an evaluation event-log, which can be visualized using TensorBoard.
    - **Test image classifier model**:
        - Evaluate your trained model on your test set, and generate the corresponding inference results (prediction.csv).
    - **Convert to TFRecord format**
        - Converts your images to TFRecord format.
    - **Generate Frozen graph**
        - Generate a frozen graph model, and use [NeuroUtility](http://softwareservices.flir.com/Camera-Resources/Content/10-Front/Camera-Resources-FFY-DL.htm) to convert and deploy the trained model to FLIR's FireFly-DL camera.
    - **Hyperparameter Optimization**
        - Using [Guildai](https://guild.ai/)
- **Input**: Image (.jpeg and .png).
- **Output**: Frozen-model graph and trained weights (.pb).
- **OS**: Ubuntu 16.04, Windows 10 (Setup documentation to be added).
- **Hardware**: This script was tested on a GeForce GTX 1080 Nvidia GPU card.
- **Others**:
    - Enviroment setup using docker images.
    - This repository was tested on the following system setup:
      - Training on GPU/CPU: CUDA 10.0 (Nvidia GPU) cudnn 7.2. Alternatively, you can train a model using only CPU (see train_image_classifier.py script input arguments for more details).
      - Python 3.7 libraries: TensorFlow Version 1.13.2, Tensorboard 2.2.2, guild 0.7.0.post1.


## Latest Features
- **Image capture and labeling tool**
    - To be added.
- **Supported model architectures**
    - Mobilenet_v1_1.0_224
    - Mobilenet_v1_0.75_224
    - Mobilenet_v1_0.5_224
    - Mobilenet_v1_0.25_224
    - Inception_v1

## Results
### Runtime Analysis
Inference time comparison between the five model architectures that are compatible on FireFly-DL:

Model | ImageNet Accuracy | FireFly-DL Inference Time (ms)|
:----:|:-----------------:|:-----------------------------:|
[Inception_v1_224](https://arxiv.org/abs/1409.4842v1)|69.8|222|
[MobileNet_v1_1.0_224](https://arxiv.org/pdf/1704.04861.pdf)|70.9|78|
[MobileNet_v1_0.75_224](https://arxiv.org/pdf/1704.04861.pdf)|68.4|55|
[MobileNet_v1_0.50_224](https://arxiv.org/pdf/1704.04861.pdf)|63.7|36|
[MobileNet_v1_0.25_224](https://arxiv.org/pdf/1704.04861.pdf)|50.6|22|

- Input image size (224x224 pixels)

The reported accuracy of the pre-trained models are based on a subset of the ImageNet classification challenge, which contain a training set of 1.2 million images, and 1000 categories. You can find more information [here](http://www.image-net.org/challenges/LSVRC/2012/).
In addition, we present the average inference time on FireFly-DL for each model. You can find more information [here](https://www.flir.ca/products/firefly-dl/).



## Contents
1. [Features](#features)
2. [Latest Features](#latest-features)
3. [Results](#results)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Output](#output)
7. [Preparing the datasets](#preparing-the-datasets)
8. [Foot Dataset](#foot-dataset)
9. [Send Us Failure Cases and Feedback!](#send-us-failure-cases-and-feedback)
10. [Citation](#citation)
11. [License](#license)
12. [References](#References)

## Contacts
This repository is maintained by FLIR-IIS R&D team.
* Ahmed Sigiuk, Ahmed.Sigiuk@flir.com
* Di Xu, Di.Xu@flir.com
* Douglas Chong, Douglas.Chong@flir.com


<!-- ## Table of contents
<a href="#Install">Installation and setup</a><br>
<a href='#Data'>Preparing the datasets</a><br>
<a href='#Pretrained'>Using pre-trained models</a><br>
<a href='#Training'>Training from scratch</a><br>
<a href='#Tuning'>Fine tuning to a new task</a><br>
<a href='#Eval'>Evaluating performance</a><br>
<a href='#Export'>Exporting Inference Graph</a><br>
<a href='#Troubleshooting'>Troubleshooting and Current Known Issues</a><br> -->

## Installation
<a id='Install'></a>
In this section, we describe the steps required to setup the training environment in preperation for running the script provided in this repository.

We provide two options for instaling Tensorflow on your system.

<a href="#Host">Setup environment on native host machine </a><br>

<a href="#Docker">Setup environment using Docker</a><br>

### Setup environment on native host machine
<a id='Host'></a>
This section assumes that the following requirements are satisfied:
- Ubuntu 16.04 or later releases (Also tested on Windows 10).
- Cuda 10.0 and cudnn 7.
- Python 3.5 or later release.
- Nvidia GTX GPU card.

#### Insall Tensorflow using python pip
You can use `pip` python package manager to install Tensorflow library on your host machine.

```bash
# Install tensorflow.
# If you have GPU,
pip install --user tensorflow-gpu==1.13.2  
# or, for training on CPU
pip install --user tensorflow==1.13.2
# Install scikit-learn and guildai  
pip install pip install --user guildai scikit-learn
```

### Setup environment using Docker
<a id='Docker'></a>
This section assumes that the following requirements are satisfied:
- Ubuntu 16.04 or later releases (Also tested on Windows 10).
- Cuda 10.0/Cudnn 7 or later releases.
- Docker-ce 19.03.12 or later releases.
- Nvidia GTX GPU card.

#### Pull and run TensorFlow docker environment

```bash
# Pull and run tensorflow runtime docker environment.
docker run --gpus all --rm -it --name tensorflow-env-1  -e DISPLAY=${DISPLAY}  --net=host  --privileged --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864  -v /dev:/dev -v ~:/home/docker/ asigiuk/tensorflow_runtime:latest

```

Note: After running the docker environment you can terminate the environment by typing `exit` in your terminal.
## Quick Start

After setting up the training environment on you machine.

### Clone this repository.

```bash
git clone ....

cd ...

```
### Collect and label image datasets
Collect and label some training images. Refer to the collect image dataset for more details regarding supported image formats (.jpg and .png) and the expected directory structure.
Important Note: If you are using docker environment, make sure that the docker container has access to your image directory. Optionally, you can copy your images (folders) inside this repository folder.

### Start training your model
```bash
python train_image_classifier.py \
        --project_name=<Select a project name> \
        --dataset_name=<Select a dataset name> \
        --image_dir=</path/to/image_directory>
```

## Outputs
A trained model (a completed training experiment) is identified based on the following input parameters: project name, experiment name and the dataset name used. These are all input arguments to the training scripts. In addition, the corresponding generated files are saved under the following directory structure.
- **Project directory**: A project name given by `--project_name` and located by default under `./project_dir/<project name>`.
  - Required input arguments for the following scripts (train/eval/test_image_classifier.py convert.py).
  - **dataset directory**: A dataset name given by `--dataset_name` and located by default under `./project_dir/<project name>/datasets/<dataset name>`
    - Dataset TFRecord shreads (.tfrecord).
    - Dataset settings (dataset_config.json).
    - Dataset label file (label.txt).
  - **experiments directory**: An experiment name given by `--experiment_name` and located by default under `./project_dir/<project name>/experiments/<experiment name>`.
    - Trained checkpoint files (train/ .ckpt).
    - Trained frozen graph file (train/frozen_ .pb).
    - Train/Eval Tensorboard event file logs (train/events. , eval/events. )
    - Train experiment setting file (train/experiment_setting.txt)
    - Test prediction results (test/predictions.csv)

## Preparing the datasets
<!-- <a id='Data'></a>

As part of this library, we've included scripts to download several popular
image datasets (listed below) and convert them to slim format.

Dataset | Training Set Size | Testing Set Size | Number of Classes | Comments
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
Flowers|2500 | 2500 | 5 | Various sizes (source: Flickr)
[Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) | 60k| 10k | 10 |32x32 color
[MNIST](http://yann.lecun.com/exdb/mnist/)| 60k | 10k | 10 | 28x28 gray
[ImageNet](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 | Various sizes
VisualWakeWords|82783 | 40504 | 2 | Various sizes (source: MS COCO) -->

<!-- ## Downloading and converting flower dataset to TFRecord format

For each dataset, we'll need to download the raw data and convert it to
TensorFlow's native
[TFRecord](https://www.tensorflow.org/versions/r0.10/api_docs/python/python_io.html#tfrecords-format-details)
format. Each TFRecord contains a
[TF-Example](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/core/example/example.proto)
protocol buffer. Below we demonstrate how to do this for the Flowers dataset.

```shell
$ DATA_DIR=/tmp/data/flowers
$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```

When the script finishes you will find several TFRecord files created:

```shell
$ ls ${DATA_DIR}
flowers_train-00000-of-00005.tfrecord
...
flowers_train-00004-of-00005.tfrecord
flowers_validation-00000-of-00005.tfrecord
...
flowers_validation-00004-of-00005.tfrecord
labels.txt
```

These represent the training and validation data, sharded over 5 files each.
You will also find the `$DATA_DIR/labels.txt` file which contains the mapping
from integer labels to class names.

You can use the same script to create the mnist, cifar10 and visualwakewords
datasets. However, for ImageNet, you have to follow the instructions
[here](https://github.com/tensorflow/models/blob/master/research/inception/README.md#getting-started).
Note that you first have to sign up for an account at image-net.org. Also, the
download can take several hours, and could use up to 500GB. -->

## Collect and convert your own dataset

First, you must collect and label a sample of images (at least 100 images per class) that you would like to train the model to classify. Then convert the label images to TFRecord format.

### Collect training images.

For each dataset, we'll need to label the dataset into classes by placing the raw image file into directory with matching class name. Please note the following;

* The `train_image_classifier.py` script only supports the following image formats 'jpg', 'jpeg', 'png', and 'bmp'.
* Label the images into classes using the parent directory name.
* Each image most be save into only one folder (representing the class)
* The ground-truth label for each image is taken from the parent directory name.

The diagram below shows the expected folder structure.

```

    dataset-name
    |
    |-- class_1
    |   |
    |   |--image_1.jpg
    |   |--image_2.jpg
    |           :
    |           :
    |-- class_2
    |   |
    |   |--image_1.jpg
    |   |--image_2.jpg
    |           :
    |           :
    |-- class_3
    |   |
    |   |--image_1.jpg
    |   |--image_2.jpg
    |           :
                :
```

### Convert custom dataset to TFRecord format

For each dataset, we'll need to label the dataset into classes by placing the raw image file into directory with matching class name.
We provide two options for converting your image dataset to TFRecord format.

<a href="#train-script">Using the training script (recommended)</a><br>

<a href="#convert-script">Using the convert to TFRecord script</a><br>
Below we demonstrate how to do this for the blocks dataset.

Using the training script
<a id='convert-script'></a>

You can convert your images to TFRecord format using the `train_image_classifier.py` by specify the image directory with `--image_dir` flag and selecting a dataset name for it with `--dataset_name` flag.
```bash
python train_image_classifier.py \
        --project_name=<Select a project name> \
        --dataset_name=<Select a dataset name> \
        --image_dir=</path/to/image_directory>
```
Note: That if you rerun the training command again on the same dataset. You can omit the `--image_dir` and use the same dataset name. This will skip the image conversion and use the saved TFRecord dataset.


Using the convert to TFRecord script
<a id='convert-script'></a>

Optionally you can convert your image dataset using the `convert_images_to_tfrecord.py` script.

```shell
$ python convert_images_to_tfrecord.py \
        --project_name=<Select a project name> \
        --dataset_name=<Select a dataset name> \
        --image_dir=</path/to/image_directory>
```

You access the convert dataset files under the following directory `./project_dir/<project name>/datasets/<dataset name>`. Bellow is an example of the files gerated.

```shell
$ ls ${DATA_DIR}
dataset_name_train-00000-of-00005.tfrecord
...
dataset_name_train-00004-of-00005.tfrecord
dataset_name_validation-00000-of-00005.tfrecord
...
dataset_name_validation-00004-of-00005.tfrecord
dataset_name-00000-of-00005.tfrecord
...
dataset_name-00004-of-00005.tfrecord
labels.txt
dataset_config.json
```

These represent the training, validation and test data, sharded over five files in this example.
You will also find the `labels.txt` file which contains the mapping
from integer labels to class names. In addition, you will find the  `dataset_config.json`file which stores some of the dataset attributes. An example `dataset_config.json` file is shown below:

```shell
{"dataset_name": <dataset name>,
"dataset_dir": "./path/to/datasets/<dataset name>",
"class_names": [class_1, class_2],
"number_of_classes": 2,
"dataset_split":
        {"train": <number of training samples>,
        "train_per_class":
                  {class_1: <number of training samples in class 1>,
                  class_2: <number of training samples in class 2>},
        "test": <number of test samples>,
        "test_per_class":
                  {class_1: <number of test samples in class 1>,
                  class_2: <number of test samples in class 2>},
        "validation": <number of validation samples>,
        "validation_per_class":
                  {class_1: <number of validation samples in class 1>,
                  class_2: <number of validation samples in class 2>}
        }
}
```

## Training, Evaluation, and Testing your classification model


### Pre-trained Models
<a id='Pretrained'></a>

Neural nets work best when they have many parameters, making them powerful
function approximators.
However, this  means they must be trained on very large datasets. Because
training models from scratch can be a very computationally intensive process
requiring days or even weeks, we provide various pre-trained models,
as listed below. These CNNs have been trained on the
[ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/)
image classification dataset.

In the table below, we list each model, the corresponding
TensorFlow model file, the link to the model checkpoint, and the top 1 and top 5
accuracy (on the imagenet test set).
Note that the VGG and ResNet V1 parameters have been converted from their original
caffe formats
([here](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014)
and
[here](https://github.com/KaimingHe/deep-residual-networks)),
whereas the Inception and ResNet V2 parameters have been trained internally at
Google. Also be aware that these accuracies were computed by evaluating using a
single image crop. Some academic papers report higher accuracy by using multiple
crops at multiple scales.

Model | TF-Slim File | Checkpoint | Top-1 Accuracy| Top-5 Accuracy |
:----:|:------------:|:----------:|:-------:|:--------:|
[MobileNet_v1_1.0_224](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)|[mobilenet_v1_1.0_224.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)|70.9|89.9|
[MobileNet_v1_0.50_160](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)|[mobilenet_v1_0.50_160.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz)|59.1|81.9|
[MobileNet_v1_0.25_128](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)|[mobilenet_v1_0.25_128.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz)|41.5|66.3|



All 16 float MobileNet V1 models reported in the [MobileNet Paper](https://arxiv.org/abs/1704.04861) and all
16 quantized [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) compatible MobileNet V1 models can be found
[here](https://github.com/tensorflow/models/tree/r1.13/research/slim/nets/mobilenet_v1.md).


Here is an example of how to download the MobileNet V1 checkpoint:

```shell
$ CHECKPOINT_DIR=./checkpoints
$ mkdir ${CHECKPOINT_DIR}
$ cd ${CHECKPOINT_DIR}
$ wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
$ tar -xvf mobilenet_v1_1.0_224.tgz
$ rm mobilenet_v1_1.0_224.tgz
```


### Training a customized model for your own application

Training a model from scratch is no easy job. It is time consuming and requires extensive deep learning expertise. Rather than training from scratch, we'll often want to start from a pre-trained model and fine-tune it to create customized models for your new applications.

### Fine-tuning a model from an existing checkpoint
<a id='Tuning'></a>



To indicate a checkpoint from which to fine-tune, we'll call training with
the `--checkpoint_path` flag and assign it an absolute path to a checkpoint
file.

When fine-tuning a model, we need to be careful about restoring checkpoint
weights. In particular, when we fine-tune a model on a new task with a different
number of output labels, we wont be able restore the final logits (classifier)
layer. For this, we'll use the `--checkpoint_exclude_scopes` flag. This flag
hinders certain variables from being loaded. When fine-tuning on a
classification task using a different number of classes than the trained model,
the new model will have a final 'logits' layer whose dimensions differ from the
pre-trained model. For example, if fine-tuning an ImageNet-trained model on
Flowers, the pre-trained logits layer will have dimensions `[2048 x 1001]` but
our new logits layer will have dimensions `[2048 x 5]`. Consequently, this
flag indicates to TF-Slim to avoid loading these weights from the checkpoint.

Keep in mind that warm-starting from a checkpoint affects the model's weights
only during the initialization of the model. Once a model has started training,
a new checkpoint will be created in `${TRAIN_DIR}`. If the fine-tuning
training is stopped and restarted, this new checkpoint will be the one from which weights are restored and not the `${checkpoint_path}$`. Consequently, the flags `--checkpoint_path` and `--checkpoint_exclude_scopes` are only used during the `0-`th global step (model initialization).

Typically for fine-tuning you only wants to train a sub-set of layers. The flag `--trainable_scopes` allows you to specify which subset of layers should be trained, and the rest would remain frozen. Where the `--trainable_scopes`  flag expects a string of comma separated variable names with no spaces. In addition you can specify all the trainable variables in a specific layer by only specifying the common name (name_scope) of the variables in that layer, as defined in the graph. For example, if you only want to train all the variables in the logits, Conv2d_13, and Conv2d_12 layers. You would set the `--trainable_scopes` argument as such


```shell
--trainable_scopes=MobilenetV1/Logits,MobilenetV1/Conv2d_13,MobilenetV1/Conv2d_12
```
The training script (`train_image_classifier.py`) will print out a list of all the trainable variable that are defined in the selected model graph `--model_name=mobilenet_v1` . Note that if the specified variable name(s) do not match the name(s) defined in the select model graph, that variable will be ignored with no error messages. Hence it is important to check that the provided variable names are correct, and that all the desired trainable variable have be selected. For the above example where we wanted to train the last three layers of the model (logits, Conv2d_13, and Conv2d_12 layers) you should get the follow trainable variable list as a screen print out when you run the (`train_image_classifier.py`) script.

<!-- ```shell
######## List of all Trainable Variables ###########
 [<tf.Variable 'MobilenetV1/Logits/Conv2d_1c_1x1/weights:0' shape=(1, 1, 256, 16) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Logits/Conv2d_1c_1x1/biases:0' shape=(16,) dtype=float32_ref>,
<tf.Variable 'MobilenetV1/Conv2d_13_depthwise/depthwise_weights:0' shape=(3, 3, 256, 1) dtype=float32_ref>, <tf.Variable MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma:0' shape=(256,) dtype=float32_ref>,
<tf.Variable 'MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Conv2d_13_pointwise/weights:0' shape=(1, 1, 256, 256) dtype=float32_ref>,
<tf.Variable 'MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta:0'
shape=(256,) dtype=float32_ref>,
<tf.Variable 'MobilenetV1/Conv2d_12_depthwise/depthwise_weights:0' shape=(3, 3, 128, 1) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma:0' shape=(128,) dtype=float32_ref>,
<tf.Variable 'MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Conv2d_12_pointwise/weights:0' shape=(1, 1, 128, 256) dtype=float32_ref>,
<tf.Variable 'MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta:0' shape=(256,) dtype=float32_ref>]
``` -->

Below we give an example of mobilenet_v1 that was trained on ImageNet with 1000 class labels, however, now we set `--datasetdir=${DATASET_DIR}`  to point to our custom dataset. Since the dataset is quite small we will only train the last two layers.


```shell
$ CHECKPOINT_PATH=./checkpoints/mobilenet_v1_0.25_224/mobilenet_v1_0.25_224.ckpt

$ python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=blocks \
    --batch_size=64 \
    --dataset_split_name=train \
    --model_name=mobilenet_v1_025 \
    --preprocessing_name=mobilenet_v1 \
    --train_image_size=224 \
    --max_number_of_steps=1000 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=MobilenetV1/Logits,MobilenetV1/AuxLogits \
    --trainable_scopes=MobilenetV1/Logits,MobilenetV1/AuxLogits \
    --clone_on_cpu=True
```

For training on cpu (with tensorflow package, instead of tensorflow-gpu), set flag `--clone_on_cpu` to `True`. For training on gpu, this flag can be ignored or set to `False`.

We suggest to use a different directory `TRAIN_DIR` is suggested to be in a different directory each time  


### Evaluating performance of a model while training
<a id='Eval'></a>

To evaluate the performance of a model (whether pretrained or your own),
you can use the eval_image_classifier.py script, as shown below.

The script should be run while training and the `--eval_dir` flag should point to the same directory as your training script `--train_dir` flag. In addition, the script will create a new directory inside the `--eval_dir` directory. The assigned name to this new directory is taken from the `--dataset_split_name` flag.

By defualt the `--checkpoint_path` flag will point to the following directory `eval_dir/dataset_split_name` . Optinally, you call also specify the `--checkpoint_path` flag,  which should point to the directory where the training job checkpoints are stored.

```shell
$ python eval_image_classifier.py \
    --alsologtostderr \
    --eval_dir=${TRAIN_DIR} \
    --dataset_dir=${TFRECORD_OUTPUT_DIR} \
    --dataset_name=blocks \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1 \
    --preprocessing_name=mobilenet_v1 \
    --eval_image_size=224
```

See the [evaluation module example](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/slim#evaluation-loop)
for an example of how to evaluate a model at multiple checkpoints during or after the training.

### Test performance of a model


### TensorBoard

To visualize the losses and other metrics during training, you can use
[TensorBoard](https://github.com/tensorflow/tensorboard)
by running the command below.

```shell
tensorboard --logdir=./project_dir/<project_name>/experiments/<experiment name> --port 6006
```

Once TensorBoard is running, navigate your web browser to http://localhost:6006

### Exporting the Inference Graph
<a id='Export'></a>

Saves out a GraphDef containing the architecture of the model.

To use it with a model name defined by slim, run:

```shell
$ python export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v1
  --dataset_dir=${TFRECORD_OUTPUT_DIR}  
  --output_file=${TRAIN_DIR}/inference_graph_mobilenet_v1.pb --dataset_name=blocks

```

### Freezing the exported Graph
If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

```shell
python freeze_graph.py \
  --input_graph=${TRAIN_DIR}/inference_graph_mobilenet_v1.pb  \
  --input_checkpoint=${TRAIN_DIR}/model.ckpt-1000 \
  --input_binary=true --output_graph=${TRAIN_DIR}/frozen_mobilenet_v1.pb \
  --output_node_names=MobilenetV1/Predictions/Reshape_1
```
[Note: The bazel commands were replaced with a python file. Same arguments were used.]
The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:

## Guildai for Hyperparameter search



# Troubleshooting and Current Known Issues
<a id='Troubleshooting'></a>

#### Known issues that need to be adressed:

* Tensorboard stops updating after `eval_image_classifier.py` script while training script is running.

* Add to readme;
    * Section on custom preprocessing scripts
    * Section on custom model scripts.


#### The model runs out of CPU memory.

See
[Model Runs out of CPU memory](https://github.com/tensorflow/models/tree/r1.13/research/inception#the-model-runs-out-of-cpu-memory).

#### The model runs out of GPU memory.

See
[Adjusting Memory Demands](https://github.com/tensorflow/models/tree/r1.13/research/inception#adjusting-memory-demands).

#### The model training results in NaN's.

See
[Model Resulting in NaNs](https://github.com/tensorflow/models/tree/r1.13/research/inception#the-model-training-results-in-nans).

#### The ResNet and VGG Models have 1000 classes but the ImageNet dataset has 1001

The ImageNet dataset provided has an empty background class which can be used
to fine-tune the model to other tasks. If you try training or fine-tuning the
VGG or ResNet models using the ImageNet dataset, you might encounter the
following error:

```bash
InvalidArgumentError: Assign requires shapes of both tensors to match. lhs shape= [1001] rhs shape= [1000]
```
This is due to the fact that the VGG and ResNet V1 final layers have only 1000
outputs rather than 1001.

To fix this issue, you can set the `--labels_offset=1` flag. This results in
the ImageNet labels being shifted down by one:


#### I wish to train a model with a different image size.

The preprocessing functions all take `height` and `width` as parameters. You
can change the default values using the following snippet:

```python
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name,
    height=MY_NEW_HEIGHT,
    width=MY_NEW_WIDTH,
    is_training=True)
```

#### What hardware specification are these hyper-parameters targeted for?

See
[Hardware Specifications](https://github.com/tensorflow/models/tree/master/research/inception#what-hardware-specification-are-these-hyper-parameters-targeted-for).

## References
"TensorFlow-Slim image classification model library"
N. Silberman and S. Guadarrama, 2016.
https://github.com/tensorflow/models/tree/master/research/slim
