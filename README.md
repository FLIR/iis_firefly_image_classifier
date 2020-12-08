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
TAN document (Link to be added),
which provides a working examples of how to use this repository.

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
- **OS**: Ubuntu 16.04 and later releases.
- **Hardware**: This script was tested on a GeForce GTX 1080 Nvidia GPU card.
- **Others**:
    - Enviroment setup using docker images.

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

Model | Flowers Classifier Accuracy| FireFly-DL Inference Time (ms) |Max Number of Steps | Train Time (min) |
:----:|:--------------------------:|:------------------------------:|-------------------:|:----------------:
[Inception_v1_224](https://arxiv.org/abs/1409.4842v1)|87.8|222|30k|45
[MobileNet_v1_1.0_224](https://arxiv.org/pdf/1704.04861.pdf)|91.9|78|30k|65
[MobileNet_v1_0.75_224](https://arxiv.org/pdf/1704.04861.pdf)|89.2|55|30k|55
[MobileNet_v1_0.50_224](https://arxiv.org/pdf/1704.04861.pdf)|89.2|36|30k|41
[MobileNet_v1_0.25_224](https://arxiv.org/pdf/1704.04861.pdf)|89.2|22|30k|32

- Input image size (224x224 pixels)

The reported accuracy of the pre-trained models are based on a subset of the Oxford Flowers dataset, which contain a training set of 3000 images, and 5 categories. You can find more information [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/).
We report the average inference time on FireFly-DL (Mono) for each model (more information [here](https://www.flir.ca/products/firefly-dl/)). We also report the maximum number of training steps and training time using Nvidia's GPU (GeForce GTX 1080) card.

## Contents
1. [Features](#features)
2. [Latest Features](#latest-features)
3. [Results](#results)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Output](#output)
7. [Preparing the datasets](#preparing-the-datasets)
8. [Training, Evaluation, and Testing your Classification Model](#training-evaluation-and-testing-your-classification-model)
10. [Troubleshooting and Current Known Issues](#troubleshooting-and-current-known-issues)
11. [Send Us Failure Cases and Feedback!](#send-us-failure-cases-and-feedback)
12. [Contacts](#citation)
13. [License](#license)
14. [References](#References)

## Installation
In this section, we describe the steps required to setup the training environment in preperation for running the script provided in this repository.

We provide two options for setting up your TensorFlow environment.

1. [Setup environment using Pip](#setup-environment-using-pip)
2. [Setup environment using Docker](#setup-environment-using-docker)

### Setup environment using pip
The repository was test on the following system setup:

- Ubuntu 16.04 or later releases.
- Cuda 10.0 and cudnn 7.
- Python 3.5 or 3.7 release.
- Nvidia GTX GPU card.

#### Insall TensorFlow using python Pip
You can use `pip` python package manager to install TensorFlow library on your host machine.

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
- Ubuntu 16.04 or later releases.
- Cuda 10.0/Cudnn 7 or later releases.
- Docker-ce 19.03.12 or later releases.
- Nvidia GTX GPU card.

#### Pull and run TensorFlow docker environment

```bash
# Pull and run tensorflow runtime docker environment.
docker run --gpus all --rm -it --name tensorflow-env-1  -e DISPLAY=${DISPLAY}  --net=host  --privileged --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864  -v /dev:/dev -v path/to/host_target_directory:/home/docker/ asigiuk/tensorflow_runtime:latest

```
Important Notes:
1. Modify `-v path/to/host_target_directory:/home/docker` in the above command and replace `path/to/host_target_directory` with your host machine target directory path. This will mount your specified target host directory to the docker container home directory `/home/docker`.
2. The docker `-v` or `--volume` flag is used to mount a target directory in your host machine (i.e. `path/to/host_target_directory`) to the docker container directory (i.e. `/home/docker`) . You can find more information regarding the `docker run` command [here](https://docs.docker.com/engine/reference/commandline/run/#mount-volume--v---read-only).
3. You can terminate the docker environment by typing `exit` in your terminal.

## Quick Start

After setting up the training environment on you machine.

### Clone this repository.
```bash
git clone https://github.com/FLIR/iis_firefly_image_classifier.git

cd iss_firefly_image_classifier
```
### Collect and label image datasets
Collect and label some training images. Refer to the collect image dataset for more details regarding supported image formats and the expected directory structure.
Important Note:
- If you are using a docker container environment, verify that the docker container has access to your image directory. Optionally, you can copy your image directory under this repository folder.

### Start training your model
```bash
python train_image_classifier.py \
        --project_name=<Select a project name> \
        --dataset_name=<Select a dataset name> \
        --image_dir=</path/to/image_directory>
```

## Outputs
A trained model (a completed training experiment) is identified by three input arguments
  - project name `--project_name` .
  - dataset name `--dataset_name` .
  - experiment name `--experiment_name` (Optional, default `experiment_#` with ascending number is used ).

Below we summaries the input arguments and corresponding output directory structure, and file outputs.
- **Project directory**: A project name given by `--project_name` and located by default under `./project_dir/<project name>`.
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
In this section, we describe options for convert your image dataset to TensorFlow's native
[TFRecord](https://www.tensorflow.org/versions/r0.10/api_docs/python/python_io.html#tfrecords-format-details)
format. In addition, we provide the option to download and convert an example flowers dataset.

We provide two options for converting an image dataset to TFRecord format

1. [Convert images with convert script](#convert-images-with-convert-script)
2. [Convert images using training script](#convert-images-using-training-script)

#### Convert images with convert script

Convert your image dataset using the `convert_images_to_tfrecor.py` script.

```bash
$ python convert_images_to_tfrecord.py \
        --project_name=<Select a project name (Required)> \
        --dataset_name=<Select a dataset name (Required)> \
        --image_dir=</path/to/image_directory (Required)> \
        --train_percentage=<Percentage of images used for training (Optional)> \
        --validation_percentage=<Percentage of images used for training (Optional)> \
        --test_percentage=<Percentage of images used for training (Optional)> \
        --image_height=<Target image height (Optional)> \
        --image_width=<Target image width (Optional)>
```

**Download flowers image dataset and convert to TFRecord format (OPTIONAL)**

Below we demonstrate how we download the Flowers dataset and convert it to
TensorFlow's native
[TFRecord](https://www.tensorflow.org/versions/r0.10/api_docs/python/python_io.html#tfrecords-format-details)
format.

Dataset | Dataset Size | Number of Classes | Comments
:------:|:------------:|:-----------------:|:-----------:
Flowers| 3076 | 5 | Various sizes (source: Flickr)


```bash
$ python convert_images_to_tfrecord.py \
          --project_name=flowers_classifier \
          --dataset_name=flowers \
```

#### Convert images using training script
You can convert your images to TFRecord format using the `train_image_classifier.py` by specify the image directory with `--image_dir` flag and specifying a dataset name with `--dataset_name` flag.
```bash
python train_image_classifier.py \
        --project_name=<Select a project name> \
        --dataset_name=<Select a dataset name> \
        --image_dir=</path/to/image_directory>
```


Note: You can omit the `--image_dir` and use the same dataset name. This will skip the image conversion and use the saved TFRecord dataset.

#### Dataset folder:
You can access the convert TFRecord dataset files under the following directory: `./project_dir/<project name>/datasets/<dataset name>`.
```bash
$ ls ./project_dir/<project name>/datasets/<dataset name>
```

Below are some file examples.
```bash
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
List of dataset files:
  - Training, validation and test set shard files.
  - `labels.txt` file which contains the mapping from integer labels to class names.  
  - `dataset_config.json` file which stores some of the dataset attributes.

Example `dataset_config.json` file.      
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
This section covers how to run training, evaluation and test scripts.

### Pre-trained Models

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
Notes:
  - The Inception parameters have been trained by Google.
  - Also be aware that these accuracies were computed by evaluating using a single image crop. Some academic papers report higher accuracy by using multiple crops at multiple scales.

Model | TF-Slim File | Checkpoint | ImageNet Accuracy |
:----:|:------------:|:----------:|:-----------------:
[MobileNet_v1_1.0_224](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)|[mobilenet_v1_1.0_224.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)|70.9|
[MobileNet_v1_0.75_224](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)|[mobilenet_v1_0.75_224.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.75_224.tgz)|68.4|
[MobileNet_v1_0.50_224](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)|[mobilenet_v1_0.50_224.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_224.tgz)|63.7|
[MobileNet_v1_0.25_224](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)|[mobilenet_v1_0.25_224.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_224.tgz)|50.6|
[Inception_v1_224](https://arxiv.org/pdf/1409.4842v1.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)|[inception_v1_224.tgz](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)|69.8|


All 16 float MobileNet V1 models reported in the [MobileNet Paper](https://arxiv.org/abs/1704.04861) and all
16 quantized [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) compatible MobileNet V1 models can be found
[here](https://github.com/tensorflow/models/tree/r1.13/research/slim/nets/mobilenet_v1.md).

### Training a customized model for your own application
Training a model from scratch is no easy job. It is time consuming and requires extensive deep learning expertise. Rather than training from scratch, we'll often want to start from a pre-trained model and fine-tune it to create customized models for your new applications.
The training script has the following default input arguments set:
- If you do not specify a `--checkpoint_path` and you select with `--model_name` one of the four models (default `mobilenet_v1`) given in the above table the script will automatically download the corresponding ImageNet checkpoint from which to fine-tune from.
- If you do not specify a `--trainable_scopes` and `--checkpoint_exclude_scopes` the script will exclude the last layer `logits`
and train the `logits` and `BatchNorm` layers of the network.


### Fine-tuning a model from an existing checkpoint
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
a new checkpoint will be created in `./project_dir/<project name>/experiments/<experiment name>/train`.
If the fine-tuning training is stopped and restarted. Two options
- If you do not specify an existing experiment name with `--experiment_name`. The script will create a new experiment name and give it a name it is next in numarical ascending order (i.e. experiment_1, experiment_2 ...)
- If you specify an existing experiment name with `--experiment_name`. The script will use checkpoint found under `./project_dir/<project name>/experiments/<experiment name>/train`  from which weights are restored and not the `--checkpoint_path` or default ImageNet checkpoint.
Consequently, the flags `--checkpoint_path` and `--checkpoint_exclude_scopes` are only used during the `0-`th global step (model initialization).

Typically for fine-tuning you only wants to train a sub-set of layers. The flag `--trainable_scopes` allows you to specify which subset of layers should be trained, and the rest would remain frozen. Where the `--trainable_scopes`  flag expects a string of comma separated variable names with no spaces. In addition you can specify all the trainable variables in a specific layer by only specifying the common name (name_scope) of the variables in that layer, as defined in the graph. For example, if you only want to train all the variables in the logits, Conv2d_13, and Conv2d_12 layers. You would set the `--trainable_scopes` argument as such

```shell
--trainable_scopes=BatchNorm,MobilenetV1/Logits,MobilenetV1/Conv2d_13,MobilenetV1/Conv2d_12
```
The training script (`train_image_classifier.py`) will print out a list of all the trainable variable that are defined in the selected model graph `--model_name=mobilenet_v1` . Note that if the specified variable name(s) do not match the name(s) defined in the select model graph, that variable will be ignored with no error messages. Hence it is important to check that the provided variable names are correct, and that all the desired trainable variable have be selected. For the above example where we wanted to train the last three layers of the model (logits, Conv2d_13, and Conv2d_12 layers) you should get the follow trainable variable list as a screen print out when you run the (`train_image_classifier.py`) script.

```bash
[<tf.Variable 'MobilenetV1/Logits/Conv2d_1c_1x1/weights:0' shape=(1, 1, 256, 16) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Logits/Conv2d_1c_1x1/biases:0' shape=(16,) dtype=float32_ref>,
<tf.Variable 'MobilenetV1/Conv2d_13_depthwise/depthwise_weights:0' shape=(3, 3, 256, 1) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Conv2d_13_depthwise/BatchNorm/gamma:0' shape=(256,) dtype=float32_ref>,
<tf.Variable 'MobilenetV1/Conv2d_13_depthwise/BatchNorm/beta:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Conv2d_13_pointwise/weights:0' shape=(1, 1, 256, 256) dtype=float32_ref>,
<tf.Variable 'MobilenetV1/Conv2d_13_pointwise/BatchNorm/gamma:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta:0'
shape=(256,) dtype=float32_ref>,
<tf.Variable 'MobilenetV1/Conv2d_12_depthwise/depthwise_weights:0' shape=(3, 3, 128, 1) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Conv2d_12_depthwise/BatchNorm/gamma:0' shape=(128,) dtype=float32_ref>,
<tf.Variable 'MobilenetV1/Conv2d_12_depthwise/BatchNorm/beta:0' shape=(128,) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Conv2d_12_pointwise/weights:0' shape=(1, 1, 128, 256) dtype=float32_ref>,
<tf.Variable 'MobilenetV1/Conv2d_12_pointwise/BatchNorm/gamma:0' shape=(256,) dtype=float32_ref>, <tf.Variable 'MobilenetV1/Conv2d_12_pointwise/BatchNorm/beta:0' shape=(256,) dtype=float32_ref>]
```

Below command is an example for training the flowers dataset on mobilenet_v1_025 fine-tuned from ImageNet, We set the following input arguments
- `--experiment_name`: Set experiment name directory to save experiment files under (default will select `experiment_1` if it is the first experiment in the project)
- `--max_number_of_steps`: Set the maximum number of training steps to 1000 (default 50000 steps)
- `--model_name`: Set the model architecture used to mobilenet_v1_025 (default mobilenet_v1)
- `--trainable_scopes`: Set the model layers to trained (default `logits` and `BatchNorm` layers)
- `--clone_on_cpu=True`: Set training using CPU only (default train mainly on GPU if available)

```bash
$ python train_image_classifier.py \
    --project_name=flowers_classifier \
    --dataset_name=flowers \
    --experiment_name=experiment_1 \
    --batch_size=64 \
    --model_name=mobilenet_v1_025 \
    --max_number_of_steps=1000 \
    --trainable_scopes=BatchNorm,MobilenetV1/Logits,MobilenetV1/Conv2d_13,MobilenetV1/Conv2d_12 \
    --clone_on_cpu=True
```

For training on cpu (with tensorflow package, instead of tensorflow-gpu), set flag `--clone_on_cpu` to `True`. For training on gpu, this flag can be ignored or set to `False`.

### Evaluating performance of a model while training

To evaluate the performance of a model while training,
you can use the eval_image_classifier.py script, as shown below.

The script should be run while training and the `--project_name` and `--dataset_name` flags should point to the same project and dataset names as your training script. In addition, the script will save the event files under a new directory `./project_dir/<project name>/experiments/<experiment name>/eval`.

By default the `--checkpoint_path` flag will point to the following directory ``./project_dir/<project name>/experiments/<experiment name>/train`` . Optionally, you call also specify the `--checkpoint_path` flag,  which should point to the directory where the training job checkpoints are stored.

Below command is an example for monitoring and evaluating the training process for our flowers classifier. We set the following input arguments
- `--experiment_name`: Set experiment name directory load trained checkpoints from and save event logfiles to (default will select `experiment_#` the last folder with the highest number)
- `--model_name`: Set the model architecture used to mobilenet_v1_025 (default mobilenet_v1). This has to be the same as the training.
- `--batch_size`: Set batch size used to 64 (default 16)

```bash
$ python eval_image_classifier.py \
    --project_name=flowers_classifier \
    --dataset_name=flowers \
    --experiment_name=experiment_1 \
    --batch_size=64 \
    --model_name=mobilenet_v1_025
```

See the [evaluation module example](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/slim#evaluation-loop)
for an example of how to evaluate a model at multiple checkpoints during or after the training.

### Test performance of a model
To Test the performance of a model after completing training,
you can use the test_image_classifier.py script, as shown below.

The script should be run while/after training and the `--project_name` and `--dataset_name` flags should point to the same project and dataset names as your training script. In addition, the script will save the event files under a new directory `./project_dir/<project name>/experiments/<experiment name>/test`.

By default the `--checkpoint_path` flag will point to the following directory `./project_dir/<project name>/experiments/<experiment name>/train`. Optionally, you call also specify the `--checkpoint_path` flag,  which should point to the directory where the training job checkpoints are stored.

Below command is an example for monitoring and evaluating the training process for our flowers classifier. We set the following input arguments
- `--experiment_name`: Set experiment name directory load trained checkpoints from and save event predictions to (default will select `experiment_#` the last folder with the highest number)
- `--model_name`: Set the model architecture used to mobilenet_v1_025 (default mobilenet_v1). This has to be the same as the training.
- `--batch_size`: Set batch size used to 64 (default 16)

```bash
$ python test_image_classifier.py \
    --project_name=flowers_classifier \
    --dataset_name=flowers \
    --experiment_name=experiment_1 \
    --model_name=mobilenet_v1_025
```

### TensorBoard

To visualize the losses and other metrics during training, you can use
[TensorBoard](https://github.com/tensorflow/tensorboard)
by running the command below.

```shell
tensorboard --logdir=./project_dir/flower_classifier/experiments/experiment_1 --port 6006
```

Once TensorBoard is running, navigate your web browser to http://localhost:6006

### Export and Freeze Graph
Saves out a GraphDef containing the architecture with your pretrained
checkpoints. Run export_freeze_inference_graph to get a frozen graph
(.pb) file. By default when you run the training script the script will automatically export and freeze functions and generate a frozen graph under your the following directory `./project_dir/<project name>/experiments/<experiment name>/train`.

By default the `--checkpoint_path` flag will point to the following directory `./project_dir/<project name>/experiments/<experiment name>/train` and will select the latest saved checkpoint in that directory. Optionally, you call also specify the `--checkpoint_path` flag,  which should point to the directory where the training job checkpoints are stored.

Below command is an example for exporting and freezing the latest trained checkpoint for our flowers classifier. We set the following input arguments
- `--experiment_name`: Set experiment name directory load trained checkpoints from and save event predictions to (default will select `experiment_#` the latest folder with the highest number)
- `--model_name`: Set the model architecture used to mobilenet_v1_025 (default mobilenet_v1). This has to be the same as the training.

```bash
python export_freeze_inference_graph.py \
      --project_name=flowers_classifier \
      --dataset_name=flowers \
      --experiment_name=experiment_1 \
      --model_name=mobilenet_v1_025
```

<!-- ## Guildai for Hyperparameter search
 -->
## Troubleshooting and Current Known Issues
<a id='Troubleshooting'></a>

#### Known issues that need to be addressed:




#### The model runs out of CPU memory.

See
[Model Runs out of CPU memory](https://github.com/tensorflow/models/tree/r1.13/research/inception#the-model-runs-out-of-cpu-memory).

#### The model runs out of GPU memory.

See
[Adjusting Memory Demands](https://github.com/tensorflow/models/tree/r1.13/research/inception#adjusting-memory-demands).

#### The model training results in NaN's.

See
[Model Resulting in NaNs](https://github.com/tensorflow/models/tree/r1.13/research/inception#the-model-training-results-in-nans).


#### I wish to train a model with a different image size.

The preprocessing functions all take `height` and `width` as parameters. You
can change the default values using the following snippet:

```bash
$ python train_image_classifier.py \
    --project_name=flowers_classifier \
    --dataset_name=flowers \
    --experiment_name=experiment_1 \
    --train_image_size=224
```

#### What hardware specification are these hyper-parameters targeted for?

See
[Hardware Specifications](https://github.com/tensorflow/models/tree/master/research/inception#what-hardware-specification-are-these-hyper-parameters-targeted-for).

## Send Us Failure Cases and Feedback!
Our library is open source and we want to continuously improve it! So please, let us know if...

1. ... you find that the default training script argument settings do not seems to work well for your dataset (low accuracy). Feel free to send use a sample of your dataset. We will try to give you some suggestions.
2. ... you find any bug (in functionality or speed).
3. ... you added some functionality to some class.
4. ... you know how to speed up or improve any part of the library.
5. ... you have a request about possible functionality.
6. ... edits to our readme file.
7. ... etc.


## Contacts
This repository is maintained by FLIR-IIS R&D team.
* Ahmed Sigiuk, Ahmed.Sigiuk@flir.com
* Di Xu, Di.Xu@flir.com
* Douglas Chong, Douglas.Chong@flir.com

## License
Please, see the [license](LICENSE) for further details.

## References
"TensorFlow-Slim image classification model library"
N. Silberman and S. Guadarrama, 2016.
https://github.com/tensorflow/models/tree/master/research/slim
