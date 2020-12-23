# The offica tensorflow/tensorflow docker package
#FROM tensorflow/tensorflow:latest-devel-gpu

# The offical Nvidia cuda 10.0 docker package
#FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04
FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
# Add user
ARG USER=docker
ARG UID=1000
ARG GID=1000

# Sudo user password
ARG PW=docker

# Temporary assign user as root to perform apt-get and sudo functions
USER root
RUN useradd -m ${USER} --uid=${UID} &&  echo "${USER}:${PW}" | chpasswd

# This line is optional
# RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


# Install sudo and add user to sudo group
RUN apt-get --allow-insecure-repositories update
RUN apt-get install -y -q \
	build-essential cmake checkinstall \
	pkg-config \
  wget git curl \
  unzip yasm \
  pkg-config \
  nano vim \
  mc sudo \
  python3-tk \
  x11-apps

# add sudo user
RUN  adduser ${USER} sudo

# python libraries
RUN apt-get install -y -q python3-dev

# Install the latest version of pip (https://pip.pypa.io/en/stable/installing/#using-linux-package-managers)
# TODO: upgrade from python 3.5 to 3.7
RUN wget --no-check-certificate  https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN pip install numpy

#RUN apt install python3-tesresources libjasper-dev

#optional dependencies Image I/O libs
RUN apt-get install -y -q  libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev qt5-default

# Video/Audio Libs - FFMPEG, GSTREAMER, x264 and so on.
RUN apt-get install -y -q libavcodec-dev libavformat-dev libswscale-dev \
libavresample-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev libfaac-dev libmp3lame-dev libvorbis-dev

# Parallelism library C++ for CPU
RUN apt-get install -y -q libtbb2 libtbb-dev

#Optimization libraries for OpenCV
RUN apt-get install -y -q libatlas-base-dev gfortran
#Optional libraries:
RUN apt-get install -y -q libprotobuf-dev protobuf-compiler \
libgoogle-glog-dev libgflags-dev \
libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

# GTK (gtk3) support for GUI features for the graphical user functionalites
RUN apt-get install -y -q libgtk2.0-dev \
            libgtk-3-dev libpq-dev libqt4-dev


# Install extra packages that require root privilege if needed
RUN apt-get install -y -q protobuf-compiler

RUN pip install opencv-contrib-python

#######OPENCV INSTALLATION##############
ENV OPENCV_TEST_DATA_PATH=/opt/opencv_extra/testdata/
ENV OPENCV_ROOT=/opt
WORKDIR $OPENCV_ROOT

# set build arguments
ARG CLONE_TAG=master
ARG OPENCV_TEST_DATA_PATH=/opt/opencv_extra/testdata

# opencv extra test dataset
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/opencv/opencv_extra.git
# contrib repo
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/opencv/opencv_contrib.git
# opencv repo
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/opencv/opencv.git

#&& git checkout master
#-D CMAKE_INSTALL_PREFIX=/usr/local     \
RUN cd opencv && mkdir build && cd build && \
    cmake   -D CMAKE_BUILD_TYPE=RELEASE \
            -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
            -D PYTHON_EXECUTABLE=$(which python) \
            -D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
            -D PYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
            -D INSTALL_C_EXAMPLES=ON     \
            -D INSTALL_PYTHON_EXAMPLES=ON     \
            -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules     \
            -D BUILD_EXAMPLES=ON  \
            -D BUILD_NEW_PYTHON_SUPPORT=ON  \
            -D BUILD_opencv_python3=ON  \
            -D HAVE_opencv_python3=ON \
            -D BUILD_TIFF=ON \
            -D BUILD_opencv_java=OFF \
            -D WITH_CUDA=OFF \
            -D WITH_OPENGL=ON \
            -D WITH_OPENCL=ON \
            -D WITH_IPP=ON \
            -D WITH_TBB=ON \
            -D WITH_EIGEN=ON \
            -D WITH_V4L=ON \
            -D WITH_QT=ON \
            -D WITH_GTK=ON \
            -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 ..

RUN cd opencv/build && make -j8
RUN cd opencv/build && make install
RUN opencv/build/bin/opencv_test_core

ENV QT_X11_NO_MITSHM=1

####### TENSORFLOW INSTALLATION ########
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

# Setup default user, when enter docker container
ENV PATH=$PATH:/home/docker/.local/bin
USER ${UID}:${GID}
WORKDIR /home/${USER}

## Specify tensorflow version

# Install extra packages without root privilege if need
RUN pip install --user tensorflow-gpu==1.13.2 guildai pillow scikit-learn scikit-image imgaug

# RUN ln -s python3 /usr/bin/python
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
