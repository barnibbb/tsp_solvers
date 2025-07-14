###Setup base
#
#Base image can be tricky. In my oppinion you should only use a few base images. Complex ones with 
#everything usually have special use cases, an in my experience they take more time to understand, 
#than building one from the ground up.
#The base iamges I suggest you to use:
#- ubuntu: https://hub.docker.com/_/ubuntu
#- osrf/ros:version-desktop-full: https://hub.docker.com/r/osrf/ros
#- nvidia/cuda: https://hub.docker.com/r/nvidia/cuda
#
#We are mostly not space constrained so a little bigger image with everything is usually better,
#than a stripped down version.

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
#this is a small but basic utility, missing from osrf/ros. It is not trivial to know that this is
#missing when an error occurs, so I suggest installing it just to bes sure.
RUN apt-get update && apt-get install -y netbase
#set shell 
SHELL ["/bin/bash", "-c"]
#set colors
ENV BUILDKIT_COLORS=run=green:warning=yellow:error=red:cancel=cyan
#start with root user
USER root

###Create new user
#
#Creating a user inside the container, so we won't work as root.
#Setting all setting all the groups and stuff.
#
###

#expect build-time argument
ARG HOST_USER_GROUP_ARG
#create group appuser with id 999
#create group hostgroup with ID from host. This is needed so appuser can manipulate the host files without sudo.
#create appuser user with id 999 with home; bash as shell; and in the appuser group
#change password of appuser to admin so that we can sudo inside the container
#add appuser to sudo, hostgroup and all default groups
#copy default bashrc and add ROS sourcing
RUN groupadd -g 999 appuser && \
    groupadd -g $HOST_USER_GROUP_ARG hostgroup && \
    useradd --create-home --shell /bin/bash -u 999 -g appuser appuser && \
    echo 'appuser:admin' | chpasswd && \
    usermod -aG sudo,hostgroup,plugdev,video,adm,cdrom,dip,dialout appuser && \
    cp /etc/skel/.bashrc /home/appuser/ 

###Install the project
#
#If you install multiple project, you should follow the same 
#footprint for each:
#- dependencies
#- pre install steps
#- install
#- post install steps
#
###

# Dependencies
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    g++ \
    gcc \
    git \
    libffi-dev \
    libglib2.0.0 \
    libgomp1 \
    libsm6 \
    libssl-dev \
    libtool \
    libxext6 \
    libxrender-dev \
    python3 \
    python3-dev \
    python3-pip \
    sudo \
    swig \
    unzip \
    wget \
    zlib1g-dev


# Install OpenCV for graph generation
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libopencv-dev \
    python3-opencv


# Python libraries
USER appuser
RUN pip install torch==1.12.1+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install numpy tqdm scikit-learn


# Copy qsopt
USER root
COPY qsopt /home/appuser/qsopt


# Install concorde
USER root
RUN cd /home/appuser && \
    wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz --no-check-certificate && \
    tar xzf co031219.tgz && \
    cd concorde && \
    CFLAGS="-fPIC" ./configure --with-qsopt=/home/appuser/qsopt && \
    make -j$(nproc)

ENV PATH="/home/appuser/concorde/TSP:${PATH}"

ENV CONCORDE_DIR=/home/appuser/concorde
ENV QSOPT_DIR=/home/appuser/qsopt


# Install pyconcorde
USER appuser
RUN cd /home/appuser && \
    git clone https://github.com/jvkersch/pyconcorde.git && \
    cd pyconcorde && \
    pip install .


# Install LKH
USER root
RUN cd /home/appuser && \
    wget http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz && \
    tar xzvf LKH-3.0.13.tgz && \
    cd LKH-3.0.13 && \
    make -j$(nproc)

ENV PATH="/home/appuser/LKH-3.0.13:${PATH}"


# Download and setup NeuroLKH
USER appuser
RUN cd /home/appuser && \
    git clone https://github.com/liangxinedu/NeuroLKH.git && \
    cd /home/appuser/NeuroLKH && \
    mkdir /home/appuser/NeuroLKH/SRC/OBJ && \
    make

# Download VSR-LKH
USER appuser
RUN cd /home/appuser && \
    git clone https://github.com/JHL-HUST/VSR-LKH.git && \
    cd /home/appuser/VSR-LKH && \
    make


# Download mlrose for GA reference
USER appuser
RUN pip install matplotlib && \
    pip install joblib==1.1.0 mlrose-hiive


