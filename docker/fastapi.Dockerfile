FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]

ARG REPO_DIR="."
ARG CONDA_ENV_FILE="FastSAM.yml"
ARG CONDA_ENV_NAME="FastSAM"
ARG PROJECT_NAME="tap-va"
ARG PROJECT_USER="user"
ARG HOME_DIR="/home/$PROJECT_USER"

# Miniconda arguments
ARG CONDA_HOME="/miniconda3"
ARG CONDA_BIN="$CONDA_HOME/bin/conda"
ARG MINI_CONDA_SH="Miniconda3-py39_4.12.0-Linux-x86_64.sh"

WORKDIR $HOME_DIR

RUN groupadd -g 8888 $PROJECT_USER && useradd -u 8888 -g 8888 -m $PROJECT_USER

RUN touch "$HOME_DIR/.bashrc"

#Get CUDA GPG key for ubuntu1804
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && \
    apt-get -y install curl locales git libgl1-mesa-glx libglib2.0-0 libsm6 libxrender-dev libxext6 && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8 && \
    apt-get clean

RUN mkdir $CONDA_HOME && chown -R 8888:8888 $CONDA_HOME
RUN chown -R 8888:8888 $HOME_DIR

ENV PYTHONIOENCODING utf8
ENV LANG "C.UTF-8"
ENV LC_ALL "C.UTF-8"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Miniconda
RUN curl -O https://repo.anaconda.com/miniconda/$MINI_CONDA_SH && \
    chmod +x $MINI_CONDA_SH && \
    ./$MINI_CONDA_SH -u -b -p $CONDA_HOME && \
    rm $MINI_CONDA_SH
ENV PATH $CONDA_HOME/bin:$HOME_DIR/.local/bin:$PATH

COPY $REPO_DIR/$CONDA_ENV_FILE $PROJECT_NAME/$CONDA_ENV_FILE

# Install conda environment
RUN $CONDA_BIN env create -f $PROJECT_NAME/$CONDA_ENV_FILE && \
    $CONDA_BIN init bash && \
    $CONDA_BIN clean -a -y && \
    echo "source activate $CONDA_ENV_NAME" >> "$HOME_DIR/.bashrc"

COPY $REPO_DIR $PROJECT_NAME

RUN mkdir $HOME_DIR/$PROJECT_NAME/weights

RUN chown -R 8888:8888 $HOME_DIR && \
    rm /bin/sh && ln -s /bin/bash /bin/sh

WORKDIR $HOME_DIR/$PROJECT_NAME

USER 8888

EXPOSE 4000

RUN chmod -R +x scripts
ENTRYPOINT [ "/bin/bash", "./scripts/entrypoint/api-entrypoint.sh" ]