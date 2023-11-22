FROM osrf/ros:melodic-desktop-full-bionic

#install required commons
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa
    # apt install -y python3.8 python3-pip
RUN apt-get -y install python-pip

# RUN apt-get -y install qt5-default python3-pyqt5
# RUN pip install pyqt5 --verbose
RUN pip install debugpy
RUN apt-get -y install net-tools
RUN apt-get install htop
# RUN pip install cv_bridge

RUN mkdir -p /home/user/ws/event_stereo_ICCV2019
WORKDIR /home/user/ws/event_stereo_ICCV2019
# RUN pip install -r ../requirements.txt

ADD ../Anaconda3-2023.09-0-Linux-x86_64.sh /home/user/ws/Anaconda3-2023.09-0-Linux-x86_64.sh
ENV CONDA_DIR /opt/conda
RUN bash ../Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /opt/conda
RUN bash
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda config --append channels conda-forge
ADD ./tools/requirements.txt /home/user/ws/requirements.txt
RUN conda create --name  convert_mvsec --file /home/user/ws/requirements.txt --yes python=2.7
# ENV CUDA_VISIBLE_DEVICES 4,5,6,7
RUN apt-get install -y wget

ADD ./docker_start.sh /home/user/ws/docker_start.sh
RUN chmod +x /home/user/ws/docker_start.sh
RUN mkdir -p /home/mvsec_data
RUN pip install pycryptodomex
RUN pip install gnupg
RUN pip install pyclick
RUN pip install opencv-python
# RUN pip install torch==1.13.1
# torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
CMD ["home/user/ws/docker_start.sh"]