FROM paperspace/transformers-gpu:0.4.0

RUN export TZ=Europe/Madrid \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y build-essential cmake unzip pkg-config libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libx264-dev libgtk-3-dev libopenblas-dev libatlas-base-dev liblapack-dev gfortran libhdf5-serial-dev graphviz python3-dev python3-tk python-imaging-tk linux-image-generic linux-image-extra-virtual linux-source linux-headers-generic \
    && apt-get install -y wget \
    && wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run \
    && sh cuda_11.2.2_460.32.03_linux.run --silent --toolkit \
    && rm cuda_11.2.2_460.32.03_linux.run \
    && echo "export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}" >> ~/.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc \
    && echo "/usr/local/cuda-11.2/lib64" >> /etc/ld.so.conf \
    && wget https://www.dropbox.com/s/x3blciiax9qza49/cudnn-11.2-linux-x64-v8.1.0.77.tgz?dl=1 \
    && tar -zvxf cudnn-11.2-linux-x64-v8.1.0.77.tgz?dl=1 \
    && rm cudnn-11.2-linux-x64-v8.1.0.77.tgz?dl=1 \
    && cp cuda/include/*.h /usr/local/cuda/include/ \
    && chmod a+r /usr/local/cuda/include/cudnn.h \
    && ldconfig \
    && /bin/bash -c 'source ~/.bashrc'


#sudo docker build -t macarse/transformers-cuda11.2 .

#sudo docker push macarse/transformers-cuda11.2