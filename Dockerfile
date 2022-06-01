FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata build-essential wget git \
    && apt-get clean
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda
RUN mkdir -p /src
WORKDIR /src
SHELL ["/bin/bash", "--login", "-c"]
RUN /root/miniconda/bin/conda init bash
RUN echo "export PATH=$PATH:/root/miniconda/bin"

COPY . .
RUN conda env create -f environment.yaml
RUN echo "/root/miniconda/bin/conda activate ldm" >> ~/.bashrc

ENTRYPOINT ["python", "scripts/txt2img.py"]
