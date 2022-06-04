FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata build-essential python3 python3-dev python3-pip python3-opencv python-is-python3 wget git git-lfs \
    && apt-get clean

RUN mkdir -p /src
WORKDIR /src
RUN mkdir -p models/ldm/text2img-large/
RUN wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt

RUN pip install torch==1.7.0 torchvision==0.8.1 numpy==1.19.2 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install albumentations==0.4.3
RUN pip install pudb==2019.2
RUN pip install imageio==2.9.0
RUN pip install imageio-ffmpeg==0.4.2
RUN pip install pytorch-lightning==1.5
RUN pip install omegaconf==2.1.1
RUN pip install test-tube>=0.7.5
RUN pip install streamlit>=0.73.1
RUN pip install einops==0.3.0
RUN pip install torch-fidelity==0.3.0
RUN pip install transformers==4.3.1
RUN pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
RUN pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
COPY . .
RUN pip install -e .
ENTRYPOINT ["python", "scripts/txt2img.py"]
