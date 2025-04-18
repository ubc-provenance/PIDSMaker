FROM ubuntu:22.04

# setting up environment variables (timezone for postgresql)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update \
    && apt-get install -y wget ca-certificates graphviz gnupg lsb-release maven

# installing JDK1.8
RUN apt update && \
    apt install -y openjdk-8-jdk && \
    apt clean

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
ENV PATH=$JAVA_HOME/bin:$PATH

# installing sudo
RUN apt-get update && apt-get install -y sudo
RUN apt-get install -y git

# installing Anaconda version 23.3.1
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
RUN bash Anaconda3-2023.03-1-Linux-x86_64.sh -b -p /opt/conda
RUN rm Anaconda3-2023.03-1-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"

# installing python libraries
RUN conda create -n velox python=3.9 && \
    echo "source /opt/conda/bin/activate velox" >> ~/.bashrc
# https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["conda", "run", "-n", "velox", "/bin/bash", "-c"]
# Activate the environment and install dependencies
RUN conda install -y psycopg2 tqdm && \
    pip install scikit-learn==1.2.0 networkx==2.8.7 xxhash==3.2.0 \
                graphviz==0.20.1 psutil scipy==1.10.1 matplotlib==3.8.4 \
                wandb==0.16.6 chardet==5.2.0 nltk==3.8.1 igraph==0.11.5 \
                cairocffi==1.7.0 wget==3.2

RUN conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
                pytorch-cuda=11.7 -c pytorch -c nvidia

RUN pip install torch_geometric==2.5.3 --no-cache-dir && \
    pip install pyg_lib==0.2.0 torch_scatter==2.1.1 torch_sparse==0.6.17 \
                torch_cluster==1.6.1 torch_spline_conv==1.2.2 \
                -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --no-cache-dir

RUN pip install gensim==4.3.1 pytz==2024.1 pandas==2.2.2 yacs==0.1.8

# RUN pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.6.* cuml-cu12==24.6.*

RUN pip uninstall -y scipy && pip install scipy==1.10.1 && \
    pip uninstall -y numpy && pip install numpy==1.26.4

RUN pip install gdown

# Create a user and set the working directory
# RUN useradd -m velox
# USER velox
WORKDIR /home
