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
RUN apt-get update && apt-get install -y sudo git

# installing Anaconda version 23.3.1
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
RUN bash Anaconda3-2023.03-1-Linux-x86_64.sh -b -p /opt/conda
RUN rm Anaconda3-2023.03-1-Linux-x86_64.sh

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME}
WORKDIR /home/pids

ENV PATH="/opt/conda/bin:$PATH"
ENV PATH="/opt/conda/envs/pids/bin:$PATH"
RUN conda create -n pids python=3.9 && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /home/${USER_NAME}/.bashrc && \
    echo "conda activate pids" >> /home/${USER_NAME}/.bashrc
# https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["conda", "run", "-n", "pids", "/bin/bash", "-c"]
# Activate the environment and install dependencies
RUN conda install -y psycopg2 tqdm && \
    pip install scikit-learn==1.2.0 networkx==2.8.7 xxhash==3.2.0 \
                graphviz==0.20.1 psutil scipy==1.10.1 matplotlib==3.8.4 \
                wandb==0.16.6 chardet==5.2.0 nltk==3.8.1 igraph==0.11.5 \
                cairocffi==1.7.0 wget==3.2

RUN RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip install torch_geometric==2.5.3 --no-cache-dir && \
    pip install pyg_lib==0.2.0 torch_scatter==2.1.1 torch_sparse==0.6.17 \
                torch_cluster==1.6.1 torch_spline_conv==1.2.2 \
                -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --no-cache-dir

RUN pip install gensim==4.3.1 pytz==2024.1 pandas==2.2.2 yacs==0.1.8

RUN pip uninstall -y scipy && pip install scipy==1.10.1 && \
    pip uninstall -y numpy && pip install numpy==1.26.4

RUN pip install gdown==5.2.0
RUN pip install pytest==8.3.5 pytest-cov==6.1.1 pre-commit==4.2.0 setuptools==61.0 mkdocs-material==9.6.12 mkdocs-glightbox==0.4.0

COPY . .

# COPY is done by the docker daemon as root, so we need to chown
RUN chown -R ${USER_NAME}:${USER_NAME} /home
USER ${USER_NAME}


RUN [ -f pyproject.toml ] && pip install -e . || echo "No pyproject.toml found, skipping install"
RUN [ -f .pre-commit-config.yaml ] && pre-commit install || echo "No pre-commit found, skipping install"
