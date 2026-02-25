FROM lsj_habitat_base:0.2.4

COPY our_requirements.txt /app/our_requirements.txt
RUN pip install --no-cache-dir --no-build-isolation -r /app/our_requirements.txt
RUN pip install --no-cache-dir flash-attn==2.8.2 --no-build-isolation

RUN pip install --no-cache-dir python-dotenv wandb autoawq
RUN pip install --no-cache-dir pycocotools==2.0.6 groundingdino-py --no-build-isolation
RUN pip install --no-cache-dir pre-commit six
RUN pip install --no-cache-dir sentencepiece accelerate bitsandbytes compressed-tensors
RUN pip install --no-cache-dir numpy==1.24.4 numba==0.57.1 numpy-quaternion==2023.0.4

ARG USER_ID_LSJ=1002
ARG GROUP_ID_LSJ=1002
RUN addgroup --gid $GROUP_ID_LSJ lsj && \
    adduser --home /home/lsj --disabled-password --gecos '' --uid $USER_ID_LSJ --gid $GROUP_ID_LSJ lsj
RUN echo "lsj ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/lsj

RUN chown -R lsj:lsj /app /tmp/xdg_runtime_dir && \
    chmod -R 755 /app

USER lsj

# docker build -t lsj_vlfm:0.1 -f Dockerfile .

# docker run --rm -it --gpus all --shm-size=64G -v /hdd/hdd3/lsj/vlfm:/app/repo lsj_vlfm:0.1 /bin/bash
# sudo su
# export PATH="/opt/conda/bin:/app:$PATH"
# export PATH="/usr/local/cuda/bin:${PATH}"
# export CUDA_HOME="/usr/local/cuda"
# pip install -e .[habitat]

# docker commit <container_id> lsj_vlfm:0.2

