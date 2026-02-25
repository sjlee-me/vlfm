#!/bin/bash

docker run --rm -it \
  --gpus all \
  --shm-size=64G \
  -v /hdd/hdd3/lsj/vlfm:/app/repo \
  -v /hdd/hdd3/lsj/HM3D_object_nav:/hdd/hdd3/lsj/HM3D_object_nav \
  -v /hdd/hdd3/lsj/LLM:/hdd/hdd3/lsj/LLM \
  lsj_vlfm:0.3 \
  /bin/bash
