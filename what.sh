#!/bin/bash

export TORCH_DISTRIBUTED_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export NCL_P2P_DISABLE=1

#--master_port 29501 \
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node=4 --nnodes=1 --master_port 29501 \
#    /home/allanz/ModelMerging/src/modelmerge/train/temp.py
deepspeed --include localhost:4,5,6,7 --master_port 29501 /home/allanz/ModelMerging/src/modelmerge/train/temp.py

