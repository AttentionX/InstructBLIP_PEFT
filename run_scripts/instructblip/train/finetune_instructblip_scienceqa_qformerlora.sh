#!/bin/bash

# Set the environment variables
export TORCH_HOME=/input/torch
export HUGGINGFACE_HUB_CACHE=/input/huggingface

# Run the python command
python3 -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/instructblip/train/finetune_instructblip_vicuna_qformer_lora.yaml