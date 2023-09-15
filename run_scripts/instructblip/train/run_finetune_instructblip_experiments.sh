#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Please provide a benchmark and an experiment number as arguments."
    echo "Usage: ./script_name.sh <benchmark> <experiment>"
    exit 1
fi

benchmark=$1
experiment=$2

export TORCH_HOME=/output/torch
export HUGGINGFACE_HUB_CACHE=/output/huggingface
# The directory to copy from
dir=/output/results/$benchmark/$benchmark_$experiment
# The directory to copy to
dest=/input/results/$benchmark/$benchmark_$experiment

mkdir -p $dir
mkdir -p $dest
touch $dir/$benchmark_$experiment.log
nohup python3 -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/instructblip/train/$benchmark/finetune_instructblip_$benchmark_$experiment.yaml 2>&1 | tee $dir/$benchmark_$experiment.log

rsync -av --no-o --no-g --chmod=777 $dir/$benchmark_$experiment.log $dest/$benchmark_$experiment.log

# Iterate over all subdirectories
for subdir in "$dir"/*; do
    rsync -av --no-o --no-g --chmod=777 "$subdir"/log.txt "$dest"/log.txt
    rsync -av --no-o --no-g --chmod=777 "$subdir"/result "$dest"/result
done
