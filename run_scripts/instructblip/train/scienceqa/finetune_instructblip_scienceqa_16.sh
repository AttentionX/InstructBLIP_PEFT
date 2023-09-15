export TORCH_HOME=/output/torch
export HUGGINGFACE_HUB_CACHE=/output/huggingface

mkdir -p /output/results/scienceqa/scienceqa_16
mkdir -p /input/results/scienceqa/scienceqa_16
touch /output/results/scienceqa/scienceqa_16/scienceqa_16.log
nohup python3 -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/instructblip/train/scienceqa/finetune_instructblip_scienceqa_16.yaml 2>&1 | tee /output/results/scienceqa/scienceqa_16/scienceqa_16.log
wait
touch /input/results/scienceqa/scienceqa_16/scienceqa_16.log
rsync -av --no-o --no-g --chmod=777 /output/results/scienceqa/scienceqa_16/scienceqa_16.log /input/results/scienceqa/scienceqa_16/scienceqa_16.log
wait
# The directory to iterate over
dir=/output/results/scienceqa/scienceqa_16

# The directory to copy to
dest=/input/results/scienceqa/scienceqa_16

# Iterate over all subdirectories
for subdir in "$dir"/*; do

    # Only copy directories and files that do not end with .pth
    if [[ ! "$subdir"/*.pth ]]; then
        rsync -av "$subdir" "$dest"
    fi
done