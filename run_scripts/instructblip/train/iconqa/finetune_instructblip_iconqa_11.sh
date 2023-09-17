export TORCH_HOME=/input/torch
export HUGGINGFACE_HUB_CACHE=/input/huggingface

mkdir -p /output/results/iconqa/iconqa_11
mkdir -p /input/results/iconqa/iconqa_11
touch /output/results/iconqa/iconqa_11/iconqa_11.log
nohup python3 -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/instructblip/train/iconqa/finetune_instructblip_iconqa_11.yaml 2>&1 | tee /output/results/iconqa/iconqa_11/iconqa_11.log
wait
touch /input/results/iconqa/iconqa_11/iconqa_11.log
rsync -av --no-o --no-g --chmod=777 /output/results/iconqa/iconqa_11/iconqa_11.log /input/results/iconqa/iconqa_11/iconqa_11.log
wait
# The directory to iterate over
dir=/output/results/iconqa/iconqa_11

# The directory to copy to
dest=/input/results/iconqa/iconqa_11

# Iterate over all subdirectories
for subdir in "$dir"/*; do

    # Only copy directories and files that do not end with .pth
    if [[ ! "$subdir"/*.pth ]]; then
        rsync -av "$subdir" "$dest"
    fi
done