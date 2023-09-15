mkdir -p /output/results/flickr/flickr_1
mkdir -p /input/results/flickr/flickr_1
touch /output/results/flickr/flickr_1/flickr_1.log
export TORCH_HOME=/input/torch
export HUGGINGFACE_HUB_CACHE=/input/huggingface
nohup python3 -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/instructblip/train/flickr/finetune_instructblip_flickr_1.yaml 2>&1 | tee /output/results/flickr/flickr_1/flickr_1.log
wait
touch /input/results/flickr/flickr_1/flickr_1.log
rsync -av --no-o --no-g --chmod=777 /output/results/flickr/flickr_1/flickr_1.log /input/results/flickr/flickr_1/flickr_1.log
wait
# The directory to iterate over
dir=/output/results/flickr/flickr_1

# The directory to copy to
dest=/input/results/flickr/flickr_1

# Iterate over all subdirectories
for subdir in "$dir"/*; do

    # Only copy directories and files that do not end with .pth
    if [[ ! "$subdir"/*.pth ]]; then
        rsync -av "$subdir" "$dest"
    fi
done