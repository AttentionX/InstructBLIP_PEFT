touch /output/InstructBLIP/flickr/flickr_1.log
nohup python3 -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/instructblip/train/finetune_instructblip_flickr30k.yaml > /output/InstructBLIP/flickr/1.log
wait
touch /input/InstructBLIP/flickr/flickr_1.log
rsync -av --no-o --no-g --chmod=777 /output/InstructBLIP/flickr/1.log /input/InstructBLIP/flickr/1.log
wait
# The directory to iterate over
dir=/output/InstructBLIP/flickr

# The directory to copy to
dest=/input/InstructBLIP/flickr/1

# Iterate over all subdirectories
for subdir in "$dir"/*; do

    # Only copy directories and files that do not end with .pth
    if [[ ! "$subdir"/*.pth ]]; then
        rsync -av "$subdir" "$dest"
    fi
done
