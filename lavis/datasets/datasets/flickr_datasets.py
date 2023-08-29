import os
import pandas as pd
from PIL import Image
import torch

from lavis.datasets.datasets.base_dataset import BaseDataset


class FlickrDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])
        self.annotation = []
        print("annotation")
        print(ann_paths)
        for ann in ann_paths:
            # self.annotation.extend(pd.read_parquet(ann))
            # self.annotation = pd.read_parquet(ann)
            self.annotation = pd.read_json(ann)

    def __getitem__(self, index):
        print("vizwiz item!")
        ann = self.annotation.iloc[index]

        image_path = os.path.join(self.vis_root, ann["image"]) # ann['image'] = flickr30k-images/00001.jpg
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        instruction = '<Image> A short image description:'

        instruction = self.text_processor(instruction)

        answer = ann["caption"]

        return {
            "image": image,
            "text_input": instruction,
            "text_output" : answer,
        }