import os
import pandas as pd
from PIL import Image
import torch

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset


class FlickrDataset(BaseDataset):
    """Flickr30k caption dataset in instruction format"""
    def __getitem__(self, index):
        print("Flickr item!")
        ann = self.annotation[index]

        image_path = os.path.join(
            self.vis_root, ann["image"]
        )  # ann['image'] = flickr30k-images/00001.jpg
        image = Image.open(image_path).convert("RGB")

        # img_id = ann["image_id"]
        image = self.vis_processor(image)

        instruction = "<Image> A short image description:"
        instruction = self.text_processor(instruction)

        caption = ann["caption"]
        caption = self.text_processor(caption)

        return {
            "image": image,
            "text_input": instruction,
            "text_output": caption,
            # "image_id": img_id,
        }


class FlickrEvalDataset(CaptionEvalDataset):
    """Flickr30k eval dataset in instruction format"""

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.text_processor = text_processor

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        instruction = "<Image> A short image description:"
        instruction = self.text_processor(instruction)

        caption = ann["caption"][0]  # ann["caption"] is a list of 5 possible captions
        caption = self.text_processor(caption)

        print("eval data")
        print(
            {
                "image": image,
                "text_input": instruction,
                "text_output": caption,
            }
        )
        return {
            "image": image,
            "text_input": instruction,
            "text_output": caption,
        }
