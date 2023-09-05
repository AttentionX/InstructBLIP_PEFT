import os
import json
import pandas as pd
from PIL import Image
import torch

from lavis.datasets.datasets.base_dataset import BaseDataset

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from collections import OrderedDict

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["id"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )
    
class IconQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images 
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])

        self.annotation = []

        for ann in ann_paths:
            print(f"ann path is :", ann)
            self.annotation = pd.read_json(ann)

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]

        image_path = os.path.join(self.vis_root, f'{ann["id"]}/image.png')
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        options = []
        for index, el in enumerate(ann["choices"]):
            option = f'({chr(index+ord("a"))}) {el}'
            options.append(option)
        options = " ".join(options)

        instruction = f'<Image> Question: {ann["question"]} Options: {options}. Short answer:'
        
        instruction = self.text_processor(instruction)

        answer = ann["answers"]

        return {
            "image": image,
            "text_input": instruction,
            "text_output" : answer,
        }
    
    def collater(self, samples):
        image_list, question_list, answer_list = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])

            answers = sample["text_output"]

            answer_list.extend([answers])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }
    
class IconQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images 
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])

        self.annotation = []

        for ann in ann_paths:
            self.annotation = pd.read_json(ann)

        # self._add_instance_ids() -> 왜 필요한지?

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]

        image_path = os.path.join(self.vis_root, f'{ann["id"]}/image.png')
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        options = []
        for index, el in enumerate(ann["choices"]):
            option = f'({chr(index+ord("a"))}) {el}'
            options.append(option)
        options = " ".join(options)

        instruction = f'<Image> Question: {ann["question"]} Options: {options}. Short answer:'
        
        instruction = self.text_processor(instruction)

        answer = '('+chr(ann["answers"]+ord("a"))+')'

        return {
            "image": image,
            "text_input": instruction,
            "text_output" : answer,
            "answer" : answer,
            "question_id": ann["id"]
        }
    
    def collater(self, samples):
        image_list, question_list, answer_list, id_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])

            answers = sample["text_output"]

            answer_list.extend([answers])

            id_list.extend(sample["id"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            # "text_output": answer_list,
            "answer": answer_list,
            "question_id": id_list
        }
