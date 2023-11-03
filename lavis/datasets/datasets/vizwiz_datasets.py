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
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )

    
class VizWizDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, train_samples_portion="all"):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])
        self.annotation = []
        for ann in ann_paths:
            self.annotation = pd.read_json(ann)
                       
        if not (type(train_samples_portion) == int or train_samples_portion == "all" ):
            raise ValueError("train_samples_portion must be a positive integer or \"all\"")
        if train_samples_portion != "all":
            self.annotation = self.annotation.sample(n=train_samples_portion)

    
    def __getitem__(self, index):
        ann = self.annotation.iloc[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        instruction = f'<Image> Question: {ann["question"]} Short answer:'
        instruction = self.text_processor(instruction)
        answer = []
        for i in range(10):
            answer.append(ann["answers"][i]["answer"])
        # answer =  # 0~9

        return {
            "image": image,
            "text_input": instruction,
            "text_output" : answer,
        }
    


class VizWizEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images 
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = [] 
        for ann in ann_paths:
            self.annotation.extend(json.load(open(ann, encoding='UTF-8')))

        ## TODO: support inference method == 'ranking'
        answer_list_path = ann_paths[0] if len(ann_paths) > 0 else ''
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path, encoding='UTF-8'))
        else:
            print("None!!")
            self.answer_list = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        if "answers" in ann.keys(): 
            answer = ann["answers"]
        else:
            answer = [""]

        return {
            "image": image,
            "image_name" : ann["image"],
            "text_input": question,
            "answer": answer,
            "question_id": ann["instance_id"]
        }
