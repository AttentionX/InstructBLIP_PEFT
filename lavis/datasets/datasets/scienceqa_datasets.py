"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import pandas as pd
from PIL import Image
import torch
import json

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
        
class ScienceQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])
        self.annotation = []
        for ann in ann_paths:
            # self.annotation.extend(pd.read_parquet(ann))
            self.annotation = pd.read_json(ann)
        

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        
        # question = self.get_question(ann)
        # question = self.text_processor(question)
        
        text_input = self.get_text_input(ann)
        text_input = self.text_processor(text_input)

        answer = ann["answer"]

        return {
            "image": image,
            "text_input": text_input,
            "text_output" : answer,
        }
        
    def collater(self, samples):
        image_list, question_list, answer_list = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])

            answers = sample["text_output"]

            answer_list.extend(answers)

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }
    
    @staticmethod
    def get_question(sample):
        choices = ""
        
        i = 0
        for choice in sample["choices"]:
            choices += f"{i}. {choice}\n"
            i += 1
        
        question = f"""
        {sample["question"]}
        
        Choose from one of the following:
        {choices}
        
        Answer with a single number only.
        """
        return question
    
    @staticmethod
    def get_text_input(sample):
        choices = ""
        
        i = 0
        for choice in sample["choices"]:
            choices += f"{i}. {choice}\n"
            i += 1
        
        text_input = f"""Context: {sample['context']} Question: {sample['question']} Options: {choices}. Answer:"""
        
        return text_input
        
class ScienceQAEvalDataset(VQAEvalDataset, __DisplMixin):
    # scienceQA's test annotation is same as train annotation
    # So this class might not be necessary
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images 
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = []
        for ann in ann_paths:
            # self.annotation.extend(pd.read_parquet(ann))
            # to use self._add_instance_ids(), we need to use json.load
            self.annotation.extend(json.load(open(ann)))
        

        ## TODO: support inference method == 'ranking'
        answer_list_path = ann_paths[0] if len(ann_paths) > 0 else ''
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            print("None!!")
            self.answer_list = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        
        # question = self.get_question(ann)
        # question = self.text_processor(question)
        
        text_input = self.get_text_input(ann)
        text_input = self.text_processor(text_input)
        if "answer" in ann.keys(): 
            answer = ann["answer"]
        else:
            answer = ""
            
        return {
            "image": image,
            "text_input": text_input,
            "text_output" : answer,
            "question_id": ann["instance_id"]
        }
    
    @staticmethod
    def get_text_input(sample):
        choices = ""
        
        i = 0
        for choice in sample["choices"]:
            choices += f"{i}. {choice}\n"
            i += 1
        
        text_input = f"""Context: {sample['context']} Question: {sample['question']} Options: {choices}. Answer:"""
        
        return text_input