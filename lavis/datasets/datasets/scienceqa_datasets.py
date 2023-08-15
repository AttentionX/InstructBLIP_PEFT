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

from lavis.datasets.datasets.base_dataset import BaseDataset


class ScienceQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])
        self.annotation = []
        for ann in ann_paths:
            # self.annotation.extend(pd.read_parquet(ann))
            self.annotation = pd.read_parquet(ann)
        

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        
        question = self.get_question(ann)
        
        question = self.text_processor(question)

        answers = [ann["answer"]]

        return {
            "image": image,
            "text_input": question,
            "text_output" : answers,
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
