# Parameter-Efficient Fine-tuning of InstructBLIP for Visual Reasoning Tasks

We inspect the effectiveness of PEFT methods on the Q-Former and LLM layer for Visual Reasoning Tasks.

## Overview

Visual language models have recently demonstrated enhanced capabilities in visual reasoning tasks by employing external modules upon language models for visual language alignment. InstructBLIP uses a Q-Former and a projection layer to convert input image embeddings into soft visual prompts to enhance the instruction-following capabilities of large language models (LLMs). Although fine-tuning InstructBLIP has shown great results in downstream tasks, previous works have been restrictive, only full fine-tuning the Q-Former, while freezing the LLM.

In this work, we investigate the performance of the PEFT method, LoRA, on both the Q-Former and the base LLMs, specifically Flan-T5-XL and Vicuna-7B, using visual reasoning benchmarks ScienceQA and IconQA. We observe that, when the LLM is frozen, training the Q-Former with LoRA achieves comparable performance to full fine-tuning using under 2% of the trainable parameters. Furthermore, fine-tuning the LLM consistently result in better performances than InstructBLIP. Lastly, applying LoRA to both the LLM and the Q-Former surpasses the performance of only full fine-tuning the Q-Former while using less than 12% of the trainable parameters. These results highlight the effectiveness of applying PEFT to visual language models for visual reasoning tasks. The code is available at https://github.com/AttentionX/InstructBLIP_PEFT.
## Contents (index)

- [Install](#install)
- [Train](#train)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Install

### Install Code

1. Clone this repository and navigate to InstructBLIP_PEFT folder.

```bash
git clone https://github.com/AttentionX/InstructBLIP_PEFT.git
cd InstructBLIP_PEFT
```

2. Install Package

```bash
pip install -r requirements.txt
```

### Install ScienceQA dataset

1. download ScienceQA dataset from <https://scienceqa.github.io/>
2. run scienceqa_data_preprocess.py

This will save preprocessed scienceQA dataset in `/input/scienceqa/`.

This is the Instruction Format for ScienceQA dataset.

```md
Context: { {hint} {lecture} } Question: { {question} } Options: { {choices} } Answer: (a) { {answer} }
```

### Install IconQA dataset

1. download multi-text-choice dataset from <https://iconqa.github.io/>
2. run iconqa_data_preprocess.py

This will save preprocessed scienceQA dataset in `/input/iconqa/`.

This is the Instruction Format for IconQA dataset.

```md
<Image> Question: { {question} } Options: { {choices} }. Short answer: (a) { {answer} }
```

## Train

We train our model using a single A100 GPU.

### Dataset

Datasets must be placed in the location specified in the file `lavis/config/datasets/{dataset_name}/defaults.yaml` .

This is an example of dataset default.yaml file.

```yaml
# lavis/config/datasets/scienceqa/default.yaml
datasets:
  scienceqa:
    # data_dir: ${env.data_dir}/datasets
    data_type: images # [images|videos|features]

    build_info:
      # Be careful not to append minus sign (-) before split to avoid itemizing
      annotations:
        train:
          storage: /input/scienceqa/scienceqa_train.json
        val:
          storage: /input/scienceqa/scienceqa_val.json
        test:
          storage: /input/scienceqa/scienceqa_test.json
      images:
        storage: /input
        train:
          storage: /input
        val:
          storage: /input
        test:
          storage: /input
```

In this case, dataset json files (`scienceqa_train.json`, `scienceqa_test.json` and `scienceqa_val.json`) should be located at `/input/scienceqa`.  
Images files should be located at `input/scienceqa/images/train`, `input/scienceqa/images/test` and `input/scienceqa/images/val` because of the content in json files.

### Experiment ID

This is the table for the ID for each experiements.

|                                        | r = 1 | r = 2 | r = 4 | r = 8 |
| -------------------------------------- | ----- | ----- | ----- | ----- |
| LLM LoRA (ffn, FlanT5-XL)              | 1     | 2     | 3     | 4     |
| LLM LoRA (attn, FlanT5-XL)             | 5     | 6     | 7     | 8     |
| LLM LoRA (all, FlanT5-XL)              | 9     | 10    | 11    | 12    |
| Q-Former LoRA (ffn, FlanT5-XL)         | 13    | 14    | 15    | 16    |
| Q-Former LoRA (self-attn, FlanT5-XL)   | 17    | 18    | 19    | 20    |
| Q-Former LoRA (cross-attn, FlanT5-XL)  | 21    | 22    | 23    | 24    |
| Q-Former LoRA (all, FlanT5-XL)         | 25    | 26    | 27    | 28    |
| Q-Former and LLM LoRA (all, FlanT5-XL) | 29    | 30    | 31    | 32    |
| LLM LoRA (ffn, Vicuna-7B)              | 33    | 34    | 35    | 36    |
| LLM LoRA (attn, Vicuna-7B)             | 37    | 38    | 39    | 40    |
| LLM LoRA (all, Vicuna-7B)              | 41    | 42    | 43    | 44    |
| Q-Former LoRA (ffn, Vicuna-7B)         | 45    | 46    | 47    | 48    |
| Q-Former LoRA (self-attn, Vicuna-7B)   | 49    | 50    | 51    | 52    |
| Q-Former LoRA (cross-attn, Vicuna-7B)  | 53    | 54    | 55    | 56    |
| Q-Former LoRA (all, Vicuna-7B)         | 57    | 58    | 59    | 60    |
| Q-Former and LLM LoRA (all, Vicuna-7B) | 61    | 62    | 63    | 64    |

### Run Script

You can run experiment with this command.

```bash
bash run_scripts/instructblip/train/run_finetune_instructblip_experiments.sh {dataset_name} {experiment_id}
```

The result will be saved in `/input/results/{dataset_name}/{experiment_id}`. You can change this in `sh` file `run_finetune_instructblip_experiments.sh`.

For example, If you want to try experiment 15 for scienceqa, you can use this command.

```bash
bash run_scripts/instructblip/train/run_finetune_instructblip_experiments.sh scienceqa 15
```

## Citation
```bibtex
@inproceedings{kim2024towards,
  title={Towards Efficient Visual-Language Alignment of the Q-Former for Visual Reasoning Tasks},
  author={Kim, Sungkyung and Lee, Adam and Park, Junyoung and Chung, Andrew and Oh, Jusang and Lee, Jay-Yoon},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
  pages={15155--15165},
  year={2024}
}
```

## Acknowledgement

- [@Lightning-AI](https://github.com/Lightning-AI) for [lit-llama](https://github.com/Lightning-AI/lit-llama)
- [@FacebookResearch](https://github.com/facebookresearch) for the original [LLaMA implementation](https://github.com/facebookresearch/llama)
- [@Salesforce](https://github.com/salesforce) for [LAVIS](https://github.com/salesforce/LAVIS)

## License

[BSD 3-Clause License](LICENSE.txt) (from LAVIS)

[Apache 2.0 License](LICENSE) (From lit-llama)
