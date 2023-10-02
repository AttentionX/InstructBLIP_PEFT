### For ScienceQA dataset

1. download ScienceQA dataset from https://scienceqa.github.io/
2. run scienceqa_data_preprocess.py

This will save preprocessed scienceQA dataset in `/input/scienceqa/`.

### For IconQA dataset

1. download multi-text-choice dataset from https://iconqa.github.io/

IconQA does not need extra preprocessing.

### How to run

**Dataset**

Datasets must be placed in the location specified in the file `lavis/config/datasets/{dataset_name}/defaults.yaml` .

**Experiment ID**

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

**Run Script**

You can run experiment with this command.

```python
bash run_scripts/instructblkp/train/run_finetune_instructblip_experiments.sh {dataset_name} {experiment_id}
```

The result will be saved in `/input/results/{dataset_name}/{experiment_id}`. You can change this in sh file `run_finetune_instructblip_experiments.sh`.

For example, If you want to try experiment 15 for scienceqa, you can use this command.

```python
bash run_scripts/instructblkp/train/run_finetune_instructblip_experiments.sh scienceqa 15
```
