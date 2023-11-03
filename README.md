# Parameter-Efficient Fine-tuning of InstructBLIP for Visual Reasoning Tasks

we inspect the effectiveness of PEFT methods on the Q-Former and LLLM layer for Visual Reasoning Tasks.

## Overview

Visual language models have recently demonstrated enhanced capabilities in visual reasoning tasks by employing external modules upon language models for visual language alignment. InstructBLIP uses a Q-Former and a projection layer to convert input image embeddings into soft visual prompts to enhance the instruction-following capabilities of large language models (LLMs). Although fine-tuning InstructBLIP has shown great results in downstream tasks, previous works have been restrictive, only fine-tuning the Q-Former, while freezing the LLM.

In this work, we investigate the performance of the PEFT method, LoRA, on both the Q-Former and the base LLMs, specifically Flan-T5-XL and Vicuna-7B, using visual reasoning benchmarks ScienceQA and IconQA. We observe that training the Q-Former with LoRA achieves comparable performance to full fine-tuning using under 2\% of the trainable parameters. Furthermore, fine-tuning the LLM consistently result in better performances, regardless of how the Q-Former is fine-tuned.
Lastly, applying LoRA to both the LLM and the Q-Former surpasses the performance of only full fine-tuning the Q-Former while using less than 10\% of the trainable parameters. These results highlight the effectiveness of applying PEFT to visual language models for visual reasoning tasks.


## Citation

## Acknowledgement

- [@Lightning-AI](https://github.com/Lightning-AI) for [lit-llama](https://github.com/Lightning-AI/lit-llama)
- [@FacebookResearch](https://github.com/facebookresearch) for the original [LLaMA implementation](https://github.com/facebookresearch/llama)
- [@Salesforce](https://github.com/salesforce) for [LAVIS](https://github.com/salesforce/LAVIS)

## License

[BSD 3-Clause License](LICENSE.txt) (from LAVIS)

[Apache 2.0 License](LICENSE) (From lit-llama)
