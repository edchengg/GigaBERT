## RE
This repo contains code and data for RE in [GigaBERT](https://arxiv.org/pdf/2004.14519.pdf):

## Dependency
transformers==2.4.1

## How to run
1. Please download checkpoint folders (e.g., GigaBERT-v3). Each folder contains config.json (model configuration file), pytorch_model.bin (pytorch version model weights), and vocab.txt (for tokenizer). Please see [Wuwei Lan](https://github.com/lanwuwei/GigaBERT)'s repo for checkpoints.

2. Run zero-shot experiment:
```
python main.py --source_language en --target_language ar --bert_model GigaBERT-v3 --output_dir save --exp_name zeroshot --gpuid 0
