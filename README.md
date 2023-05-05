
this is gpt2 scripts for musa/cpu/cuda

# train
Run the script directly and it will automatically download the dataset from huggingface
1. python start_clm_wiki2.py     # for wiki2 dataset
2. python start_clm_wiki103.py   # for wiki103 dataset
# refs
There are a few changes to run on the musa card.

scripts：https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py

datasets：https://huggingface.co/datasets/wikitext

