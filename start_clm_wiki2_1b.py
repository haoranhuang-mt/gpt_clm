import subprocess

cmd = 'python run_clm.py \
    --device musa \
    --config_name ./model \
    --tokenizer_name ./model \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --output_dir ./test-clm \
    --block_size 896 \
    --cache_dir ./gpt2_ckpt/wikitext/wikitext-2-raw'.split()

res = subprocess.call(cmd)
print(res)
