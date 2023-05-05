import subprocess

cmd = 'python run_clm.py \
    --device musa \
    --model_name_or_path ./gpt2_ckpt \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --output_dir ./test-clm \
    --cache_dir ./gpt2_ckpt/wikitext/wikitext103'.split()

res = subprocess.call(cmd)
print(res)
