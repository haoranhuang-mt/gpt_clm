import subprocess

cmd = 'python run_clm.py \
    --device musa \
    --model_name_or_path ./gpt2_ckpt \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --output_dir ./test-clm \
    --cache_dir ./gpt2_ckpt/wikitext/wikitext-2-raw'.split()

res = subprocess.call(cmd)
print(res)