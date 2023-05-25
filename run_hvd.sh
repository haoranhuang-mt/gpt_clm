#horovodrun -np 1 -H 127.0.0.1:1 python3 train_hvd_clm.py \
#   --device musa \
#    --model_name_or_path gpt2 \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --per_device_train_batch_size 3 \
#    --per_device_eval_batch_size 3 \
#    --checkpointing_steps 1000 \
#    --output_dir /data01/clm-output \
#    --cache_dir /data01/gpt2-data/wikitext/wikitext-2-raw

DATA=${DATADIR:-"/data01"}

horovodrun -np 1 -H 127.0.0.1:1 python3 train_hvd_clm.py \
    --device musa \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --checkpointing_steps 1000 \
    --output_dir $DATA/clm-output \
    --cache_dir $DATA/cache \
    --from_checkpoint_meta $DATA/checkpoints