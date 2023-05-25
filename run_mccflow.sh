
DATA=${DATADIR:-"/data01"}
mkdir -p $DATA

horovodrun -np 248 -hostfile /etc/mccflow/hostfile \
python3 train_hvd_clm.py \
    --device musa \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --checkpointing_steps 1000 \
    --output_dir $DATA/clm-output \
    --cache_dir $DATA/cache \ 
    --from_checkpoint_meta $DATA/checkpoints \
    --log_dir $DATA/logs