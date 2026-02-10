PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_NAME=CHECKPOINT_NAME

accelerate launch --num_processes 4 --gpu_ids 0,1,6,7 --config_file accelerate_configs/deepspeed_zero3.yaml --main_process_port 11002 \
    train.py \
    --model_name_or_path PRETRAINED_MODEL_PATH \
    --dataset_name musique \
    --train_data_path TRAIN_DATA_PATH \
    --eval_data_path TRAIN_DATA_PATH \
    --output_dir SAVE_PATH \
    --bf16 True \
    --gradient_checkpointing True \
    --max_steps 400 \
    --eval_strategy no \
    --eval_steps 100 \
    --logging_steps 10 \
    --save_strategy no \
    --save_steps 100 \
    --save_total_limit 5 \
    --save_only_model True \
    --metric_for_best_model reward \
    --learning_rate 1e-6 \
    --min_lr 0 \
    --start_pA 0.9 \
    --end_pA 0.2 \
    --sampling_schedule cosine \
    --beta 0.04 \
    --max_prompt_length 32000 \
    --max_completion_length 2000 \
    --num_generations 4 \
    --num_iterations 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --use_vllm False \
    --vllm_device auto \
    --vllm_enable_prefix_caching False \
    --vllm_gpu_memory_utilization 0.6 \
    --temperature 0.9 \
    --top_p 0.95 \
    --top_k 50 \
    --run_name $RUN_NAME \
    --report_to wandb