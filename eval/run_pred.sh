CUDA_VISIBLE_DEVICES=0,1,2,3 python pred_fast.py \
--model_path SAVE_PATH \
--data_dir ./longbench/data \
--max_length 128000 \
--output_dir RESULT_PATH