# DATA_DIR=/home/ychen/217/data/longbench/model_preds/rl-contamination-400-steps-two-ways/merge_50_perc_uncontam_0025/completely_two_ways_cosine_sampling_05_05_merge_50_perc_wo_extreme-qwen2.5-7b
DATA_DIR=/home/ychen/217/data/longbench/model_preds/rl-contamination-400-steps-two-ways/merge_50_perc_uncontam_0025/completely_two_ways_cosine_sampling_09_02_merge_75_perc_no_extreme_10-qwen2.5-7b

python llm_rating.py \
--res_dir $DATA_DIR/pred \
--output_dir $DATA_DIR/gpt-4o-ratings