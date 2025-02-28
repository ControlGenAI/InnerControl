export CONTROLNET_DIR="MultiGen/reward_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_scale-1.0_iter10k_fp16_train0-1k_reward0-200_mse-loss_new_scale_seed_42_tune_model_10_alpha_lr_1e-5_attn/checkpoint-1000/controlnet"  # Eval our ControlNet++
# How many GPUs and processes you want to use for evaluation.
# export CONTROLNET_DIR="weights/controlnet"
export NUM_GPUS=1
# Guidance scale and inference steps
export SCALE=7.5
export NUM_STEPS=20
export CUDA_VISIBLE_DEVICES=0,

accelerate launch --main_process_port=12356 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

export DATA_DIR="work_dirs/eval_dirs/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}

