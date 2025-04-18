export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="lllyasviel/control_v11f1p_sd15_depth"
export REWARDMODEL_DIR="Intel/dpt-hybrid-midas"
export OUTPUT_DIR="work_dirs/MultiGen/reward_readout_blured_200_920_42_control_fix_diff_interpolated_200_readout_first_timesteps_1_05_1"
export HF_HUB_CACHE='/workspace-SR008.fs2/test/datasets'
accelerate launch --config_file "train/config.yml" \
 --main_process_port=23156 train/reward_control_noisy_fix_diff_readout.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="depth" \
 --dataset_name="limingcv/MultiGen-20M_depth" \
 --caption_column="text" \
 --conditioning_image_column="control_depth" \
 --cache_dir '/workspace-SR008.fs2/test/datasets/datasets--limingcv--MultiGen-20M_depth' \
 --resolution=512 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=4 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=8 \
 --max_train_steps=10000 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=10 \
 --checkpointing_steps=1000 \
 --grad_scale=0.5 \
 --use_ema \
 --validation_steps=1000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200 \
 --seed 42 \
 --readout_path '/workspace-SR008.fs2/test/controlnet_redout/weights/checkpoint_step_5000.pt' \
 --max_timestep_readout 920 \
 --min_timestep_readout 0 \
 --readout_alpha 0.5 \
 --readout_beta 1 \
 --enable_xformers_memory_efficient_attention \


# export CONTROLNET_DIR="work_dirs/MultiGen/reward_readout_blured_200_920_42_control_fix_diff_interpolated_200_readout_first_timesteps/checkpoint-10000/controlnet"  # Eval our ControlNet++
# # How many PUs and processes you want to use for evaluation.
# export NUM_GPUS=8
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20

# # Generate images for evaluation
# # If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
# accelerate launch --main_process_port=12668 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --output_dir "work_dirs/reproduce/"

# # Path to the above generated images
# # guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/reproduce/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Calculate RMSE
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}


export DATA_DIR="/home/jovyan/shares/SR008.fs2/test/controlnet_redout/work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_readout_blured_200_920_42_control_fix_diff_interpolated_200_first_timesteps_checkpoint-10000_controlnet_7.5-20"

python3 eval/eval_depth.py --root_dir ${DATA_DIR}

