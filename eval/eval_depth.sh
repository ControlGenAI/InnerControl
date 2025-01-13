# Path to the controlnet weight (can be huggingface or local path)
# export CONTROLNET_DIR="lllyasviel/control_v11f1p_sd15_depth"  # Eval ControlNet

export CONTROLNET_DIR="/home/jovyan/konovalova/controlnet_redout/MultiGen_correct/reward_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_scale-1.0_iter10k_fp16_train0-1k_reward0-200_mse-loss_920_step_seed_666/checkpoint-10000/controlnet"  # Eval our ControlNet++
# How many GPUs and processes you want to use for evaluation.
export NUM_GPUS=1
# Guidance scale and inference steps
export SCALE=7.5
export NUM_STEPS=20
export CUDA_VISIBLE_DEVICES=0,

# Generate images for evaluation
# If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
accelerate launch --main_process_port=65335 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

# Path to the above generated images
# guidance_scale=7.5, sampling_steps=20 by default
export DATA_DIR="work_dirs/eval_dirs/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}


# export CONTROLNET_DIR="/home/jovyan/konovalova/controlnet_redout/MultiGen/reward_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_scale-1.0_iter10k_fp16_train0-1k_reward0-200_mse-loss_0_reward/checkpoint-10000/controlnet"  # Eval our ControlNet++
# # How many GPUs and processes you want to use for evaluation.
# export NUM_GPUS=4
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20
# # export CUDA_VISIBLE_DEVICES=0,1,

# # Generate images for evaluation
# # If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
# accelerate launch --main_process_port=65335 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

# # Path to the above generated images
# # guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/eval_dirs/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Calculate RMSE
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}


# # Path to the controlnet weight (can be huggingface or local path)
# # export CONTROLNET_DIR="lllyasviel/control_v11f1p_sd15_depth"  # Eval ControlNet
# export CONTROLNET_DIR="/home/jovyan/konovalova/controlnet_redout/MultiGen/reward_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_scale-1.0_iter10k_fp16_train0-1k_reward0-200_mse-loss_0_reward_all_steps/checkpoint-10000/controlnet"  # Eval our ControlNet++
# # How many GPUs and processes you want to use for evaluation.
# export NUM_GPUS=4
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20
# # export CUDA_VISIBLE_DEVICES=0,1,

# # Generate images for evaluation
# # If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
# # accelerate launch --main_process_port=65335 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

# # Path to the above generated images
# # guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/eval_dirs/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Calculate RMSE
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}



# # Path to the controlnet weight (can be huggingface or local path)
# # export CONTROLNET_DIR="lllyasviel/control_v11f1p_sd15_depth"  # Eval ControlNet
# export CONTROLNET_DIR="/home/jovyan/konovalova/controlnet_redout/MultiGen/reward_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_scale-1.0_iter10k_fp16_train0-1k_reward0-200_mse-loss_all_steps/checkpoint-10000/controlnet"  # Eval our ControlNet++
# # How many GPUs and processes you want to use for evaluation.
# export NUM_GPUS=4
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20
# # export CUDA_VISIBLE_DEVICES=0,1,

# # Generate images for evaluation
# # If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
# # accelerate launch --main_process_port=65335 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

# # Path to the above generated images
# # guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/eval_dirs/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Calculate RMSE
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}



# # Path to the controlnet weight (can be huggingface or local path)
# # export CONTROLNET_DIR="lllyasviel/control_v11f1p_sd15_depth"  # Eval ControlNet
# export CONTROLNET_DIR="/home/jovyan/konovalova/controlnet_redout/MultiGen/reward_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_scale-1.0_iter10k_fp16_train0-1k_reward0-200_mse-loss_another_scale/checkpoint-10000/controlnet"  # Eval our ControlNet++
# # How many GPUs and processes you want to use for evaluation.
# export NUM_GPUS=4
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20
# # export CUDA_VISIBLE_DEVICES=0,1,

# # Generate images for evaluation
# # If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
# # accelerate launch --main_process_port=65335 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

# # Path to the above generated images
# # guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/eval_dirs/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Calculate RMSE
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}



# # Path to the controlnet weight (can be huggingface or local path)
# # export CONTROLNET_DIR="lllyasviel/control_v11f1p_sd15_depth"  # Eval ControlNet
# export CONTROLNET_DIR="/home/jovyan/konovalova/controlnet_redout/MultiGen/reward_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_scale-1.0_iter10k_fp16_train0-1k_reward0-200_mse-loss/checkpoint-10000/controlnet"  # Eval our ControlNet++
# # How many GPUs and processes you want to use for evaluation.
# export NUM_GPUS=4
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20
# # export CUDA_VISIBLE_DEVICES=0,1,

# # Generate images for evaluation
# # If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
# # accelerate launch --main_process_port=65335 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS}

# # Path to the above generated images
# # guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/eval_dirs/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Calculate RMSE
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}