
# 0_600_1e-5_readout_init_gauss_gauss_augmentation + 
# 0_600_1e-5_readout_init_gauss_blur_augmentation + 
# 0_600_1e-5_readout_init_gauss_noise_augmentation + 
# 0_600_1e-5_readout_cosine_depth + 
# 0_600_1e-5_readout_lpips_depth + 
# 0_600_1e-5_readout_zero_tensor_null_0.4 +
# 0_600_1e-5_readout_zero_tensor_null_only_readout_augment + 
# 0_600_1e-5_new_loss + 
# 0_600_1e-5_new_loss_1 + 
# 0_600_1e-5_new_loss_2 +
# 0_600_1e-5_reward_only + 
# 0_600_1e-5_readout_edges_depth_control + 
# 0_600_1e-5_readout_edges_control + 

# 0_600_1e-5_reward_only + 
# weights_controlnet +
# 0_600_1e-5_readout_unet_consistency + 
# 0_600_1e-5 +


# export CONTROLNET_DIR="/workspace-SR008.fs2/test/controlnet_redout/work_dirs/MultiGen/reward_readout_reproduce_920/checkpoint-3000/controlnet"  # Eval our ControlNet++
# # How many PUs and processes you want to use for evaluation.
# export NUM_GPUS=8
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20

# # Generate images for evaluation
# # If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
# #accelerate launch --main_process_port=12668 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --output_dir "work_dirs/reproduce/"

# # Path to the above generated images
# # guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/reproduce/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Calculate RMSE
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}


export CONTROLNET_DIR="/workspace-SR008.fs2/test/controlnet_redout/work_dirs/MultiGen/reward_readout_920_seed_3407/checkpoint-10000/controlnet"  # Eval our ControlNet++
# How many PUs and processes you want to use for evaluation.
export NUM_GPUS=8
# Guidance scale and inference steps
export SCALE=7.5
export NUM_STEPS=20

# Generate images for evaluation
# If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
accelerate launch --main_process_port=12668 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --output_dir "work_dirs/reproduce/"

# Path to the above generated images
# guidance_scale=7.5, sampling_steps=20 by default
export DATA_DIR="work_dirs/reproduce/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}


export CONTROLNET_DIR="/workspace-SR008.fs2/test/controlnet_redout/work_dirs/MultiGen/reward_readout_920_seed_3407/checkpoint-3000/controlnet"  # Eval our ControlNet++
# How many PUs and processes you want to use for evaluation.
export NUM_GPUS=8
# Guidance scale and inference steps
export SCALE=7.5
export NUM_STEPS=20

# Generate images for evaluation
# If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
#accelerate launch --main_process_port=12668 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --output_dir "work_dirs/reproduce/"

# Path to the above generated images
# guidance_scale=7.5, sampling_steps=20 by default
export DATA_DIR="work_dirs/reproduce/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}


# export CONTROLNET_DIR="work_dirs/MultiGen/reward_readout_reproduce_920/checkpoint-10000/controlnet"  # Eval our ControlNet++
# # How many PUs and processes you want to use for evaluation.
# export NUM_GPUS=8
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20

# # Generate images for evaluation
# # If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
# #accelerate launch --main_process_port=12668 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --output_dir "work_dirs/reproduce/"

# # Path to the above generated images
# # guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/reproduce/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Calculate RMSE
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export CONTROLNET_DIR="work_dirs/reward_model/MultiGen/controlnet_tuning/checkpoint-10000/controlnet"  # Eval our ControlNet++
# # How many GPUs and processes you want to use for evaluation.
# export NUM_GPUS=8
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20

# # Generate images for evaluation
# # If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
# #accelerate launch --main_process_port=12668 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --output_dir "work_dirs/reproduce/"

# # Path to the above generated images
# # guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/reproduce/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Calculate RMSE
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export CONTROLNET_DIR="lllyasviel/control_v11f1p_sd15_depth"  # Eval our ControlNet++
# # How many GPUs and processes you want to use for evaluation.
# export NUM_GPUS=8
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20

# # Generate images for evaluation
# # If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
# #accelerate launch --main_process_port=12668 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --output_dir "work_dirs/reproduce/"

# # Path to the above generated images
# # guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/reproduce/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Calculate RMSE
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export CONTROLNET_DIR="work_dirs/MultiGen/reward_readout_permuted/checkpoint-10000/controlnet"  # Eval our ControlNet++
# # How many GPUs and processes you want to use for evaluation.
# export NUM_GPUS=8
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20

# # Generate images for evaluation
# # If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
# #accelerate launch --main_process_port=12669 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --output_dir "work_dirs/reproduce/"

# # Path to the above generated images
# # guidance_scale=7.5, sampling_steps=20 by default
# export DATA_DIR="work_dirs/reproduce/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# # Calculate RMSE
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}


####################################################################

# export DATA_DIR="/home/jovyan/shares/SR008.fs2/test/controlnet_redout/work_dirs/reproduce/MultiGen-20M_depth_eval/validation/lllyasviel_control_v11f1p_sd15_depth_7.5-20"

# # Calculate RMSE
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

export DATA_DIR="/home/jovyan/shares/SR008.fs2/test/controlnet_redout/work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_readout_reproduce_checkpoint-10000_controlnet_7.5-20"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}

export DATA_DIR="/home/jovyan/shares/SR008.fs2/test/controlnet_redout/work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_readout_reproduce_another_seed_checkpoint-10000_controlnet_7.5-20"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}


export DATA_DIR="/home/jovyan/shares/SR008.fs2/test/controlnet_redout/work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_reward_model_MultiGen_controlnet_tuning_checkpoint-10000_controlnet_7.5-20"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}

export DATA_DIR="/home/jovyan/shares/SR008.fs2/test/controlnet_redout/work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_reproduce_another_starts_checkpoint-10000_controlnet_7.5-20"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}


export DATA_DIR="/home/jovyan/shares/SR008.fs2/test/controlnet_redout/work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_readout_reproduce_920_checkpoint-10000_controlnet_7.5-20"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}


export DATA_DIR="/home/jovyan/shares/SR008.fs2/test/controlnet_redout/work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_readout_permuted_checkpoint-10000_controlnet_7.5-20"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}



export DATA_DIR="/home/jovyan/shares/SR008.fs2/test/controlnet_redout/work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_readout_train_920_500_checkpoint-10000_controlnet_7.5-20"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}

export DATA_DIR="/home/jovyan/shares/SR008.fs2/test/controlnet_redout/work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_readout_train_920_300_1_2_checkpoint-10000_controlnet_7.5-20"

# Calculate RMSE
python3 eval/eval_depth.py --root_dir ${DATA_DIR}


# export CONTROLNET_DIR="work_dirs/MultiGen/reward_reproduce/checkpoint-10000/controlnet" # Eval our ControlNet++
# # How many GPUs and processes you want to use for evaluation.
# # export CONTROLNET_DIR="weights/controlnet"
# export NUM_GPUS=1
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20
# export CUDA_VISIBLE_DEVICES=1,

# accelerate launch --main_process_port=13156 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --output_dir "work_dirs/reproduce" 

# export DATA_DIR="work_dirs/reproduce/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export CONTROLNET_DIR="MultiGen_small_tests/0_600_1e-5_readout_only_reward_200/checkpoint-1000/controlnet"  # Eval our ControlNet++
# # How many GPUs and processes you want to use for evaluation.
# # export CONTROLNET_DIR="weights/controlnet"
# export NUM_GPUS=1
# # Guidance scale and inference steps
# export SCALE=7.5
# export NUM_STEPS=20
# export CUDA_VISIBLE_DEVICES=0,

# accelerate launch --main_process_port=12456 --num_processes=$NUM_GPUS eval/eval.py --task_name='depth' --dataset_name='limingcv/MultiGen-20M_depth_eval' --dataset_split='validation' --condition_column='control_depth' --label_column='control_depth' --prompt_column='text'  --model_path $CONTROLNET_DIR --guidance_scale=${SCALE} --num_inference_steps=${NUM_STEPS} --output_dir "work_dirs/small_test" 

# export DATA_DIR="work_dirs/small_test/MultiGen-20M_depth_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR="work_dirs/eval_dirs/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_reward_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_scale-1.0_iter10k_fp16_train0-1k_reward0-200_mse-loss_new_scale_seed_42_reward_only_920_steps_checkpoint-10000_controlnet_7.5-20/"
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR="work_dirs/eval_dirs_1/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_reward_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_scale-1.0_iter10k_fp16_train0-1k_reward0-200_mse-loss_new_scale_checkpoint-10000_controlnet_7.5-20/"
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR="/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_readout_zero_tensor_null_only_readout_augment_checkpoint-1000_controlnet_7.5-20"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}
# _home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_readout_zero_tensor_null_only_readout_augment_checkpoint-1000_controlnet_7.5-20
# export DATA_DIR="/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_readout_zero_tensor_null_0.4_checkpoint-1000_controlnet_7.5-20"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR="/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_readout_zero_tensor_augmentation_checkpoint-1000_controlnet_7.5-20"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR="/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_readout_zero_tensor_consistency_checkpoint-1000_controlnet_7.5-20"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR="/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_reward_only_checkpoint-1000_controlnet_7.5-20"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR="/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_new_loss_only_1_checkpoint-1000_controlnet_7.5-20"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}


# export DATA_DIR="/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_new_loss_2_checkpoint-1000_controlnet_7.5-20"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR="/home/jovyan/konovalova/controlnet_redout/work_dirs/eval_dirs/MultiGen-20M_depth_eval/validation/weights_controlnet_7.5-20"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR="work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_200_800_1e-5_checkpoint-1000_controlnet_7.5-20/"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}



# export DATA_DIR="work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_200_600_1e-5_checkpoint-1000_controlnet_7.5-20/"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR="work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_300_700_1e-5_checkpoint-1000_controlnet_7.5-20/"

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}


# export DATA_DIR="work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_4/"


# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # # Calculate RMSE
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_Mu'
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_500_1e-5_checkpoint-1000_controlnet_7.5-20'
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_100_920_1e-5_checkpoint-1000_controlnet_7.5-20'
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_300_920_1e-5_checkpoint-1000_controlnet_7.5-20'
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_500_920_1e-5_checkpoint-1000_controlnet_7.5-20'
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_700_920_1e-5_checkpoint-1000_controlnet_7.5-20'
# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

##########################

# 0_600_1e-5_readout_init_gauss_gauss_augmentation + +
# 0_600_1e-5_readout_init_gauss_blur_augmentation + +
# 0_600_1e-5_readout_init_gauss_noise_augmentation + +
# 0_600_1e-5_readout_cosine_depth + +
# 0_600_1e-5_readout_lpips_depth + +
# 0_600_1e-5_readout_zero_tensor_null_0.4 + +
# 0_600_1e-5_readout_zero_tensor_null_only_readout_augment + +
# 0_600_1e-5_new_loss + +
# 0_600_1e-5_new_loss_1 + +
# 0_600_1e-5_new_loss_2 + +
# 0_600_1e-5_readout_edges_depth_control + +
# 0_600_1e-5_readout_edges_control + + 

# 0_600_1e-5_reward_only + + 
# weights_controlnet + +
# 0_600_1e-5_readout_unet_consistency + +
# 0_600_1e-5 + +


# # reward 
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_reward_only_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # new loss 1
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_new_loss_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # new loss 2
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_new_loss_2_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # new loss both 
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_new_loss_1_checkpoint-1000_controlnet_7.5-20'


# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # depth + edges
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_edges_depth_control_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # edges 
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_edges_control_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # null
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_readout_zero_tensor_null_0.4_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # null readout
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_readout_zero_tensor_null_only_readout_augment_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # blur
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_init_gauss_gauss_augmentation_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # noise
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_init_gauss_noise_augmentation_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # noise + gauss
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_init_gauss_blur_augmentation_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # lpips
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_lpips_depth_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # cosine
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_cosine_depth_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # unet consistency
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_unet_consistency_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# origin
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/smal_test/MultiGen-20M_depth_eval/validation/weights_controlnet__7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# only readout
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}


# # permuted
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_permuted_0.4_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# # consistent
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_permuted_consistency_0.4_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_readout_trained_700_steps_2_0_control_checkpoint-2000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}



# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_readout_trained_700_steps_2_0_unet_checkpoint-2000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}



# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_readout_trained_1000_steps_2_checkpoint-2000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}



# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_readout_trained_700_steps_2_checkpoint-2000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_readout_trained_600_2_checkpoint-2000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}


# ###############

# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_readout_trained_700_steps_2_0_control_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}



# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_readout_trained_700_steps_2_0_unet_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}



# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_readout_trained_1000_steps_2_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}



# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_readout_trained_700_steps_2_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}

# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_readout_trained_600_2_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}



# # consistent
# export DATA_DIR='/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_permuted_consistency_0.8_300_steps_checkpoint-1000_controlnet_7.5-20'

# python3 eval/eval_depth.py --root_dir ${DATA_DIR}


    # ,
    # ,
    # ,
    # ,
    #  ,
    #  ,
    #  ,
    #  ,
    #  ,
    #  ,
    #  ,
