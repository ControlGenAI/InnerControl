export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="lllyasviel/control_v11f1p_sd15_depth"
export REWARDMODEL_DIR="Intel/dpt-hybrid-midas"
export OUTPUT_DIR="/home/jovyan/konovalova/controlnet_redout/MultiGen_correct/reward_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_scale-1.0_iter10k_fp16_train0-1k_reward0-200_mse-loss_920_step_seed_666"


accelerate launch --config_file "train/config.yml" \
 --main_process_port=14563 train/reward_control.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="depth" \
 --dataset_name="limingcv/MultiGen-20M_depth" \
 --caption_column="text" \
 --conditioning_image_column="control_depth" \
 --cache_dir=None \
 --resolution=512 \
 --train_batch_size=12 \
 --gradient_accumulation_steps=4 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=8 \
 --max_train_steps=10000 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=10 \
 --checkpointing_steps=1000 \
 --grad_scale=1.0 \
 --use_ema \
 --validation_steps=1000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200 \
 --seed 666 \
 #--resume_from_checkpoint "MultiGen/reward_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_scale-1.0_iter10k_fp16_train0-1k_reward0-200_mse-loss_all_steps/checkpoint-5000"
