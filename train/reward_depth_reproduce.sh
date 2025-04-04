export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="lllyasviel/control_v11f1p_sd15_depth"
export REWARDMODEL_DIR="Intel/dpt-large"
export OUTPUT_DIR="work_dirs/MultiGen/reward_readout_920_test"
export HF_HUB_CACHE='/workspace-SR008.fs2/test/datasets'


accelerate launch --config_file "train/config.yml" \
 --main_process_port=23256 train/reward_control.py \
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
 --readout_alpha=1 \
 --readout_beta=3 \
 --seed 3407 \
 --validation_steps=1000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200 \
 --readout_path '/workspace-SR008.fs2/test/controlnet_redout/weights/checkpoint_step_5000.pt' \
 --max_timestep_readout=920 \
 --resume_from_checkpoint '/workspace-SR008.fs2/test/controlnet_redout/work_dirs/MultiGen/reward_readout_920_seed_3407/checkpoint-3000'
