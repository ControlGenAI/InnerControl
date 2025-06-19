export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="lllyasviel/control_v11p_sd15_softedge"
export REWARDMODEL_DIR="https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
export OUTPUT_DIR="work_dirs/MultiGen/hed_reward_aligned"


accelerate launch --config_file "train/config.yml" \
 --main_process_port=23256 train/reward_control_aligned.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="hed" \
 --dataset_name="limingcv/MultiGen-20M_depth" \
 --caption_column="text" \
 --conditioning_image_column="hed" \
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
 --grad_scale=1.0 \
 --use_ema \
 --aligning_alpha=1 \
 --validation_steps=1000 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200 \
 --readout_path 'checkpoints/aligning/hed.pt' \
 --max_timestep_readout=920 \
