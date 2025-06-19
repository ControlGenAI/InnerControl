export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="lllyasviel/control_v11f1p_sd15_depth"
export REWARDMODEL_DIR="Intel/dpt-hybrid-midas"
export OUTPUT_DIR="work_dirs/reward_model/MultiGen/test"

accelerate launch --config_file "train/config.yml" \
 --main_process_port=23256 train/train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="limingcv/MultiGen-20M_depth" \
 --caption_column="text" \
 --conditioning_image_column="control_depth" \
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
 --validation_steps=5000 \
