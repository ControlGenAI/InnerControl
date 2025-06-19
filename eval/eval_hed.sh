# Path to the controlnet weight (can be huggingface or local path)

export CONTROLNET_DIR='/home/jovyan/konovalova/controlnet_redout/reward_hed_readout_920_400_new_loss_ema'
# How many GPUs and processes you want to use for evaluation.
export NUM_GPUS=4
# Guidance scale and inference steps
export SCALE=7.5
export NUM_STEPS=20

# Generate images for evaluation
# # If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
accelerate launch --main_process_port=22509 --num_processes=$NUM_GPUS eval/eval.py --task_name='hed' \
                                                                                   --dataset_name='limingcv/MultiGen-20M_canny_eval' \
                                                                                   --dataset_split='validation' \
                                                                                   --condition_column='image' \
                                                                                   --prompt_column='text' \
                                                                                   --model_path $CONTROLNET_DIR \
                                                                                   --guidance_scale=${SCALE} \
                                                                                   --num_inference_steps=${NUM_STEPS} \
                                                                                   --output_dir 'work_dirs/hed'

# Path to the above generated images
# guidance_scale=7.5, sampling_steps=20 by default
export DATA_DIR="work_dirs/hed/MultiGen-20M_canny_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

# Run the evaluation code
python3 eval/eval_edge.py --task hed --root_dir ${DATA_DIR} --num_processes ${NUM_GPUS}
