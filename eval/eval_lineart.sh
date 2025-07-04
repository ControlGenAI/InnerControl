# Path to the controlnet weight (can be huggingface or local path)
export CONTROLNET_DIR="/home/jovyan/konovalova/controlnet_redout/reward_lineart_readout_920_new_start_old_weights_ema"  # Eval our ControlNet++

# How many GPUs and processes you want to use for evaluation.
export NUM_GPUS=4
# Guidance scale and inference steps
export SCALE=7.5
export NUM_STEPS=20

# Generate images for evaluation
# If the command is interrupted unexpectedly, just run the code again. We will skip the already generated images.
accelerate launch --main_process_port=12607 --num_processes=$NUM_GPUS eval/eval.py --task_name='lineart' \
                                                                                   --dataset_name='limingcv/MultiGen-20M_canny_eval' \
                                                                                   --dataset_split='validation' \
                                                                                   --condition_column='image' \
                                                                                   --prompt_column='text' \
                                                                                   --model_path $CONTROLNET_DIR \
                                                                                   --guidance_scale=${SCALE} \
                                                                                   --num_inference_steps=${NUM_STEPS} \
                                                                                   --output_dir 'work_dirs/lineart'

# Path to the above generated images
# guidance_scale=7.5, sampling_steps=20 by default
export DATA_DIR="work_dirs/lineart/MultiGen-20M_canny_eval/validation/${CONTROLNET_DIR//\//_}_${SCALE}-${NUM_STEPS}"

#export DATA_DIR='/home/jovyan/shares/SR008.fs2/test/controlnet_redout/work_dirs/lineart_new/MultiGen-20M_canny_eval/validation/_home_jovyan_shares_SR008.fs2_test_controlnet_redout_work_dirs_canny_MultiGen20M_train_reward_lineart_readout_920_new_start_old_weights_ema_7.5-20/'
# Run the evaluation code
python3 eval/eval_edge.py --task lineart --root_dir ${DATA_DIR} --num_processes ${NUM_GPUS}


