
# origin controlnet
# python eval/eval_fid.py --real_image_path '../test_depth_im' --generated_image_path 'work_dirs/reproduce/MultiGen-20M_depth_eval/validation/lllyasviel_control_v11f1p_sd15_depth_7.5-20/images/'

# # tuned controlnet
# python eval/eval_fid.py --real_image_path '../test_depth_im' --generated_image_path 'work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_reward_model_MultiGen_controlnet_tuning_checkpoint-10000_controlnet_7.5-20/images/'

# # reward new start
# # python eval/eval_fid.py --real_image_path '../test_depth_im' --generated_image_path 'work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_reproduce_another_starts_checkpoint-10000_controlnet_7.5-20/images/'

# # reward new start
# python eval/eval_fid.py --real_image_path '../test_depth_im' --generated_image_path 'work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_reproduce_another_starts_checkpoint-10000_controlnet_7.5-20/images/'


# reward-readout
python eval/eval_fid.py --real_image_path '../test_depth_im' --generated_image_path 'work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_readout_reproduce_920_checkpoint-10000_controlnet_7.5-20/images/'


# reward readout random seed
python eval/eval_fid.py --real_image_path '../test_depth_im' --generated_image_path 'work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_readout_reproduce_920_random_seed_checkpoint-10000_controlnet_7.5-20/images/'

# reward fix diff 200
python eval/eval_fid.py --real_image_path '../test_depth_im' --generated_image_path 'work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_readout_blured_200_920_42_control_fix_diff_interpolated_checkpoint-10000_controlnet_7.5-20/images/'


# reward fix diff 200 init steps
python eval/eval_fid.py --real_image_path '../test_depth_im' --generated_image_path 'work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_readout_blured_200_920_42_control_fix_diff_interpolated_200_readout_first_timesteps_checkpoint-10000_controlnet_7.5-20/images/'


# reward readout fix diff 200
python eval/eval_fid.py --real_image_path '../test_depth_im' --generated_image_path 'work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_readout_blured_200_920_42_control_fix_diff_interpolated_200_readout_checkpoint-10000_controlnet_7.5-20/images/'


# reward freadout fix diff 200 init steps
python eval/eval_fid.py --real_image_path '../test_depth_im' --generated_image_path 'work_dirs/reproduce/MultiGen-20M_depth_eval/validation/work_dirs_MultiGen_reward_readout_blured_200_920_42_control_fix_diff_interpolated_200_first_timesteps_checkpoint-10000_controlnet_7.5-20/images/'






# python eval/eval_fid.py --real_image_path '/home/jovyan/konovalova/test_depth_im' --generated_image_path '/home/jovyan/konovalova/controlnet_redout/work_dirs/eval_dirs/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_reward_controlnet_sd15_depth_res512_bs256_lr1e-5_warmup100_scale-1.0_iter10k_fp16_train0-1k_reward0-200_mse-loss_new_scale_seed_42_reward_only_920_steps_checkpoint-10000_controlnet_7.5-20/images'