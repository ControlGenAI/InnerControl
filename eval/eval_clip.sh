# # reward
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_reward_only_checkpoint-1000_controlnet_7.5-20/images/'

# # new loss 1
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_new_loss_checkpoint-1000_controlnet_7.5-20/images/'


# # new loss 2
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_new_loss_2_checkpoint-1000_controlnet_7.5-20/images/'


# # new loss both
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_new_loss_1_checkpoint-1000_controlnet_7.5-20/images/'


# # depth + edges
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_edges_depth_control_checkpoint-1000_controlnet_7.5-20/images/'



# # edges
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_edges_control_checkpoint-1000_controlnet_7.5-20/images/'


# # null
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_readout_zero_tensor_null_0.4_checkpoint-1000_controlnet_7.5-20/images/'

# # null readout
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/_home_jovyan_konovalova_controlnet_redout_MultiGen_small_tests_0_600_1e-5_readout_zero_tensor_null_only_readout_augment_checkpoint-1000_controlnet_7.5-20/images/'


# # blur
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_init_gauss_gauss_augmentation_checkpoint-1000_controlnet_7.5-20/images/'


# # noise
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_init_gauss_noise_augmentation_checkpoint-1000_controlnet_7.5-20/images/'

# # noise + blur
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_init_gauss_blur_augmentation_checkpoint-1000_controlnet_7.5-20/images/'


# # lpips
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_lpips_depth_checkpoint-1000_controlnet_7.5-20/images/'

# # cosine
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_cosine_depth_checkpoint-1000_controlnet_7.5-20/images/'


# # unet consistency
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_unet_consistency_checkpoint-1000_controlnet_7.5-20/images/'


# # origin
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/smal_test/MultiGen-20M_depth_eval/validation/weights_controlnet__7.5-20/images/'

# # only readout
# python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_checkpoint-1000_controlnet_7.5-20/images/'

# consistent permuted
python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_permuted_consistency_0.4_checkpoint-1000_controlnet_7.5-20/images'


# permuted
python eval/eval_clip.py --generated_image_dir '/home/jovyan/konovalova/controlnet_redout/work_dirs/small_test/MultiGen-20M_depth_eval/validation/MultiGen_small_tests_0_600_1e-5_readout_permuted_0.4_checkpoint-1000_controlnet_7.5-20/images'


