# [ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback (ECCV 2024)](https://liming-ai.github.io/ControlNet_Plus_Plus/)

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2404.07987-b31b1b.svg)](https://arxiv.org/abs/2404.07987)&nbsp;
[![huggingface demo](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-ControlNet++-yellow)](https://huggingface.co/spaces/limingcv/ControlNet-Plus-Plus)&nbsp;

</div>

## My experiments

### Only reward run

To run only reward

```
bash reward_depth_ablate.sh/reward_canny_ablate.sh
```

Нужно только поменять на другой датасет и задачу, а также посмотреть в статье, какое значение у `--grad_scale` для нужного эксперимента. Для вариации шагов, на которых используется reward ` --min_timestep_rewarding=0`, `--max_timestep_rewarding=200`


### Reward + readout
Для начала нужно скачать отсюда https://github.com/google-research/readout_guidance модельку риадут (для depth/edge). 

```
bash reward_depth_reproduce.sh/ bash reward_canny_reproduce.sh
```
тут надо обратить внимание на следующие параметры:

`--readout_path '/workspace-SR008.fs2/test/controlnet_redout/weights/checkpoint_step_5000.pt'`, `--max_timestep_readout=920`, `--readout_alpha`

Кусок в обучении начинается с комментария 

```
"""
Rewarding ControlNet each t
"""
```
Из того, что еще важно, кажется для edge нужно поставить `--normalize False`, потому что у нас глубина ап-ту скейл вычисляется, а edges нет. А еще надо `--readout_type 'conv'`, потому что я отдельно тестировала ридаут для глубины на аттншнах, но для edges пока только на свёртках есть.

## 🕹️ Environments
```bash
git clone https://github.com/liming-ai/ControlNet_Plus_Plus.git
pip3 install -r requirements.txt
pip3 install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
pip3 install "mmsegmentation>=1.0.0"
pip3 install mmdet
pip3 install clean-fid
pip3 install torchmetrics
```

## 🕹️ Data Preperation
**All the organized data has been put on Huggingface and will be automatically downloaded during training or evaluation.** You can preview it in advance to check the data samples and disk space occupied with following links.
|   Task    | Training Data 🤗 | Evaluation Data 🤗 |
|:----------:|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
|  LineArt, Hed, Canny   | [Data](https://huggingface.co/datasets/limingcv/MultiGen-20M_train), 1.14 TB | [Data](https://huggingface.co/datasets/limingcv/MultiGen-20M_canny_eval), 2.25GB |
|  Depth   |  [Data](https://huggingface.co/datasets/limingcv/MultiGen-20M_depth), 1.22 TB | [Data](https://huggingface.co/datasets/limingcv/MultiGen-20M_depth_eval), 2.17GB |
|  Segmentation ADE20K   | [Data](https://huggingface.co/datasets/limingcv/Captioned_ADE20K), 7.04 GB | Same Path as Training Data |
|  Segmentation COCOStuff   | [Data](https://huggingface.co/datasets/limingcv/Captioned_COCOStuff), 61.9 GB | Same Path as Training Data |


## 🕹️ Training
By default, our training is based on 8 A100-80G GPUs. If your computational resources are insufficient for training, you may need to reduce the batch size and increase gradient accumulation at the same time, and we have not observed any performance degradation. Reducing the training resolution will result in performance degradation.

### For segmentation task
[ControlNet V1.1 Seg](https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/README.md#controlnet-11-segmentation) is trained on both ADE20K and COCOStuff, and these two datasets have different masks. To this end, we first perform normal model fine-tuning on each dataset, and then perform reward fine-tuning.
```bash
# Please refer to the reward script for details
bash train/reward_ade20k.sh
bash train/reward_cocostuff.sh
```

### For other tasks
We can directly perform reward fine-tuning.
```bash
bash train/reward_canny.sh
bash train/reward_depth.sh
bash train/reward_hed.sh
bash train/reward_linedrawing.sh
```

### Core Code
Please refer to the [core code here](https://github.com/liming-ai/ControlNet_Plus_Plus/blob/9167f0d85ccc5ad1eb9a83f3e7fa8d3422d5d9d5/train/reward_control.py#L1429), in summary:
#### Step 1: Predict the single-step denoised RGB image with noise sampler:
```python
# Predict the single-step denoised latents
pred_original_sample = [
    noise_scheduler.step(noise, t, noisy_latent).pred_original_sample.to(weight_dtype) \
        for (noise, t, noisy_latent) in zip(model_pred, timesteps, noisy_latents)
]
pred_original_sample = torch.stack(pred_original_sample)

# Map the denoised latents into RGB images
pred_original_sample = 1 / vae.config.scaling_factor * pred_original_sample
image = vae.decode(pred_original_sample.to(weight_dtype)).sample
image = (image / 2 + 0.5).clamp(0, 1)
```
#### Step 2: Normalize the single-step denoised images according to different reward models
```python
# The normalization depends on different reward models.
if args.task_name == 'depth':
    image = torchvision.transforms.functional.resize(image, (384, 384))
    image = normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
elif args.task_name in ['canny', 'lineart', 'hed']:
    pass
else:
    image = normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
```
#### Step 3: Apply both diffusion training loss and reward loss:
```python
# reward model inference
if args.task_name == 'canny':
    outputs = reward_model(image.to(accelerator.device), low_threshold, high_threshold)
else:
    outputs = reward_model(image.to(accelerator.device))

# Determine which samples in the current batch need to calculate reward loss
timestep_mask = (args.min_timestep_rewarding <= timesteps.reshape(-1, 1)) & (timesteps.reshape(-1, 1) <= args.max_timestep_rewarding)

# Calculate reward loss
reward_loss = get_reward_loss(outputs, labels, args.task_name, reduction='none')

# Calculate final loss
reward_loss = reward_loss.reshape_as(timestep_mask)
reward_loss = (timestep_mask * reward_loss).sum() / (timestep_mask.sum() + 1e-10)
loss = pretrain_loss + reward_loss * args.grad_scale
```

## 🕹️ Evaluation
### Checkpoints Preparation
Please download the model weights and put them into each subset of `checkpoints`:
|   model    |HF weights🤗                                                                        |
|:----------:|:------------------------------------------------------------------------------------|
|  LineArt   | [model](https://huggingface.co/limingcv/reward_controlnet/tree/main/checkpoints/lineart) |
|  Depth   |  [model](https://huggingface.co/limingcv/reward_controlnet/tree/main/checkpoints/depth) |
|  Hed (SoftEdge)   | [model](https://huggingface.co/limingcv/reward_controlnet/tree/main/checkpoints/hed) |
| Canny | [model](https://huggingface.co/limingcv/reward_controlnet/tree/main/checkpoints/canny) |
|  Segmentation (ADE20K)   | [UperNet-R50](https://huggingface.co/limingcv/reward_controlnet/tree/main/checkpoints/ade20k_reward-model-UperNet-R50/checkpoint-5000/controlnet), [FCN-R101](https://huggingface.co/limingcv/reward_controlnet/tree/main/checkpoints/ade20k_reward-model-FCN-R101-d8/checkpoint-5000/controlnet) |
| Segmentation (COCOStuff) | [model](https://huggingface.co/limingcv/reward_controlnet/tree/main/checkpoints/cocostuff/reward_5k) |

### Evaluate Controllability
Please make sure the folder directory is consistent with the test script, then you can eval each model by:
```bash
bash eval/eval_ade20k.sh
bash eval/eval_cocostuff.sh
bash eval/eval_canny.sh
bash eval/eval_depth.sh
bash eval/eval_hed.sh
bash eval/eval_linedrawing.sh
```

*The segmentation mIoU results of ControlNet and ControlNet++ in the arXiv v1 version of the paper were tested using images and labels saved in `.jpg` format, which resulted in errors. We retested and reported the results using images and labels saved in `.png` format, please refer to our latest arXiv and ECCV Camera Ready releases. Other comparison methods (Gligen/T2I-Adapter/UniControl/UniControlNet) and other evaluation metrics (FID/CLIP-score) were not affected by this error.*


### Evaluate CLIP-Score and FID
Please refer to the code for evaluating [CLIP-Score](eval/eval_clip.py) and [FID](eval/eval_fid.py)


## 🕹️ Inference
Please refer to the [Inference Branch](https://github.com/liming-ai/ControlNet_Plus_Plus/tree/inference) or try our [online Huggingface demo](https://huggingface.co/spaces/limingcv/ControlNet-Plus-Plus)


## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.txt) file for details.

## 🙏 Acknowledgements
We sincerely thank the [Huggingface](https://huggingface.co), [ControlNet](https://github.com/lllyasviel/ControlNet), [OpenMMLab](https://github.com/open-mmlab) and [ImageReward](https://github.com/THUDM/ImageReward) communities for their open source code and contributions. Our project would not be possible without these amazing works.

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@inproceedings{controlnet_plus_plus,
    author    = {Ming Li and Taojiannan Yang and Huafeng Kuang and Jie Wu and Zhaoning Wang and Xuefeng Xiao and Chen Chen},
    title     = {ControlNet $$++ $$: Improving Conditional Controls with Efficient Consistency Feedback},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2024},
}
```
