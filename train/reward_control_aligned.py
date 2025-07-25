#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import os
# Set the W&B mode to offline
os.environ['WANDB_MODE'] = 'offline'
import math
import time
import kornia
import pickle
import random
import logging
import argparse
from omegaconf import OmegaConf
import itertools
from huggingface_hub import snapshot_download

import torch
import numpy as np
import accelerate
import transformers
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk
from huggingface_hub import create_repo, upload_folder
from transformers import AutoTokenizer, PretrainedConfig

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from packaging import version
from torchvision import transforms
from torch.cuda.amp import autocast
from torchvision.transforms.functional import normalize


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.Resampling.BICUBIC

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from utils import image_grid, get_reward_model, get_reward_loss, label_transform, group_random_crop

from helpers import *
import os

from PIL import PngImagePlugin
MaximumDecompressedsize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedsize * MegaByte
Image.MAX_IMAGE_PIXELS = None

os.environ['HF_DATASETS_OFFLINE ']= "1"

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__)

# Offloading state_dict to CPU to avoid GPU memory boom (only used for FSDP training)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

def make_train_dataset(args, tokenizer, accelerator, split='train'):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # dataset = load_dataset(
        #     args.dataset_name,
        #     args.dataset_config_name,
        #     cache_dir=args.cache_dir,
        #     keep_in_memory=args.keep_in_memory,
        # )
        if args.dataset_name.count('/') == 1:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                keep_in_memory=args.keep_in_memory,
            )
        else:
            dataset = load_from_disk(
                dataset_path=args.dataset_name,
                keep_in_memory=args.keep_in_memory,
            )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
                keep_in_memory=args.keep_in_memory,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # filter wrong files for MultiGen
    if args.wrong_ids_file is not None:
        all_idx = [i for i in range(len(dataset['train']))]
        exclude_idx = pickle.load(open(args.wrong_ids_file, 'rb'))
        correct_idx = [item for item in all_idx if item not in exclude_idx]
        dataset['train'] = dataset['train'].select(correct_idx)
        print(f"filtering {len(exclude_idx)} rows")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset[split].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    elif args.conditioning_image_column in ['lineart', 'hed']:
        conditioning_image_column = image_column
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    resolution = (args.resolution, args.resolution)
    image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    label_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST, antialias=True),
            # transforms.CenterCrop(args.resolution),
        ]
    )

    def preprocess_train(examples):
        pil_images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in pil_images]

        if args.conditioning_image_column in ['lineart', 'hed']:
            conditioning_images = images
        else:
            conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
            conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        if args.label_column is not None:
            dtype = torch.long
            labels = [torch.tensor(np.asarray(label), dtype=dtype).unsqueeze(0) for label in examples[args.label_column]]
            labels = [label_image_transforms(label) for label in labels]

        # perform groupped random crop for image/conditioning_image/label
        if args.label_column is not None:
            grouped_data = [torch.cat([x, y, z]) for (x, y, z) in zip(images, conditioning_images, labels)]
            grouped_data = group_random_crop(grouped_data, args.resolution)

            images = [x[:3, :, :] for x in grouped_data]
            conditioning_images = [x[3:6, :, :] for x in grouped_data]
            labels = [x[6:, :, :] for x in grouped_data]

            examples[args.label_column] = labels
        else:
            grouped_data = [torch.cat([x, y]) for (x, y) in zip(images, conditioning_images)]
            grouped_data = group_random_crop(grouped_data, args.resolution)

            images = [x[:3, :, :] for x in grouped_data]
            conditioning_images = [x[3:, :, :] for x in grouped_data]

        # Dropout some of features for classifier-free guidance.
        for i, img_condition in enumerate(conditioning_images):
            rand_num = random.random()
            if rand_num < args.image_condition_dropout:
                conditioning_images[i] = torch.zeros_like(img_condition)
            elif rand_num < args.image_condition_dropout + args.text_condition_dropout:
                examples[caption_column][i] = ""
            elif rand_num < args.image_condition_dropout + args.text_condition_dropout + args.all_condition_dropout:
                conditioning_images[i] = torch.zeros_like(img_condition)
                examples[caption_column][i] = ""
            # if random.random() < args.image_conditioning_augmentation:
            #     assert False
            #     noise = torch.randn_like(img_condition) * args.noise_std
            #     conditioning_images[i] += noise

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed)
            # rewrite the shuffled dataset on disk as contiguous chunks of data
            dataset["train"] = dataset["train"].flatten_indices()
            dataset["train"] = dataset["train"].select(range(args.max_train_samples))

        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return dataset, train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    if args.label_column is not None:
        labels = torch.stack([example[args.label_column] for example in examples])
        labels = labels.to(memory_format=torch.contiguous_format).float()
    else:
        labels = conditioning_pixel_values

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "labels": labels,
    }


def log_validation(
        vae,
        text_encoder,
        tokenizer,
        unet,
        controlnet,
        ema_controlnet,
        args,
        accelerator,
        weight_dtype,
        step,
        val_dataset):

    # randomly select some samples to log
    if val_dataset is not None:
        val_dataset = val_dataset.select(
            random.sample(range(len(val_dataset)), args.max_val_samples)
        )

    if args.task_name in ['lineart', 'hed']:
        reward_model = get_reward_model(args.task_name, args.reward_model_name_or_path)
        reward_model.to(accelerator.device)
        reward_model.eval()
    else:
        reward_model = None

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_column = args.image_column
    caption_column = args.caption_column

    if args.conditioning_image_column in ['lineart', 'hed']:
        conditioning_image_column = image_column
    else:
        conditioning_image_column = args.conditioning_image_column

    assert val_dataset is not None, "Validation dataset is required for logging validation images."
    try:
        validation_images = val_dataset[image_column]
        validation_conditions = val_dataset[conditioning_image_column]
        validation_prompts = val_dataset[caption_column]
    except:
        validation_images = [item[image_column] for item in val_dataset]
        validation_conditions = [item[conditioning_image_column] for item in val_dataset]
        validation_prompts = [item[caption_column] for item in val_dataset]

    # Avoid some problems caused by grayscale images
    validation_conditions = [x.convert('RGB') for x in validation_conditions]

    if args.conditioning_image_column in ['lineart', 'hed']:
        with autocast():
            validation_conditions = [torchvision.transforms.functional.pil_to_tensor(x) for x in validation_conditions]
            validation_conditions = [x / 255. for x in validation_conditions]
            validation_conditions = [torchvision.transforms.functional.resize(x, (512, 512), interpolation=Image.BICUBIC) for x in validation_conditions]
            with torch.no_grad():
                validation_conditions = reward_model(torch.stack(validation_conditions).to(accelerator.device))
            validation_conditions = 1 - validation_conditions if args.task_name == 'lineart' else validation_conditions
            validation_conditions = torch.chunk(validation_conditions, len(validation_conditions), dim=0)
            validation_conditions = [torchvision.transforms.functional.to_pil_image(x.squeeze(0), 'L') for x in validation_conditions]

    image_logs = []

    logger.info(f"Running validation with {len(validation_prompts)} prompts... ")
    for validation_prompt, validation_condition, validation_image in zip(validation_prompts, validation_conditions, validation_images):
        if val_dataset is not None:
            validation_image = validation_image.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
            validation_condition = validation_condition.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
        else:
            validation_condition = Image.open(validation_condition).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)

        with torch.autocast("cuda"):
            images = pipeline(
                [validation_prompt] * args.num_validation_images,
                [validation_condition] * args.num_validation_images,
                num_inference_steps=20,
                generator=generator
            ).images

        image_logs.append({
            "validation_image": validation_image,
            "validation_condition": validation_condition,
            "validation_prompt": validation_prompt,
            "images": images,
            'ema_images': []
        })

    if args.use_ema:
        # Store the ControlNet parameters temporarily and load the EMA parameters to perform inference.
        ema_controlnet.copy_to(controlnet.parameters())

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            safety_checker=None,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        logger.info(f"Running validation with {len(validation_prompts)} prompts... ")
        for idx, (validation_prompt, validation_condition, validation_image) in enumerate(zip(validation_prompts, validation_conditions, validation_images)):
            if val_dataset is not None:
                validation_condition = validation_condition.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
            else:
                validation_condition = Image.open(validation_condition).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)

            with torch.autocast("cuda"):
                images = pipeline(
                    [validation_prompt] * args.num_validation_images,
                    [validation_condition] * args.num_validation_images,
                    num_inference_steps=20,
                    generator=generator
                ).images

            image_logs[idx]['ema_images'] = images


    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                ema_images = log["ema_images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                validation_condition = log["validation_condition"]

                validation_prompt = validation_prompt + ['EMA'] * len(validation_prompt)

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                for image in ema_images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                images = log["images"]
                ema_images = log["ema_images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                validation_condition = log["validation_condition"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet input image"))
                formatted_images.append(wandb.Image(validation_condition, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

                for image in ema_images:
                    image = wandb.Image(image, caption='EMA')
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        if reward_model is not None:
            reward_model = None

        return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--reward_model_name_or_path",
        type=str,
        default=None,
        help="Path to reward model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--grad_scale", type=float, default=1, help="Scale divided for grad loss value."
    )
    parser.add_argument(
        "--reward_alpha", type=float, default=1, help="Scale divided for grad loss value."
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
 
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--aligning_alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--readout_beta",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--control_guidance_start",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--control_guidance_end",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default='limingcv/reward_controlnet',
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--timestep_sampling_start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--timestep_sampling_end",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--min_timestep_rewarding",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_timestep_rewarding",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--min_timestep_readout",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_timestep_readout",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default='segmentation',
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--keep_in_memory",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--wrong_ids_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="The column of the dataset containing the original labels. `seg_map` for ADE20K; `panoptic_seg_map` for COCO-Stuff.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=10,
        help=(
            "Max number of samples for validation during training, default to 10"
        ),
    )
    parser.add_argument(
        "--image_condition_dropout",
        type=float,
        default=0,
        help="Probability of image conditions to be replaced with tensors with zero value. Defaults to 0.",
    )
    parser.add_argument(
        "--text_condition_dropout",
        type=float,
        default=0,
        help="Probability of image prompts to be replaced with empty strings. Defaults to 0.05.",
    )
    parser.add_argument(
        "--all_condition_dropout",
        type=float,
        default=0,
        help="Probability of abandon all the conditions.",
    )
    parser.add_argument(
        "--image_conditioning_augmentation",
        type=float,
        default=0,
        help="Probability of abandon all the conditions.",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.1,
        help="Probability of abandon all the conditions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--optimize_readout",
        type=bool,
        default=False,
        help="If additionally optimize readout models",
    )
    parser.add_argument(
        "--readout_path",
        type=str,
        default='/home/jovyan/konovalova/controlnet_redout/weights/checkpoint_step_5000.pt',
        help="Path to pretrained reaout model",
    )
    parser.add_argument(
        "--readout_type",
        type=str,
        default='attn',
        choices=['attn', 'conv'],
        help="Model is based on attn or conv",
    )
    parser.add_argument(
        "--normalize",
        type=bool,
        default=True,
        help="If additionally normalize readout heads",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="reward_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.text_condition_dropout < 0 or args.text_condition_dropout > 1:
        raise ValueError("`--text_condition_dropout` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def make_train_dataset(args, tokenizer, accelerator, split='train'):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:

        if args.dataset_name.count('/') == 1:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                keep_in_memory=args.keep_in_memory,
            )
        else:
            dataset = load_from_disk(
                dataset_path=args.dataset_name,
                keep_in_memory=args.keep_in_memory,
            )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
                keep_in_memory=args.keep_in_memory,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # filter wrong files for MultiGen
    if args.wrong_ids_file is not None:
        all_idx = [i for i in range(len(dataset['train']))]
        exclude_idx = pickle.load(open(args.wrong_ids_file, 'rb'))
        correct_idx = [item for item in all_idx if item not in exclude_idx]
        dataset['train'] = dataset['train'].select(correct_idx)
        print(f"filtering {len(exclude_idx)} rows")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.

    print(dataset)
    print(dataset.keys())
    column_names = dataset[split].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    elif args.conditioning_image_column in ['lineart', 'hed']:
        conditioning_image_column = image_column
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    resolution = (args.resolution, args.resolution)
    image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            # transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    label_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST, antialias=True),
            # transforms.CenterCrop(args.resolution),
        ]
    )

    def preprocess_train(examples):
        pil_images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in pil_images]

        if args.conditioning_image_column in ['lineart', 'hed']:
            conditioning_images = images
        else:
            conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
            conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        if args.label_column is not None:
            dtype = torch.long
            labels = [torch.tensor(np.asarray(label), dtype=dtype).unsqueeze(0) for label in examples[args.label_column]]
            labels = [label_image_transforms(label) for label in labels]

        # perform groupped random crop for image/conditioning_image/label
        if args.label_column is not None:
            grouped_data = [torch.cat([x, y, z]) for (x, y, z) in zip(images, conditioning_images, labels)]
            grouped_data = group_random_crop(grouped_data, args.resolution)

            images = [x[:3, :, :] for x in grouped_data]
            conditioning_images = [x[3:6, :, :] for x in grouped_data]
            labels = [x[6:, :, :] for x in grouped_data]

            # (1, H, W) => (H, w)

            examples[args.label_column] = labels
        else:
            grouped_data = [torch.cat([x, y]) for (x, y) in zip(images, conditioning_images)]
            grouped_data = group_random_crop(grouped_data, args.resolution)

            images = [x[:3, :, :] for x in grouped_data]
            conditioning_images = [x[3:, :, :] for x in grouped_data]

        # Dropout some of features for classifier-free guidance.
        for i, img_condition in enumerate(conditioning_images):
            rand_num = random.random()
            if rand_num < args.image_condition_dropout:
                conditioning_images[i] = torch.zeros_like(img_condition)
            elif rand_num < args.image_condition_dropout + args.text_condition_dropout:
                examples[caption_column][i] = ""
            elif rand_num < args.image_condition_dropout + args.text_condition_dropout + args.all_condition_dropout:
                conditioning_images[i] = torch.zeros_like(img_condition)
                examples[caption_column][i] = ""

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed)
            # rewrite the shuffled dataset on disk as contiguous chunks of data
            dataset["train"] = dataset["train"].flatten_indices()
            dataset["train"] = dataset["train"].select(range(args.max_train_samples))

        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return dataset, train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    if args.label_column is not None:
        labels = torch.stack([example[args.label_column] for example in examples])
        labels = labels.to(memory_format=torch.contiguous_format).float()
    else:
        labels = conditioning_pixel_values

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "labels": labels,
    }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Print the initial seeds
    print("="*10)
    print(f"Initial PyTorch seed: {torch.initial_seed()}")
    print(f"Initial NumPy seed: {np.random.get_state()[1][0]}")
    print(f"Initial Python random seed: {random.getstate()[1][0]}")
    print("save seed")
    torch.save({'torch': torch.initial_seed(), 'numpy': np.random.get_state()[1][0], 'python': random.getstate()[1][0]}, 'seed.pt')
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    reward_model = get_reward_model(args.task_name, args.reward_model_name_or_path)

    if args.controlnet_model_name_or_path:
        print("++++++++++++++++++++++++++++++++++++++++++++++")
        print("HHHHHH")
        logger.info("Loading existing controlnet weights")
        print(args.controlnet_model_name_or_path)
        print("++++++++++++++++++++++++++++++++++++++++++++++")
        print("HHHHHH")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "controlnet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    reward_model.requires_grad_(False)
    controlnet.train()

    # Create EMA for the ControlNet.
    if args.use_ema:
        ema_controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        ema_controlnet = EMAModel(ema_controlnet.parameters(), model_cls=ControlNetModel, model_config=ema_controlnet.config)
    else:
        ema_controlnet = None

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    if args.use_ema:
        ema_controlnet.to(accelerator.device)

    # Optimizer creation
    
    if not args.optimize_readout:
        optimizer = optimizer_class(
            controlnet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        assert False
        print("Additionally optimize readout heads")
        optimizer = optimizer_class(
            [
                {'params': controlnet.parameters()},
                {'params': edits[0]['aggregation_network'].parameters()}
            ],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    dataset, train_dataset = make_train_dataset(args, tokenizer, accelerator)

    if args.validation_prompt is None and args.validation_image is None:
        if 'validation' in dataset.keys():
            val_dataset = dataset['validation']
        else:
            dataset = train_dataset.train_test_split(test_size=0.00005)
            train_dataset, val_dataset = dataset['train'], dataset['test']
    else:
        val_dataset = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if not args.optimize_readout:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
        # Prepare others after preparing the model
        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

        controlnet, edits[0]['aggregation_network'], optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, edits[0]['aggregation_network'], optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    #=========================
    config = {'rg_kwargs': [{'head_type': 'spatial', 
                             'loss_rescale': 0.5, 
                             'aggregation_kwargs': {'aggregation_ckpt': args.readout_path,
                             }}]
                             }
    edits = get_edits(config, accelerator.device, weight_dtype)
    init_resnet_func(unet, save_mode='hidden', idxs=None, reset=True, aggragation_type=args.readout_type)
    if args.readout_type == 'attn':
        channels = collect_channels(unet)
    else:
        channels = collect_channels_conv(unet)

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    reward_model.to(accelerator.device, dtype=weight_dtype)
    reward_model.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs={"wandb": {"name": args.output_dir.split('/')[-1]}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.resume_from_checkpoint))
            print(path)
            print(path.split("-")[-1])
            global_step = int(path.split("-")[-1])
            print(f'====== global step =========')
            print(global_step)

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        loss_per_epoch = 0.
        pretrain_loss_per_epoch = 0.
        reward_loss_per_epoch = 0.
        reward_step_loss_per_epoch = 0.

        train_loss, train_pretrain_loss, train_reward_loss = 0., 0., 0.
        train_aligned_step_loss = 0.

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]  # text condition
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)  # image condition

                # This step is necessary. It took us a long time to find out this issue
                # The input of the hed/lineart model does not require normalization of the image
                if args.conditioning_image_column in ['lineart', 'hed']:
                    with torch.no_grad():
                        # mean & std used in image transformations
                        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        std = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(accelerator.device)
                        denormalized_condition_image = controlnet_image * std + mean
                        labels = reward_model(denormalized_condition_image.to(weight_dtype))
                        controlnet_image = labels.expand(-1, 3, -1, -1)  # (B, 3, H, W)
                        controlnet_image = 1 - controlnet_image if args.task_name == 'lineart' else controlnet_image

                """
                Training ControlNet
                """
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(args.timestep_sampling_start, args.timestep_sampling_end, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                pretrain_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")


                """
                Rewarding ControlNet
                """
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

                # image normalization, depends on different reward models
                # This step is necessary. It took us a long time to find out this issue
                if args.task_name == 'depth':
                    image = torchvision.transforms.functional.resize(image, (384, 384))
                    image = normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                elif args.task_name in ['lineart', 'hed']:
                    pass
                else:
                    image = normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

                # reward model inference
               
                outputs = reward_model(image.to(accelerator.device))

                # normalize the predicted depth to (0, 1]
                if type(outputs) == transformers.modeling_outputs.DepthEstimatorOutput:

                    # map predicted depth into [0, 1]
                    outputs = outputs.predicted_depth
                    outputs = torchvision.transforms.functional.resize(outputs, (args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR)
                    max_values = outputs.view(args.train_batch_size, -1).amax(dim=1, keepdim=True).view(args.train_batch_size, 1, 1)
                    outputs = outputs / max_values

                    # map label into [0, 1]
                    labels = batch["labels"].mean(dim=1)  # (N, 3, H, W) -> (N, H, W)
                    max_values = labels.view(labels.size(0), -1).max(dim=1)[0]
                    labels = labels / max_values.view(-1, 1, 1)

                if args.task_name in ['lineart', 'hed']:
                    pass
                else:
                    labels = batch["labels"]

                # Avoid nan loss when using FP16 (happen in softmax)
                # FP32 and BF16 both work well
                if image.dtype == torch.float16:
                    if isinstance(outputs, torch.Tensor):
                        outputs = outputs.to(torch.float32)
                        labels = labels.to(torch.float32)
                    elif isinstance(outputs, list):
                        outputs = [x.to(torch.float32) for x in outputs]
                        labels = [x.to(torch.float32) for x in labels]
                    else:
                        raise NotImplementedError

                # For depth we resize the label to the size of model output
                if args.task_name in ['depth', 'lineart', 'hed']:
                    labels = label_transform(labels, args.task_name, args.dataset_name)
                else:
                    raise NotImplementedError(f"Not support task: {args.task_name}.")

                labels = [x.to(accelerator.device) for x in labels] if isinstance(labels, list) else labels.to(accelerator.device)

                # Determine which samples in the current batch need to calculate reward loss
                timestep_mask = (args.min_timestep_rewarding <= timesteps.reshape(-1, 1)) & (timesteps.reshape(-1, 1) <= args.max_timestep_rewarding)

                # calculate the reward loss
                reward_loss = get_reward_loss(outputs, labels, args.task_name, reduction='none')

                # Reawrd Loss: (B, H, W)  =>  (B)
                
                reward_loss = reward_loss.mean(dim=(-1,-2))

                reward_loss = reward_loss.reshape_as(timestep_mask)
                reward_loss = (timestep_mask * reward_loss).sum() / (timestep_mask.sum() + 1e-10)
                loss = pretrain_loss + reward_loss * args.grad_scale
                assert args.grad_scale == 0.5
                #assert loss == pretrain_loss
                

                """
                Rewarding ControlNet each t
                """
                #==========================
                obs_feat = collect_and_resize_feats(unet, None, collection_type=args.readout_type)
                aggregation_network = edits[0]["aggregation_network"]
                emb = embed_timestep(unet, latents, timesteps)
                obs_feat = run_aggregation(obs_feat, aggregation_network, emb)
                timestep_mask = (args.min_timestep_readout <= timesteps.reshape(-1, 1)) & (timesteps.reshape(-1, 1) <= args.max_timestep_readout)
                
                obs_feat = (obs_feat + 1) / 2
                assert labels.min() >= 0 and labels.max() <= 1
                assert obs_feat.min() >= 0 and obs_feat.max() <= 1

                if args.task_name == 'depth':
                    obs_feat = obs_feat.mean(1)
                    max_values = obs_feat.view(obs_feat.size(0), -1).max(dim=1)[0]
                    obs_feat = obs_feat / max_values.view(-1, 1, 1)
                    
                    aligned_step_loss = F.mse_loss(torchvision.transforms.functional.resize(obs_feat.unsqueeze(1).float(), (512, 512), interpolation=transforms.InterpolationMode.BILINEAR),  torchvision.transforms.functional.resize(labels.unsqueeze(1), (512, 512), interpolation=transforms.InterpolationMode.BILINEAR).float(), reduction="none")
                    
                elif args.task_name == 'lineart':
                    obs_feat = (obs_feat.mean(1).unsqueeze(1) + 1) / 2
                    max_values = obs_feat.view(obs_feat.shape[0], -1).amax(dim=1, keepdim=True).view(obs_feat.shape[0], 1, 1)
                    min_values = obs_feat.view(obs_feat.shape[0], -1).amin(dim=1, keepdim=True).view(obs_feat.shape[0], 1, 1)
                    obs_feat = (obs_feat - min_values[...,None]) / (max_values[...,None] - min_values[...,None] + 1e-8)
                    obs_feat = 1 - obs_feat
                    aligned_step_loss = F.mse_loss(torchvision.transforms.functional.resize(obs_feat.float().repeat(1,3,1,1), (512, 512), interpolation=transforms.InterpolationMode.BILINEAR),  torchvision.transforms.functional.resize(labels.repeat(1,3,1,1), (512, 512), interpolation=transforms.InterpolationMode.BILINEAR).float(), reduction="none")
                
                elif args.task_name == 'hed':
                    aligned_step_loss = F.mse_loss(torchvision.transforms.functional.resize(obs_feat.float(), (512, 512), interpolation=transforms.InterpolationMode.BILINEAR),  torchvision.transforms.functional.resize(labels, (512, 512), interpolation=transforms.InterpolationMode.BILINEAR).float().repeat(1,3,1,1), reduction="none")
                
                aligned_step_loss = aligned_step_loss.reshape_as(timestep_mask)
                aligned_step_loss = (timestep_mask * aligned_step_loss).sum() / (timestep_mask.sum() + 1e-10)

                loss += aligned_step_loss * args.aligning_alpha


                """
                Losses
                """
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_pretrain_loss = accelerator.gather(pretrain_loss.repeat(args.train_batch_size)).mean()
                avg_reward_loss = accelerator.gather(reward_loss.repeat(args.train_batch_size)).mean()
                avg_aligned_step_loss = accelerator.gather(aligned_step_loss.repeat(args.train_batch_size)).mean()

                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_pretrain_loss += avg_pretrain_loss.item() / args.gradient_accumulation_steps
                train_reward_loss += avg_reward_loss.item() / args.gradient_accumulation_steps
                train_aligned_step_loss += avg_aligned_step_loss.item() / args.gradient_accumulation_steps

                # Back propagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_controlnet.step(controlnet.parameters())
                progress_bar.update(1)
                global_step += 1

                # loss when perform gradient backward
                accelerator.log({
                        "train_loss": train_loss,
                        "train_pretrain_loss": train_pretrain_loss,
                        "train_reward_loss": train_reward_loss,
                        "train_readout_step_loss": train_aligned_step_loss,
                        "lr": lr_scheduler.get_last_lr()[0]
                    },
                    step=global_step
                )
                loss_per_epoch += train_loss
                pretrain_loss_per_epoch += train_pretrain_loss
                reward_loss_per_epoch += train_reward_loss
                reward_step_loss_per_epoch += train_aligned_step_loss

                train_loss, train_pretrain_loss, train_reward_loss = 0., 0., 0.
                train_aligned_step_loss = 0.

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # directly save the state_dict
                        if accelerator.distributed_type != accelerate.DistributedType.FSDP:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)

                            if args.use_ema:
                                ema_controlnet.save_pretrained(f'{save_path}/controlnet_ema')

                            logger.info(f"Saved state to {save_path}")

                    start_time = time.time()
                    if global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            ema_controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            val_dataset
                        )

                        end_time = time.time()
                        logger.info(f"Validation time: {end_time - start_time} seconds")

            # only show in the progress bar
            logs = {
                "loss_step": loss.detach().item(),
                "pretrain_loss_step": pretrain_loss.detach().item(),
                "reward_loss_step": reward_loss.detach().item(),
                "train_readout_step_loss": aligned_step_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)

            # FSDP save model need to call all the ranks
            if global_step % args.checkpointing_steps == 0:
                if accelerator.distributed_type == accelerate.DistributedType.FSDP:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved accelerator state to {save_path}")

                    # Gather all of the state in the rank 0 device
                    accelerator.wait_for_everyone()
                    with FSDP.state_dict_type(controlnet, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                        state_dict = accelerator.get_state_dict(controlnet)

                    # Saving FSDP state
                    if accelerator.is_main_process:
                        torch.save(state_dict, os.path.join(save_path, 'controlnet_state_dict.pt'))
                        logger.info(f"Saved ControlNet state to {save_path}")

            if global_step >= args.max_train_steps:
                break

        logs = {
            "loss_epoch": loss_per_epoch / len(train_dataloader),
            "pretrain_loss_epoch": pretrain_loss_per_epoch / len(train_dataloader),
            "reward_loss_epoch": reward_loss_per_epoch / len(train_dataloader),
            "reward_step_loss_per_epoch": reward_step_loss_per_epoch / len(train_dataloader),
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()

    # If we use FSDP, saving the state_dict
    if accelerator.distributed_type == accelerate.DistributedType.FSDP:
        with FSDP.state_dict_type(controlnet, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            state_dict = accelerator.get_state_dict(controlnet)
            ema_state_dict = accelerator.get_state_dict(ema_controlnet)

        if accelerator.is_main_process:
            torch.save(state_dict, os.path.join(args.output_dir, 'controlnet_state_dict.pt'))
            torch.save(ema_state_dict, os.path.join(args.output_dir, 'controlnet_state_dict_ema.pt'))
            logger.info(f"Saved ControlNet state to {args.output_dir}")
    else:
        controlnet = accelerator.unwrap_model(controlnet)

        controlnet.save_pretrained(args.output_dir)
        ema_controlnet.save_pretrained(args.output_dir + '_ema')

    if accelerator.is_main_process:
        if args.push_to_hub:
            for _ in range(100):
                try:
                    save_model_card(
                        repo_id,
                        image_logs=image_logs,
                        base_model=args.pretrained_model_name_or_path,
                        repo_folder=args.output_dir,
                    )
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        path_in_repo=args.output_dir.replace('work_dirs/', ''),
                        commit_message=f"End of training {args.output_dir.split('/')[-1]}",
                        ignore_patterns=["step_*", "epoch_*"],
                        token=args.hub_token
                    )
                    break
                except:
                    continue

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)