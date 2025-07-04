# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
# limitations under the License.

# =================================================================
# This code is adapted from resnet.py in Diffusion Hyperfeatures
# and implements resnet feature caching.
# Original source: https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures/blob/main/archs/stable_diffusion/resnet.py
# =================================================================

import torch
import einops
from omegaconf import OmegaConf
from aggregation_network import AggregationNetwork
from typing import Any, Dict, Optional
import einops


def init_resnet_func(
    unet,
    save_mode="",
    reset=True,
    idxs=None,
    aggragation_type='attn'
):
    def new_forward(self, input_tensor, temb, scale=None):
        # https://github.com/huggingface/diffusers/blob/20e92586c1fda968ea3343ba0f44f2b21f3c09d2/src/diffusers/models/resnet.py#L460
        if save_mode == "input":
            self.feats = input_tensor

        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        
        if save_mode == "hidden":
            self.feats = hidden_states

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        if save_mode == "output":
            self.feats = output_tensor

        self.hws = (output_tensor.shape[2], output_tensor.shape[3])
        return output_tensor

    def new_forward_attn(self, hidden_states, encoder_hidden_states,
                        attention_mask: Optional[torch.Tensor] = None,
                        encoder_attention_mask: Optional[torch.Tensor] = None,
                        timestep: Optional[torch.LongTensor] = None,
                        cross_attention_kwargs: Dict[str, Any] = None,
                        class_labels: Optional[torch.LongTensor] = None,
                        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,):
        
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                print("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        h = int(attn_output.shape[1]**(1/2))
        self.feats = einops.rearrange(attn_output, 'b (h w) c -> b c h w', h=h) 
        
        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states
    
    if aggragation_type == 'conv':
        layers = collect_layers(unet, idxs)
        for module in layers:
            module.forward = new_forward.__get__(module, type(module))
            if reset:
                module.feats = None
                module.hws = None
    else:
        layers = collect_layers_attn(unet, idxs)
        for module in layers:
            module.forward = new_forward_attn.__get__(module, type(module))
            if reset:
                module.feats = None
                module.hws = None

def collect_layers(unet, idxs=None):
    layers = []
    layer_idx = 0
    for up_block in unet.up_blocks:
        for module in up_block.resnets:
            if idxs is None or layer_idx in idxs:
                layers.append(module)
            layer_idx += 1
    return layers


def collect_layers_attn(unet, idxs=None, layers_use=['up']):
    layers = []
    layer_idx = 0

    if 'down' in layers_use:
        for down_block in unet.down_blocks:
            try:
                for module in down_block.attentions:
                    if idxs is None or layer_idx in idxs:
                        layers.append(module.transformer_blocks[0])
                    layer_idx += 1
            except:
                pass
    if 'mid' in layers_use:
        for mid_block in [unet.mid_block]:
            try:
                for module in mid_block.attentions:
                    if idxs is None or layer_idx in idxs:
                        layers.append(module.transformer_blocks[0])
                    layer_idx += 1
            except:
                pass
    if 'up' in layers_use:
        for up_block in unet.up_blocks:
            try:
                for module in up_block.attentions:
                    if idxs is None or layer_idx in idxs:
                        layers.append(module.transformer_blocks[0])
                    layer_idx += 1
            except:
                pass
    return layers

def collect_feats_conv(unet, idxs=None):
    return [module.feats for module in collect_layers(unet, idxs)]

def collect_feats(unet, idxs=None):
    return [module.feats for module in collect_layers_attn(unet, idxs)]

def collect_channels(unet, idxs=None):
    return [module.attn1.to_out[0].out_features for module in collect_layers_attn(unet, idxs)]

def collect_channels_conv(unet, idxs=None):
    return [module.time_emb_proj.out_features for module in collect_layers(unet, idxs)]

def collect_hws(unet, idxs=None):
    return [module.hws for module in collect_layers(unet, idxs)]

def set_timestep(unet, timestep=None):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        module.timestep = timestep

def resize(x, old_res, new_res, mode):
    # (batch_size, width * height, channels)
    batch_size, size, channels = x.shape
    x = x.reshape((batch_size, *old_res, channels))
    x = einops.rearrange(x, 'b h w c -> b c h w')
    x = torch.nn.functional.interpolate(x, size=new_res, mode=mode)
    x = einops.rearrange(x, 'b c h w -> b h w c')
    return x

def resize_feat(feat, new_res=(64, 64), resize_mode="bilinear"):
    old_res = feat.shape[2:]
    feat = einops.rearrange(feat, 'b c h w -> b (h w) c')
    feat = resize(feat, old_res=old_res, new_res=new_res, mode=resize_mode)
    return feat

def collect_and_resize_feats(model, idxs, latent_dim=(64, 64), collection_type='attn'):
    if model is None:
        return None
    if collection_type == 'attn':
        feature_store = {"up": collect_feats(model, idxs=idxs)}
    else:
        feature_store = {"up": collect_feats_conv(model, idxs=idxs)}
    
    feats = []
    for key in feature_store:
        for i, feat in enumerate(feature_store[key]):
            feat = resize_feat(feat, new_res=latent_dim)
            feats.append(feat)
    # Concatenate all layers along the channel
    # dimension to get shape (b s d)
    if len(feats) > 0:
        feats = torch.cat(feats, dim=-1)
    else:
        feats = None
    return feats

def get_edits(config, device, dtype):
    edits = []
    for item in config["rg_kwargs"]:
        aggregation_network, aggregation_config = load_aggregation_network(item["aggregation_kwargs"], device, dtype)
        # for param in aggregation_network.parameters():
        #     param.requires_grad = False
        item["aggregation_network"] = aggregation_network
        item["aggregation_network"].eval()
        item["aggregation_kwargs"] = {**item["aggregation_kwargs"], **aggregation_config}
        edits.append(item)
    return edits


def load_aggregation_network(aggregation_config, device, dtype):
    weights_path = aggregation_config["aggregation_ckpt"]
    #weights_path = '/home/jovyan/konovalova/controlnet_redout/weights/checkpoint_step_5000.pt'
    state_dict = torch.load(weights_path)
    config = state_dict["config"]
    aggregation_kwargs = config.get("aggregation_kwargs", {})
    custom_aggregation_kwargs = {k: v for k, v in aggregation_config.items() if "aggregation" not in k}
    aggregation_kwargs = {**aggregation_kwargs, **custom_aggregation_kwargs}
    aggregation_network = AggregationNetwork(
        projection_dim=config["projection_dim"],
        feature_dims=config["dims"],
        device=device,
        save_timestep=config["save_timestep"],
        num_timesteps=config["num_timesteps"],
        **aggregation_kwargs
    )
    aggregation_network.load_state_dict(state_dict["aggregation_network"], strict=True)
    #aggregation_network.load_state_dict(torch.load(aggregation_config["aggregation_ckpt"]))
    aggregation_network = aggregation_network.to(device).to(dtype)
    return aggregation_network, config

def run_aggregation(feats, aggregation_network, emb=None):
    feats = einops.rearrange(feats, 'b w h c -> b c w h')
    feats = aggregation_network(feats, emb)
    return feats

def embed_timestep(unet, sample, timestep):
    timesteps = preprocess_timestep(sample, timestep)
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = unet.time_proj(timesteps)
    t_emb = t_emb.to(dtype=sample.dtype)
    emb = unet.time_embedding(t_emb, None)
    return emb

def preprocess_timestep(sample, timestep):
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    return timesteps