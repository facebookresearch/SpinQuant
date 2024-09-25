# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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

# Adopt from https://fburl.com/code/b4jqkgir

import math

from typing import Any, Tuple

import torch
from torch._tensor import Tensor


def compute_intermediate_size(n) -> int:
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def shard_tensor(tensor: Tensor, dim: int, num_shards: int) -> Tuple[Tensor, ...]:
    total_size = tensor.shape[dim]
    n_size_per_shard = total_size // num_shards

    ret_tensors = torch.split(tensor, n_size_per_shard, dim)
    return ret_tensors


def write_model_llama(
    hf_state_dict,
    config,
    num_shards: int,
):
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    dim = config.hidden_size
    num_key_value_heads = config.num_key_value_heads
    assert n_heads % num_shards == 0
    assert dim % (n_heads * 2) == 0

    def un_permute(w, is_query=True):
        return (
            w.view(
                n_heads if is_query else num_key_value_heads,
                2,
                dim // n_heads // 2,
                dim,
            )
            .transpose(1, 2)
            .reshape(-1, dim)
        )

    model_shard_dicts = [{} for _ in range(num_shards)]
    for layer_i in range(n_layers):
        ## store the same in every shard
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.attention_norm.weight"] = (
                hf_state_dict[f"model.layers.{layer_i}.input_layernorm.weight"].clone()
            )

        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.ffn_norm.weight"] = (
                hf_state_dict[
                    f"model.layers.{layer_i}.post_attention_layernorm.weight"
                ].clone()
            )

        ### int weight
        self_attn_q_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.self_attn.q_proj.module.int_weight"
        ]
        self_attn_q_proj_weight = un_permute(self_attn_q_proj_weight)
        list_self_attn_q_proj_weight = shard_tensor(
            self_attn_q_proj_weight, 0, num_shards
        )
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.attention.wq.weight"] = (
                list_self_attn_q_proj_weight[shard_i].clone().to(torch.int8)
            )

        ###
        self_attn_k_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.self_attn.k_proj.module.int_weight"
        ]
        self_attn_k_proj_weight = un_permute(self_attn_k_proj_weight, is_query=False)
        list_self_attn_k_proj_weight = shard_tensor(
            self_attn_k_proj_weight, 0, num_shards
        )
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.attention.wk.weight"] = (
                list_self_attn_k_proj_weight[shard_i].clone().to(torch.int8)
            )

        ###
        self_attn_v_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.self_attn.v_proj.module.int_weight"
        ]
        list_self_attn_v_proj_weight = shard_tensor(
            self_attn_v_proj_weight, 0, num_shards
        )
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.attention.wv.weight"] = (
                list_self_attn_v_proj_weight[shard_i].clone().to(torch.int8)
            )

        ###
        self_attn_o_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.self_attn.o_proj.module.int_weight"
        ]
        list_self_attn_o_proj_weight = shard_tensor(
            self_attn_o_proj_weight, 1, num_shards
        )
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.attention.wo.weight"] = (
                list_self_attn_o_proj_weight[shard_i].clone().to(torch.int8)
            )

        ###
        mlp_gate_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.mlp.gate_proj.module.int_weight"
        ]
        list_mlp_gate_proj_weight = shard_tensor(mlp_gate_proj_weight, 0, num_shards)
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.feed_forward.w1.weight"] = (
                list_mlp_gate_proj_weight[shard_i].clone().to(torch.int8)
            )

        ###
        mlp_down_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.mlp.down_proj.module.int_weight"
        ]
        list_mlp_down_proj_weight = shard_tensor(mlp_down_proj_weight, 1, num_shards)
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.feed_forward.w2.weight"] = (
                list_mlp_down_proj_weight[shard_i].clone().to(torch.int8)
            )

        ###
        mlp_up_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.mlp.up_proj.module.int_weight"
        ]
        list_mlp_up_proj_weight = shard_tensor(mlp_up_proj_weight, 0, num_shards)
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.feed_forward.w3.weight"] = (
                list_mlp_up_proj_weight[shard_i].clone().to(torch.int8)
            )

        ### scale
        self_attn_q_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.self_attn.q_proj.module.scale"
        ]
        self_attn_q_proj_weight = un_permute(self_attn_q_proj_weight)
        list_self_attn_q_proj_weight = shard_tensor(
            self_attn_q_proj_weight, 0, num_shards
        )
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.attention.wq.scale"] = (
                list_self_attn_q_proj_weight[shard_i].clone()
            )

        ###
        self_attn_k_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.self_attn.k_proj.module.scale"
        ]
        self_attn_k_proj_weight = un_permute(self_attn_k_proj_weight, is_query=False)
        list_self_attn_k_proj_weight = shard_tensor(
            self_attn_k_proj_weight, 0, num_shards
        )
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.attention.wk.scale"] = (
                list_self_attn_k_proj_weight[shard_i].clone()
            )

        ###
        self_attn_v_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.self_attn.v_proj.module.scale"
        ]
        list_self_attn_v_proj_weight = shard_tensor(
            self_attn_v_proj_weight, 0, num_shards
        )
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.attention.wv.scale"] = (
                list_self_attn_v_proj_weight[shard_i].clone()
            )

        ###
        self_attn_o_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.self_attn.o_proj.module.scale"
        ]
        list_self_attn_o_proj_weight = shard_tensor(
            self_attn_o_proj_weight, 1, num_shards
        )
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.attention.wo.scale"] = (
                list_self_attn_o_proj_weight[shard_i].clone()
            )

        ###
        mlp_gate_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.mlp.gate_proj.module.scale"
        ]
        list_mlp_gate_proj_weight = shard_tensor(mlp_gate_proj_weight, 0, num_shards)
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.feed_forward.w1.scale"] = (
                list_mlp_gate_proj_weight[shard_i].clone()
            )

        ###
        mlp_down_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.mlp.down_proj.module.scale"
        ]
        list_mlp_down_proj_weight = shard_tensor(mlp_down_proj_weight, 1, num_shards)
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.feed_forward.w2.scale"] = (
                list_mlp_down_proj_weight[shard_i].clone()
            )

        ###
        mlp_up_proj_weight = hf_state_dict[
            f"model.layers.{layer_i}.mlp.up_proj.module.scale"
        ]
        list_mlp_up_proj_weight = shard_tensor(mlp_up_proj_weight, 0, num_shards)
        for shard_i in range(num_shards):
            model_shard_dicts[shard_i][f"layers.{layer_i}.feed_forward.w3.scale"] = (
                list_mlp_up_proj_weight[shard_i].clone()
            )

    ##
    for shard_i in range(num_shards):
        model_shard_dicts[shard_i]["norm.weight"] = hf_state_dict[
            "model.norm.weight"
        ].clone()

    list_embed_tokens_weight = shard_tensor(
        hf_state_dict["model.embed_tokens.int_weight"], 1, num_shards
    )
    list_lm_head_weight = shard_tensor(
        hf_state_dict["lm_head.module.int_weight"], 0, num_shards
    )
    for shard_i in range(num_shards):
        model_shard_dicts[shard_i]["tok_embeddings.weight"] = (
            list_embed_tokens_weight[shard_i].clone().to(torch.int8)
        )
        model_shard_dicts[shard_i]["output.weight"] = (
            list_lm_head_weight[shard_i].clone().to(torch.int8)
        )

    list_embed_tokens_weight = shard_tensor(
        hf_state_dict["model.embed_tokens.scale"], 1, num_shards
    )
    list_lm_head_weight = shard_tensor(
        hf_state_dict["lm_head.module.scale"], 0, num_shards
    )
    for shard_i in range(num_shards):
        model_shard_dicts[shard_i]["tok_embeddings.scale"] = (
            list_embed_tokens_weight[shard_i].clone().to(torch.float)
        )
        model_shard_dicts[shard_i]["output.scale"] = (
            list_lm_head_weight[shard_i].clone().to(torch.float)
        )

    return model_shard_dicts


def sanitize_checkpoint_from_spinquant(
    checkpoint: Any,
    group_size: int,
):
    """
    Sanitize the SpinQuant checkpoint.
        - Renames 'scale' to 'scales'
        - Groups scales
        - Removes 'o_weight'
        - Converts all tensors to contiguous format
    """
    keys_to_rename = []
    keys_to_remove = []
    for k, _ in checkpoint.items():
        if k.endswith(".scale"):
            new_key = k + "s"
            keys_to_rename.append((k, new_key))

    for old_key, new_key in keys_to_rename:
        old_val = checkpoint.pop(old_key)
        checkpoint[new_key] = old_val if group_size == -1 else old_val[:, ::group_size]
    for k in keys_to_remove:
        checkpoint.pop(k)
    for k, v in checkpoint.items():
        checkpoint[k] = v.contiguous()
    return checkpoint
