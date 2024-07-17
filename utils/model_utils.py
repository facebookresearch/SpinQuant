# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import os
import torch


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization!
    pass


def get_layer_io_save_path(args):
    return os.path.join(args.save_path, "layer_io", f"{args.layer_idx:03d}.pt")


def capture_layer_io(layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())

        return hook

    handles = []

    captured_inputs = {
        "k_proj": [],  # q_proj, v_proj has the same input as k_proj
        "o_proj": [],
        "gate_proj": [],  # up_proj has the same input as gate_proj
        "down_proj": [],
    }

    captured_outputs = {
        "v_proj": [],
    }

    for name in captured_inputs.keys():
        module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
        handles.append(
            module.register_forward_hook(hook_factory(name, captured_inputs, True))
        )

    for name in captured_outputs.keys():
        module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
        handles.append(
            module.register_forward_hook(hook_factory(name, captured_outputs, False))
        )

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx : seq_idx + 1].to("cuda")
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {"input": captured_inputs, "output": captured_outputs}
