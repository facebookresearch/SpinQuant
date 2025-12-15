import torch
import argparse
from collections import OrderedDict


def export(state_dict, group_size):
    """ Exports state_dict to LiteML format. Used to load the state dict outside LiteML's RetrainerModel class"""
    liteml_state_dict = dict()
    for key, value in state_dict['w_quantizers'].items():
        prefix = key.split('.module')[0]
        weights_quantizer_prefix = f"_model._model.{prefix}._weights_quantizer.obs"
        if group_size > -1:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale[None, :, :, None]
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero[None, :, :, None]
        else:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero


    for key, value in state_dict['model'].items():
        last, before_last = key.split('.')[-1], key.split('.')[-2]
        if 'layernorm' in key:
            key_liteml = f"_model._model.{key}"
            liteml_state_dict[key_liteml] = value
        elif key == 'model.embed_tokens.weight' or key == 'model.norm.weight':
            key_liteml = f"_model._model.{key}"
            liteml_state_dict[key_liteml] = value
        elif last == 'weight' and before_last != 'module':
            key_liteml = f"_model._model.{key.replace('weight','_model.weight')}"
            liteml_state_dict[key_liteml] = value

    liteml_state_dict = OrderedDict(sorted(liteml_state_dict.items()))
    return liteml_state_dict


def export_retrainer_model(state_dict, group_size):
    """
    Exports state_dict to LiteML format where the state_dict's path is provided in the config.yaml file, thus
    we remove the ._model._model prefix compared to the export function above.
    """
    liteml_state_dict = dict()
    for key, value in state_dict['w_quantizers'].items():
        prefix = key.split('.module')[0]
        weights_quantizer_prefix = f"{prefix}._weights_quantizer.obs"
        if group_size > -1:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale[None, :, :, None]
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero[None, :, :, None]
        else:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero


    for key, value in state_dict['model'].items():
        last, before_last = key.split('.')[-1], key.split('.')[-2]
        if 'layernorm' in key:
            liteml_state_dict[key] = value
        elif key == 'model.embed_tokens.weight' or key == 'model.norm.weight':
            liteml_state_dict[key] = value
        elif last == 'weight' and before_last != 'module':
            key_liteml = f"{key.replace('weight','_model.weight')}"
            liteml_state_dict[key_liteml] = value

    liteml_state_dict = OrderedDict(sorted(liteml_state_dict.items()))
    return liteml_state_dict


def export_retrainer_model_TrueQuantRMSNorm(state_dict, group_size):
    """
    Exports state_dict to LiteML format for a model that uses TrueQuantRMSNorm layer.
    """
    liteml_state_dict = dict()
    for key, value in state_dict['w_quantizers'].items():
        prefix = key.split('.module')[0]
        weights_quantizer_prefix = f"{prefix}._weights_quantizer.obs"
        if group_size > -1:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale[None, :, :, None]
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero[None, :, :, None]
        else:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero


    for key, value in state_dict['model'].items():
        last, before_last = key.split('.')[-1], key.split('.')[-2]
        if 'layernorm' in key or key == 'model.norm.weight':
            key = key.replace('weight', '_RMSnorm.weight')  # change RMSnorm key according to TrueQuantRMSNorm implementation
            liteml_state_dict[key] = value
        elif key == 'model.embed_tokens.weight':
            liteml_state_dict[key] = value
        elif last == 'weight' and before_last != 'module':
            key_liteml = f"{key.replace('weight','_model.weight')}"
            liteml_state_dict[key_liteml] = value

    liteml_state_dict = OrderedDict(sorted(liteml_state_dict.items()))
    return liteml_state_dict


def export_retrainer_model_fuse_lm_head_TrueQuantRMSNorm(state_dict, group_size, true_quant):
    """
    Exports state_dict to LiteML format for a model that uses TrueQuantRMSNorm layer.
    """
    liteml_state_dict = dict()
    for key, value in state_dict['w_quantizers'].items():
        prefix = key.split('.module')[0]
        weights_quantizer_prefix = f"{prefix}._weights_quantizer.obs"
        if group_size > -1:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale[None, :, :, None]
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero[None, :, :, None]
        else:
            liteml_state_dict[f"{weights_quantizer_prefix}.scale_factor"] = value.scale
            liteml_state_dict[f"{weights_quantizer_prefix}.zp"] = value.zero


    for key, value in state_dict['model'].items():
        last, before_last = key.split('.')[-1], key.split('.')[-2]
        if 'layernorm' in key:
            if true_quant:
                key = key.replace('weight', '_RMSnorm.weight')  # change RMSnorm key according to TrueQuantRMSNorm implementation
            liteml_state_dict[key] = value
        elif key == 'lm_head.weight':
            key = 'lm_head.model.lm_head._model.weight'
            liteml_state_dict[key] = value
        elif key == 'model.norm.weight':
            if true_quant:
                key = 'lm_head.model.norm._RMSnorm.weight' # with TrueQuant RMSNorm
            else:
                key = 'lm_head.model.norm.weight' # with original RMSNorm
            liteml_state_dict[key] = value
        elif key == 'model.embed_tokens.weight':
            liteml_state_dict[key] = value
        elif last == 'weight' and before_last != 'module':
            key_liteml = f"{key.replace('weight','_model.weight')}"
            liteml_state_dict[key_liteml] = value

    liteml_state_dict = OrderedDict(sorted(liteml_state_dict.items()))
    return liteml_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Export SpinQuant's state dict to LiteML format for Llama model.")
    parser.add_argument('--group_size', type=int, default=128, help='group size used for activations and weights in group quantization')
    parser.add_argument('--spinquant_path', type=str, help="path to SpinQuant's state dict file")
    parser.add_argument('--liteml_path', type=str, help="path to save the exported state dict file in LiteML format")
    parser.add_argument(
        "--true_quant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Export the model in a compatible format to "TrueQuant" mode of LiteML',
    )
    parser.add_argument(
        "--fuse_lm_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Fuse lm_head with last RMSNorm.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    spinquant_path = args.spinquant_path
    liteml_path = args.liteml_path
    group_size = args.group_size
    true_quant = args.true_quant
    fuse_lm_head = args.fuse_lm_head
    print(f"Loading SpinQuant's state dict from {spinquant_path}")
    state_dict = torch.load(spinquant_path)
    if fuse_lm_head:
        print('Exporting to LiteML format in "TrueQuant" mode and fusing lm_head with RMSNorm')
        out_state_dict = export_retrainer_model_fuse_lm_head_TrueQuantRMSNorm(state_dict, group_size, true_quant)
    elif true_quant:
        print('Exporting to LiteML format in "TrueQuant" mode')
        out_state_dict = export_retrainer_model_TrueQuantRMSNorm(state_dict, group_size)
    else:
        print("Exporting to LiteML format")
        out_state_dict = export_retrainer_model(state_dict, group_size)
    torch.save(out_state_dict, liteml_path)
    print(f'Finished exporting file {args.liteml_path}')


if __name__ == '__main__':
    main()
