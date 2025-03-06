"""
saving utilities
"""
import json
import os
from os.path import dirname, exists, join, realpath
import subprocess
from easydict import EasyDict as edict

import torch
from utils.basic_utils import save_json, make_zipfile, load_json
from utils.logger import LOGGER
from typing import Any, Dict


def save_training_meta(args):
    # args is an EasyDict object, treat it the same as a normal dict
    os.makedirs(join(args.output_dir, 'log'), exist_ok=True)
    os.makedirs(join(args.output_dir, 'ckpt'), exist_ok=True)

    # training args
    save_args_path = join(args.output_dir, 'log', 'args.json')
    save_json(args, save_args_path, save_pretty=True)

    # save a copy of the codebase
    code_dir = dirname(dirname(dirname(os.path.realpath(__file__))))
    code_zip_filename = os.path.join(args.output_dir, "code.zip")
    LOGGER.info(f"Saving code from {code_dir} to {code_zip_filename}...")
    make_zipfile(code_dir, code_zip_filename,
                 enclosing_dir="code",
                 exclude_dirs_substring="results",
                 exclude_dirs=["results", "debug_results", "__pycache__"],
                 exclude_extensions=[".pyc", ".ipynb", ".swap"])
    LOGGER.info(f"Saving code done.")


class ModelSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.max_save_load_trial = 10

    def save(self, step, model, optimizer=None, prefix="model"):
        model_path = join(self.output_dir, f"{prefix}_step_{step}.pt")
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.state_dict().items()}
        
        save_trial = 0
        while save_trial < self.max_save_load_trial:
            try:
                LOGGER.info(f"ModelSaver save trial NO. {save_trial}")
                torch.save(state_dict, model_path)
                if optimizer is not None:
                    optimizer_state_dict = \
                        {k: v.cpu() if isinstance(v, torch.Tensor) else v
                         for k, v in optimizer.state_dict().items()}
                    dump = {'step': step, 'optimizer': optimizer_state_dict}
                    torch.save(dump, f'{self.output_dir}/{prefix}_step_{step}_train_state.pt')
                break
            except Exception:
                save_trial += 1


class BestModelSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.max_save_load_trial = 10
        self.bestr1 = 0

    def save(self, step, model, prefix="model_best"):
        model_path = join(self.output_dir, f"{prefix}.pt")
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.state_dict().items()}
        
        save_trial = 0
        while save_trial < self.max_save_load_trial:
            try:
                LOGGER.info(f"BestModelSaver save trial NO. {save_trial}")
                torch.save(state_dict, model_path)
                break
            except Exception:
                save_trial += 1


def load_state_dict_with_mismatch(model, loaded_state_dict_or_path):
    """operated in-place, no need to return `model`"""
    if isinstance(loaded_state_dict_or_path, str):
        loaded_state_dict = torch.load(loaded_state_dict_or_path, map_location="cpu")
    else:
        loaded_state_dict = loaded_state_dict_or_path

    model_keys = set(model.state_dict().keys())
    load_keys = set(loaded_state_dict.keys())

    toload = {}
    mismatched_shape_keys = []
    for k in model_keys:
        if k in load_keys:
            if model.state_dict()[k].shape != loaded_state_dict[k].shape:
                mismatched_shape_keys.append(k)
            else:
                toload[k] = loaded_state_dict[k]

    LOGGER.info("You can ignore the keys with `num_batches_tracked` or from task heads")
    LOGGER.info(f"Keys in loaded but not in model: {sorted(load_keys.difference(model_keys))}")
    LOGGER.info(f"Keys in model but not in loaded: {sorted(model_keys.difference(load_keys))}")
    LOGGER.info(f"Keys in model and loaded, but shape mismatched: {sorted(mismatched_shape_keys)}")
    
    model.load_state_dict(toload, strict=False)


def _to_cuda(state):
    """ usually load from cpu checkpoint but need to load to cuda """
    if isinstance(state, torch.Tensor):
        return state.cuda()
    elif isinstance(state, (list, tuple)):
        return type(state)(_to_cuda(t) for t in state)
    elif isinstance(state, dict):
        return {n: _to_cuda(t) for n, t in state.items()}
    return state


def _to_cpu(state):
    """ store in cpu to avoid GPU0 device """
    if isinstance(state, torch.Tensor):
        return state.cpu()
    elif isinstance(state, (list, tuple)):
        return type(state)(_to_cpu(t) for t in state)
    elif isinstance(state, dict):
        return {n: _to_cpu(t) for n, t in state.items()}
    return state
