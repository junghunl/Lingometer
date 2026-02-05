"""
Model loading utilities (hyperparameters + checkpoint).
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any, Callable
import yaml
import torch
import os
import re
import random
import numpy as np

from models.wce_frame_onset import WCEFrameOnset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def load_hparams(hparam_path: Path) -> Dict[str, Any]:
    with open(hparam_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_checkpoint(model: torch.nn.Module, path: str):
    if not os.path.exists(path):
        print(f"Warning: Checkpoint {path} does not exist.")
        return

    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint: 
        state_dict = checkpoint['model']
    else: 
        state_dict = checkpoint
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_model] Warning: missing keys: {missing}")
    if unexpected:
        print(f"[load_model] Warning: unexpected keys: {unexpected}")

def get_model(model_name: str):
    if model_name == "wce_frame_onset":
        return WCEFrameOnset
    raise ValueError(f"Unknown model_name: {model_name}")

def load_model(model_name, model_args, device=None, get_epoch=False, requires_grad=True, do_compile=False):
    if device==None: device = get_device()
    checkpoint  = model_args.pop('checkpoint',None)
    model = get_model(model_name)(**model_args)
    if checkpoint!=None: load_checkpoint(model, checkpoint)
    if requires_grad==False:
        for param in model.parameters(): param.requires_grad = False
    if do_compile and device.type=='cuda': 
        model = torch.compile(model)
    model.to(device)
    if get_epoch:
        try:
            start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)", str(checkpoint))[0])+1 if checkpoint else 1
        except (IndexError, ValueError):
            start_epoch = 1
        return model, start_epoch
    else: return model

