import sys
import os
import torch
import warnings
import re
import logging
import ctypes
import numpy as np
import random






def get_logger(outdir, fname):
    logging.basicConfig(level=logging.INFO, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger("logger")
    formatter = logging.Formatter("[ %(levelname)s : %(asctime)s ] - %(message)s")
    fh = logging.FileHandler(os.path.join(outdir, fname))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def load_checkpoint(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint: state_dict = checkpoint['model']
    else: state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)


def save_checkpoint(model, path: str):
    if isinstance(model, torch.nn.Module): state_dict = model.state_dict()
    else: state_dict = model
    torch.save(state_dict, path)


def validate_path(path, is_dir=False):
    dir = path if is_dir else os.path.dirname(path)
    if not os.path.exists(dir) and (dir != ''): os.makedirs(dir)
    return path


def clean_memory():
  #gc.collect()  # This is not only slow, but it also confuses malloc_trim.
  try: ctypes.CDLL("libc.so.6").malloc_trim(0)
  except: pass


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #torch.backends.cudnn.deterministic = True  # this slows GPU down
    torch.backends.cudnn.benchmark = False  # input shape is variable
    return seed


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device




from personal_VAD.models.PVAD1_et import PVAD1ET, PVAD1ET_FILM
from personal_VAD.models.ASPVAD_sigmoid import ASPVAD as ASPVAD_sigmoid
from personal_VAD.models.ASPVAD import ASPVAD
def get_model(model_name: str):
    if model_name=="PVAD1ET": return PVAD1ET
    elif model_name=="PVAD1ET_FILM": return PVAD1ET_FILM
    elif model_name=="ASPVAD": return ASPVAD
    elif model_name=="ASPVAD_sigmoid": return ASPVAD_sigmoid
def load_model(model_name, model_args, device=None, get_epoch=False, requires_grad=True, do_compile=False):
    if device==None: device = get_device()
    checkpoint  = model_args.pop('checkpoint',None)
    model = get_model(model_name)(**model_args)
    if checkpoint!=None: load_checkpoint(model, checkpoint)
    if requires_grad==False:
        for param in model.parameters(): param.requires_grad = False
    if do_compile and device.type=='cuda': model.compile()
    model.to(device)
    if get_epoch:
        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",checkpoint)[0])+1 if checkpoint else 1
        return model, start_epoch
    else: return model