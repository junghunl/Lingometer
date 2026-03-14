import sys
import os
import torch
import warnings
import re
import logging
import ctypes
import numpy as np
import random
import torchaudio
from torch_audiomentations import AddBackgroundNoise, ApplyImpulseResponse


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


from ASPVAD import ASPVAD
def load_model(model_name, model_args, device=None, get_epoch=False, requires_grad=True, do_compile=False):
    if device==None: device = get_device()
    checkpoint  = model_args.pop('checkpoint',None)
    model = ASPVAD(**model_args)
    if checkpoint!=None: load_checkpoint(model, checkpoint)
    if requires_grad==False:
        for param in model.parameters(): param.requires_grad = False
    if do_compile and device.type=='cuda': model.compile()
    model.to(device)
    if get_epoch:
        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",checkpoint)[0])+1 if checkpoint else 1
        return model, start_epoch
    else: return model





def mono_resample(wav: torch.Tensor, sr: int=16000, target_sr: int = 16000) -> torch.Tensor:
    """
    [C,T] or [T] -> [1,T]
    """
    if wav.dim() == 1: wav = wav.unsqueeze(0)
    elif wav.dim() == 2 and wav.size(0) > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr: wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


class Augmentor:
    def __init__(self, rir_paths=None, noise_paths=None, rir_prob=0, noise_prob=0, min_snr_in_db=3, max_snr_in_db=30):
        self.do_aug = False if rir_prob==0 and noise_prob==0 else True
        if self.do_aug == False: return
        self.rir_augmentor = ApplyImpulseResponse(ir_paths=rir_paths,
                                sample_rate=16000,
                                p=rir_prob,
                                output_type='tensor')
        self.noise_augentor = AddBackgroundNoise(background_paths=noise_paths,
                            sample_rate=16000,
                            min_snr_in_db=min_snr_in_db,
                            max_snr_in_db=max_snr_in_db,
                            p=noise_prob,
                            output_type='tensor')

    def __call__(self, *args, **kwargs):
        return self.augment(*args, **kwargs)

    def augment(self, wav):
        if self.do_aug == False: return wav
        wav = wav.view(1,1,-1)
        wav = self.rir_augmentor(wav)
        wav = self.noise_augentor(wav)
        return wav.squeeze(0)


class Extractor:
    def __init__(self, apply_cmvn=False, **feature_args):
        if feature_args==None: feature_args={}
        basic_args = {'sample_rate':16000, 'n_fft':400, 'n_mels':24, 'win_length':400, 'hop_length':160}
        basic_args.update(feature_args)
        self.extractor = torchaudio.transforms.MelSpectrogram(**basic_args)
        self.apply_cmvn = apply_cmvn

    def __call__(self, *args, **kwargs):
        return self.extract(*args, **kwargs)

    def extract(self, wav:list|str, length=None):
        if isinstance(wav,str): wav, sr = torchaudio.load(wav)
        spec = self.extractor(wav)  # (1,F,T)
        spec = spec.squeeze(0).transpose(0,1)  # (T,F)
        spec = torch.log10(spec + 1e-6)
        if self.apply_cmvn==True:
            mean = spec.mean(dim=0, keepdim=True)
            std = spec.std(dim=0, keepdim=True)
            spec = (spec - mean) / (std + 1e-9)
        if length is not None: spec = spec[:length,:]
        return spec

def get_random_chunk(*data_list, chunk_len, rng=random):
    if not data_list: return None
    data_len = len(data_list[0])
    if data_len >= chunk_len:
        chunk_start = rng.randint(0, data_len-chunk_len)
        sliced = [d[chunk_start:chunk_start+chunk_len] for d in data_list]
    else:
        idx = torch.arange(chunk_len) % data_len
        sliced = [d[idx] for d in data_list]
    return sliced[0] if len(sliced) == 1 else tuple(sliced)

def get_random_chunk(*data_list, chunk_len, rng=None):
    if not data_list: return None
    if rng is None: rng = np.random.default_rng()
    data_len = len(data_list[0])
    if data_len >= chunk_len:
        chunk_start = rng.integers(0, data_len-chunk_len+1)
        sliced = [d[chunk_start:chunk_start+chunk_len] for d in data_list]
    else:
        idx = torch.arange(chunk_len) % data_len
        sliced = [d[idx] for d in data_list]
    return sliced[0] if len(sliced) == 1 else tuple(sliced)