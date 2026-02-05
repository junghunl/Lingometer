import random, pickle, gzip

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from personal_VAD.utils import *

class AudioDataset(Dataset):
    def __init__(self, conv_path, enroll_path, min_enroll_len=5, fix_conv_len=0, deterministic=False):
        self.min_enroll_len = min_enroll_len*100  # second to frame
        self.fix_conv_len = fix_conv_len*100  # second to frame
        self.deterministic = deterministic
        with gzip.open(conv_path, "rb") as f: conv_data = pickle.load(f)
        with gzip.open(enroll_path, "rb") as f: enroll_data = pickle.load(f)

        enroll_data_proc = {}
        no_enroll_spks = set()
        for spk, samples in enroll_data.items():
            enroll_data_proc[spk] = [conv_data[spk][0][int(s*100):int(e*100),:] for _,s,e in samples]
            enroll_data_proc[spk] = [x for x in enroll_data_proc[spk] if x.size(0)>=self.min_enroll_len]
            if not enroll_data_proc[spk]: no_enroll_spks.add(spk)
        self.conv_data = conv_data
        self.enroll_data = enroll_data_proc
        self.spks = list(conv_data.keys())
        self.spks.sort()

        if no_enroll_spks: print(f"There are speakers who has no utt of length > min_enroll_len:", no_enroll_spks)
        self.valid_idx = [idx for idx,spk in enumerate(self.spks) if spk not in no_enroll_spks]
        self.invalid_idx = {idx for idx,spk in enumerate(self.spks) if spk in no_enroll_spks}

    def __len__(self):
        return len(self.spks)

    def __getitem__(self, idx):
        rng = random.Random(idx) if self.deterministic else random
        if idx in self.invalid_idx: idx = rng.choice(self.valid_idx)
        enroll_spk = self.spks[idx]
        simul_feat, label = self.conv_data[enroll_spk]
        if self.fix_conv_len and self.fix_conv_len < len(label):
            chunk_start = rng.randint(0,len(label)-self.fix_conv_len)
            simul_feat = simul_feat[chunk_start:chunk_start+self.fix_conv_len]
            label = label[chunk_start:chunk_start+self.fix_conv_len]
        enroll_feat = rng.choice(self.enroll_data[enroll_spk]) if not self.deterministic else self.enroll_data[enroll_spk][0]
        return {'enroll':enroll_feat, 'simul':simul_feat, 'label':label, 'spk':idx}


def collate_fn(batch):
    enrolls = [item["enroll"] for item in batch]
    enrolls = pad_sequence(enrolls, batch_first=True, padding_value=0)
    enrolls = enrolls.transpose(1,2)

    simuls = [item["simul"] for item in batch]
    simuls = pad_sequence(simuls, batch_first=True, padding_value=0)
    simuls = simuls.transpose(1,2)
    
    labels = [item["label"] for item in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=-1).long()

    spks = torch.tensor([item["spk"] for item in batch])

    enroll_lengths = torch.tensor([len(item["enroll"]) for item in batch])
    simul_lengths = torch.tensor([len(item["simul"]) for item in batch])

    return {"spk":spks, "enroll":enrolls, "simul": simuls, "label": labels, "enroll_len":enroll_lengths, "simul_len":simul_lengths}


def get_dataloader(dataset_args, dataloader_args, eval=False):
    if dataset_args==None: return torch.utils.data.DataLoader([])
    if eval:
        dataset_args['deterministic'] = True
        dataloader_args['shuffle'] = False
        dataloader_args['num_workers'] = 0
        dataloader_args['persistent_workers'] = False
        dataloader_args['pin_memory'] = False
        dataloader_args['prefetch_factor'] = None
    dataset = AudioDataset(**dataset_args)
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, **dataloader_args)
    print(f"dataloader is loaded with eval {eval}. speaker num: {len(dataset.spks)}")
    return dataloader