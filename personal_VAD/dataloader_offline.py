import random, pickle, gzip
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, get_worker_info
from utils import *


class AudioDataset(Dataset):
    def __init__(self, data_path,
                 min_enroll_len=5, fix_conv_len=0,
                 deterministic=False):
        """
        Args:
            data_path: path to data=[(feat, label, enroll_spans)]
        """
        self.min_enroll_len = min_enroll_len*100  # second to frame
        self.fix_conv_len = fix_conv_len*100  # second to frame
        self.deterministic = deterministic
        self.rng = None

        with gzip.open(data_path, "rb") as f: data = pickle.load(f)
        self.spk_num = int(max(label.max().item() for _, label, _ in data))+1  # spk index start from 0
        data = [
            (feat, label, valid_spans) 
            for feat, label, spans in data 
            if (valid_spans := [span for span in spans if span[2]-span[1]>=self.min_enroll_len and feat.shape[0]>span[2]-span[1]])
        ]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.rng is None:
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            self.rng = np.random.default_rng(worker_id if self.deterministic else None)
        feat, label, enroll_spans = self.data[idx]
        enroll_spk,s,e = self.rng.choice(enroll_spans)
        label = (label==enroll_spk).long()
        enroll_feat = feat[s:e,:]
        simul_feat = torch.cat([feat[:s,:],feat[e:,:]], dim=0)
        label = torch.cat([label[:s],label[e:]], dim=0)
        if self.fix_conv_len and self.fix_conv_len < len(label):
            simul_feat, label = get_random_chunk(simul_feat, label, chunk_len=self.fix_conv_len, rng=self.rng)
        if enroll_feat.shape[0]==0 or simul_feat.shape[0]==0:  # for safe
            new_idx = self.rng.integers(0, len(self.data))
            return self.__getitem__(new_idx)
        return {'enroll':enroll_feat, 'simul':simul_feat, 'label':label, 'spk':enroll_spk}


def collate_fn(batch):
    enrolls = [item["enroll"] for item in batch]
    enrolls = pad_sequence(enrolls, batch_first=True, padding_value=0)
    enrolls = enrolls.transpose(1,2)

    simuls = [item["simul"] for item in batch]
    simuls = pad_sequence(simuls, batch_first=True, padding_value=0)
    simuls = simuls.transpose(1,2)
    
    labels = [item["label"] for item in batch]
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)

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
    print(f"dataloader is loaded with eval {eval}. speaker num: {dataset.spk_num}")
    return dataloader

