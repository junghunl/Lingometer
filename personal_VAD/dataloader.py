import torch
import gzip
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random


def tstamp2framelabel(tstamps, labels):
    #NOTE: spectrogram has 1 more frame than label
    frame_labels = np.zeros(tstamps[-1]//10+1, dtype=int)
    frame_labels[0] = labels[0]
    start_frame = 1
    for end_time, label in zip(tstamps, labels):
        end_frame = end_time // 10
        frame_labels[start_frame:end_frame+1] = label
        start_frame = end_frame+1
    return frame_labels


def get_random_chunk(data, chunk_len, rng=random):
    data_len = len(data)
    if data_len >= chunk_len:
        chunk_start = rng.randint(0, data_len - chunk_len)
        return data[chunk_start:chunk_start+chunk_len]
    else:
        idx = torch.arange(chunk_len) % data_len
        return data[idx]


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, simul_path, enroll_path, feat_path, fix_enroll_len=True, no_ts_prob=0.2, deterministic=False):
        self.fix_enroll_len = 5*100 if fix_enroll_len==True else False # nearly 5 seconds
        self.no_ts_prob = no_ts_prob
        self.deterministic = deterministic
        if isinstance(simul_path, str): simul_path = [simul_path]
        if isinstance(enroll_path, str): enroll_path = [enroll_path]
        if isinstance(feat_path, str): feat_path = [feat_path]

        self.simul_data = []
        self.simul_feats = []
        self.enroll_feats = {}
        for s_path, e_path, f_path in zip(simul_path, enroll_path, feat_path):
            with gzip.open(s_path, 'rb') as f: _, simul_data = pickle.load(f)
            self.simul_data.extend(simul_data)
            with gzip.open(e_path, 'rb') as f: self.enroll_feats.update(pickle.load(f))
            with gzip.open(f_path, 'rb') as f: self.simul_feats.extend(torch.load(f)) #, weights_only=False
        self.spks = sorted(self.enroll_feats.keys())
        self.spk2id = {spk:i for i,spk in enumerate(self.spks)}

    def __len__(self):
        return len(self.simul_data)

    def __getitem__(self, idx):
        rng = random.Random(idx) if self.deterministic else random
        sampled_spks, utt_paths, lengths, labels, tstamps = self.simul_data[idx]
        no_ts_prob = self.no_ts_prob*len(self.spks)/(len(self.spks)-len(sampled_spks))
        enroll_spk = rng.choice(sampled_spks)
        if rng.random()<no_ts_prob: enroll_spk = rng.choice(self.spks)
        enroll_feat = self.enroll_feats[enroll_spk]
        if self.fix_enroll_len: enroll_feat = get_random_chunk(enroll_feat, self.fix_enroll_len, rng)
        simul_feat = self.simul_feats[idx]
        labels = (labels==enroll_spk).astype(int)
        labels = torch.tensor(tstamp2framelabel(tstamps, labels))
        return {'enroll':enroll_feat, 'simul':simul_feat, 'label':labels, 'spk':self.spk2id[enroll_spk]}


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
        dataset_args['fix_enroll_len'] = False
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

