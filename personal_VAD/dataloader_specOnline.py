import random, pickle, gzip

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from personal_VAD.utils import *


def tstamp2framelabel(tstamps, labels):
    frame_labels = np.zeros(tstamps[-1]//10, dtype=int)
    start_frame = 0
    for end_time, label in zip(tstamps, labels):
        end_frame = end_time // 10
        frame_labels[start_frame:end_frame] = label
        start_frame = end_frame
    return torch.tensor(frame_labels)


def simulate(data:list[tuple[int,str,np.array,np.array]], spk_spans:list[tuple[int,int]], min_spk=1, max_spk=3, no_ts_prob=0.2, samesex=False, male_spks=None, female_spks=None):
    """
    input:
        data: individual utterences [(original_spk, feat, label)]
        spk_spans: spans for each compressed-id speaker
    output:
        simul_data: simulated conversation (target_spk, enroll_utt, utt_paths, lengths, labels, tstamps)
    """

    # sample speakers
    spk_num = np.random.randint(min_spk,max_spk+1)
    spk_pool = random.choice([male_spks, female_spks]) if samesex else np.arange(len(spk_spans)) 

    sampled_spks = np.random.choice(spk_pool, spk_num+1, replace=False) # last idx for enroll
    if no_ts_prob==0 or np.random.rand()>no_ts_prob: sampled_spks[-1] = np.random.choice(sampled_spks[:-1])
    sampled_spans = spk_spans[sampled_spks].numpy()
    sampled_idx = np.random.randint(sampled_spans[:,0], sampled_spans[:,1] + 1)
    
    target_spk = sampled_spks[-1]
    enroll_feat = data[sampled_idx[-1]][1]

    # sample simul utt
    simul_feats = []
    labels = []
    for spk, idx in zip(sampled_spks[:-1], sampled_idx[:-1]):
        _, feat, label, _ = data[idx]
        simul_feats.append(feat)
        if spk!=target_spk: label = torch.zeros_like(label)
        labels.append(label)

    simul_feat = torch.cat(simul_feats, dim=0)
    label = torch.cat(labels, dim=0)

    return enroll_feat, simul_feat, label, target_spk


class SpecOnlineAudioDataset(IterableDataset):
    def __init__(self, data_paths, fix_enroll_len=None, min_spk=1, max_spk=3, no_ts_prob=0.2, samesex=False):
        super().__init__()
        if isinstance(data_paths,str): data_paths=[data_paths]
        if fix_enroll_len is not None: print("NOTE: this dataset does not provide fix_enroll_len option")
        self.min_spk = min_spk
        self.max_spk = max_spk
        self.no_ts_prob = no_ts_prob
        self.samesex = samesex

        data = []
        for data_path in data_paths:
            with gzip.open(data_path) as f: new_data = pickle.load(f)
            for spk, samples in new_data.items():
                data.extend([(spk, feat, tstamp2framelabel(tstamp, label), gender) for feat, label, tstamp, gender in samples])
        data.sort(key=lambda x:x[0])
        self.data = data

        spk_spans = []
        prev_spk = -1
        for idx, (spk, *_) in enumerate(data):
            if prev_spk != spk:
                spk_spans.append([idx,idx])
                prev_spk = spk
            spk_spans[-1][1] = idx
        spk_spans = torch.tensor(spk_spans, dtype=torch.long)
        spk_spans.share_memory_()    
        self.spk_spans = spk_spans

        self.male_spks = []
        self.female_spks = []
        for spk, (start, end) in enumerate(self.spk_spans.tolist()):
            _, _, _, gender = self.data[start]
            if gender == True: self.male_spks.append(spk)
            elif gender == False: self.female_spks.append(spk)
            else: raise Exception()
        self.male_spks = np.array(self.male_spks, dtype=int)
        self.female_spks = np.array(self.female_spks, dtype=int)

    def __iter__(self):
        while True:
            #try:
            enroll_feat, simul_feat, label, target_spk = simulate(self.data, self.spk_spans,
                                                                  min_spk=self.min_spk, max_spk=self.max_spk, no_ts_prob=self.no_ts_prob,
                                                                  samesex=self.samesex, male_spks=self.male_spks, female_spks=self.female_spks)
            yield {'enroll':enroll_feat, 'simul':simul_feat, 'label':label, 'spk':target_spk}
            #except Exception as e: print(e)



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
    def _worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed)
        random.seed(seed)

    if dataset_args==None: return torch.utils.data.DataLoader([])
    if eval:
        dataset_args['fix_enroll_len'] = False
        dataloader_args['shuffle'] = False
        dataloader_args['num_workers'] = 0
        dataloader_args['persistent_workers'] = False
        dataloader_args['pin_memory'] = False
        dataloader_args['prefetch_factor'] = None
    dataset = SpecOnlineAudioDataset(**dataset_args)
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, worker_init_fn=_worker_init_fn, **dataloader_args)
    print(f"dataloader is loaded with eval {eval}. speaker num: {len(dataset.spk_spans)}")
    return dataloader

