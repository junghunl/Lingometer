import random, pickle, gzip
from collections import defaultdict
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, get_worker_info
from utils import *
from generate_dataset import simulate


class SimulOnlineAudioDataset(IterableDataset):
    def __init__(self, data_path, simul_args=None,
                 deterministic=False):
        """
        Args:
            data_path: list of paths to data[spk_idx]=[(feat, label)]
            simul_args: arguments for simulation (default: min_spk=1, max_spk=3, no_ts_prob=0.2)
        """
        super().__init__()
        
        #with gzip.open(data_path) as f: self.data = pickle.load(f)  # This causes RAM overload when multiple workers are used
        
        all_feats = []
        all_labels = []
        self.data_indices = []  # data_indices[spk_num][utt_num] = (start_idx, end_idx)
        with gzip.open(data_path, 'rb') as f: raw_data = pickle.load(f)
        curr_idx = 0
        for spk_utts in raw_data:
            spk_indices = []
            for feat, label in spk_utts:
                all_feats.append(feat)
                all_labels.append(label)
                feat_len = feat.shape[0] # T (시간 프레임 수)
                spk_indices.append((curr_idx, curr_idx+feat_len))
                curr_idx += feat_len
            self.data_indices.append(spk_indices)
        del raw_data
        self.mega_feat = torch.cat(all_feats, dim=0).share_memory_()
        self.mega_label = torch.cat(all_labels, dim=0).share_memory_()
        del all_feats, all_labels

        self.spk_num = len(self.data_indices)
        self.simul_args = simul_args if simul_args is not None else {}
        self.deterministic = deterministic

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(worker_id if self.deterministic else None)

        while True:
            #try:
            enroll_idx, conv_idx = simulate(self.data_indices, sample_num=1, rng=rng, **self.simul_args)
            enroll_spk, enroll_utt = enroll_idx
            e_start, e_end = self.data_indices[enroll_spk][enroll_utt]
            enroll_feat = self.mega_feat[e_start:e_end]
            conv_feats = []
            conv_labels = []
            for conv_spk, conv_utt in conv_idx:
                c_start, c_end = self.data_indices[conv_spk][conv_utt]
                conv_feats.append(self.mega_feat[c_start:c_end])
                conv_labels.append(self.mega_label[c_start:c_end])
            simul_feat = torch.cat(conv_feats, dim=0)
            if enroll_feat.shape[0] == 0 or simul_feat.shape[0] == 0: continue
            label = torch.cat(conv_labels, dim=0)
            label = (label==enroll_spk).long()
            
            yield {'enroll':enroll_feat, 'simul':simul_feat, 'label':label, 'spk':enroll_spk}
            #except Exception as e: print(e)


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
    dataset = SimulOnlineAudioDataset(**dataset_args)
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, **dataloader_args)
    print(f"dataloader is loaded with eval {eval}. speaker num: {dataset.spk_num}")
    return dataloader
