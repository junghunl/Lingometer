import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Literal, Union, Dict


def _to_tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(dtype)
    if isinstance(x, (int, float)):
        return torch.tensor(x, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)

    return torch.tensor(x, dtype=dtype)

def pad_tensor(feature_list, dtype=torch.float32):
    lengths = torch.tensor([t.shape[0] for t in feature_list], dtype=torch.long)
    T_max = int(lengths.max())
    D = feature_list[0].shape[1]
    x_pad = torch.zeros(len(feature_list), T_max, D, dtype=dtype)
    for i, t in enumerate(feature_list):
        x_pad[i, : t.shape[0]] = torch.as_tensor(t, dtype=dtype)
    return x_pad, lengths

# -----------------------------------------------------------------
# collate_fn: Word/SyllableDataset (x, y_count, y_vec) → batch tensor
# -----------------------------------------------------------------
def pad_collate(batch):
    # unzip: list[(T_i, D)] , list[(1,)] , list[(maxT,)]
    feats, y_counts, y_vecs = zip(*batch)
    x_pad, lengths = pad_tensor(feats, dtype=torch.float32)      # (B, T_max, D) , (B,)
    y_scalar = torch.stack(y_counts).squeeze(1)                  # (B,)
    y_cum = torch.stack(y_vecs)                                  # (B, maxT)
    return x_pad, y_scalar, y_cum, lengths

# -----------------------------------------------------------------
# collate_fn for frame onset task: items are (x, T, frame_labels)
# Returns (X_pad, lengths, Y_frame) where Y_frame is (B, T_max)
# -----------------------------------------------------------------
def pad_collate_frame_onset(batch):
    feats = []
    lengths = []
    frame_labels = []
    for item in batch:
        if not isinstance(item, (tuple, list)) or len(item) not in (3,4):
            raise TypeError("pad_collate_frame_onset expects items (x, T, frame_labels[, gt_count])")
        if len(item) == 3:
            x, T, fl = item
            gt_count = None
        else:
            x, T, fl, gt_count = item
        x = _to_tensor(x)  # (t_i, D)
        if x.ndim != 2:
            raise ValueError(f"Feature tensor must be 2D (T,D); got shape={tuple(x.shape)}")
        feats.append(x)
        # Accept provided length T but sanity check
        true_T = x.size(0)
        if isinstance(T, torch.Tensor):
            T_val = int(T.item())
        else:
            T_val = int(T)
        if T_val != true_T:
            T_val = true_T  # override inconsistent length
        lengths.append(T_val)
        fl = _to_tensor(fl, dtype=torch.float32).view(-1)
        if fl.numel() < true_T:
            pad = torch.zeros(true_T - fl.numel(), dtype=torch.float32)
            fl = torch.cat([fl, pad], dim=0)
        elif fl.numel() > true_T:
            fl = fl[:true_T]
        frame_labels.append(fl)
        if gt_count is not None:
            # store scalar per-sample count
            if 'counts' not in locals():
                counts = []
            counts.append(float(gt_count))

    # pad features
    X_pad, lengths_tensor = pad_tensor(feats, dtype=torch.float32)
    T_max = X_pad.size(1)
    Y_pad = torch.zeros(len(frame_labels), T_max, dtype=torch.float32)
    for i, fl in enumerate(frame_labels):
        Y_pad[i, :fl.numel()] = fl
    if 'counts' in locals():
        counts_tensor = torch.tensor(counts, dtype=torch.float32)
        return X_pad, lengths_tensor, Y_pad, counts_tensor
    return X_pad, lengths_tensor, Y_pad



class SpeechCountDataset(Dataset):
    def __init__(
        self,
        X_list: Union[List[np.ndarray], Dict[str, np.ndarray]],
        y_meta: pd.DataFrame,
        target: Literal["syllable_count", "word_count", "frame_onset"] = "frame_onset",
        max_len: Optional[int] = None,
        syllable_max_len: Optional[int] = None,
        word_max_len: Optional[int] = None,
        frame_label_column: str = "frame_label",
        speaker_id_column: str = "speaker_id",
    ):
        self.X = X_list
        self.meta = y_meta
        self.target_key = target
        self.frame_label_column = frame_label_column
        self.speaker_id_column = speaker_id_column
        self.is_dict_features = isinstance(self.X, dict)

        # Set max lengths
        if target == "frame_onset":
            self.syllable_max_len = None
            self.word_max_len = None
        elif target == "syllable_count":
            self.syllable_max_len = syllable_max_len or max_len or 91
            self.word_max_len = None
        elif target == "word_count":
            self.word_max_len = word_max_len or max_len or 26
            self.syllable_max_len = None

    def __len__(self):
        return len(self.meta)

    def get_subject_id(self, idx):
        if self.speaker_id_column in self.meta.columns:
            return self.meta.iloc[idx][self.speaker_id_column]
        return "unknown"

    def __getitem__(self, idx):
        # Load Features
        if self.is_dict_features:
            utt_id = self.meta.iloc[idx]["utt_id"]
            if utt_id not in self.X:
                raise KeyError(f"utt_id '{utt_id}' not found in feature dictionary.")
            feat = self.X[utt_id]
        else:
            feat = self.X[idx]

        # Convert to Tensor
        if isinstance(feat, torch.Tensor):
            x = feat.detach().float()
        else:
            # Handle potential non-numeric garbage if necessary, 
            # but assuming clean input for simplicity as requested.
            x = torch.from_numpy(np.asarray(feat)).float()

        if x.ndim == 1:
            x = x.unsqueeze(-1)
        
        T = x.size(0)

        # Prepare Targets based on task
        if self.target_key == "frame_onset":
            return self._get_frame_onset(idx, x, T)
        else:
            # Single task (syllable_count or word_count)
            count = float(self.meta.iloc[idx][self.target_key])
            y_count = torch.tensor([count], dtype=torch.float32)
            
            max_len = self.syllable_max_len if self.target_key == "syllable_count" else self.word_max_len
            y_vec = torch.zeros(max_len, dtype=torch.float32)
            y_vec[: int(count)] = 1.0
            
            return x, y_count, y_vec

    def _get_frame_onset(self, idx, x, T):

        raw = self.meta.iloc[idx][self.frame_label_column]
        if isinstance(raw, str):
            parts = [p for p in raw.replace(";", ",").split(",") if p.strip() != ""]
            frame_labels = torch.tensor([int(float(p)) for p in parts], dtype=torch.float32)
        elif isinstance(raw, (list, tuple, np.ndarray)):
            frame_labels = torch.as_tensor(raw, dtype=torch.float32)
        else:
            frame_labels = torch.as_tensor(raw, dtype=torch.float32)

        if frame_labels.ndim == 0:
             frame_labels = frame_labels.unsqueeze(0)

        # Truncate or Pad
        if frame_labels.numel() < T:
            pad = torch.zeros(T - frame_labels.numel(), dtype=torch.float32)
            frame_labels = torch.cat([frame_labels, pad], dim=0)
        elif frame_labels.numel() > T:
            frame_labels = frame_labels[:T]

        # Get scalar count (prioritize meta columns over sum of frames)
        gt_count = float(frame_labels.sum().item())
        for col in ["word_count", "syllable_count"]:
            if col in self.meta.columns:
                try:
                    gt_count = float(self.meta.iloc[idx][col])
                    break
                except: pass
                
        return x, T, frame_labels, torch.tensor(gt_count, dtype=torch.float32)
    

__all__ = ["SpeechCountDataset", "pad_collate", "pad_collate_frame_onset"]