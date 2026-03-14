import os, math, random, pickle, gzip, warnings, json, gc
from pathlib import Path
from os.path import join as pjoin
from collections import defaultdict
from decimal import Decimal
from multiprocessing import Pool, cpu_count
from functools import wraps
from functools import partial

from tqdm.auto import tqdm
import numpy as np
import torch
import torchaudio
from textgrid import TextGrid
import xml.etree.ElementTree as ET

from utils import validate_path, mono_resample, Augmentor, Extractor

# ignore torchaudio warnings
warnings.filterwarnings("ignore", message=".*torchaudio.load_with_torchcodec.*")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")



def tstamp2framelabel(tstamps, labels):
    if len(tstamps) == 0: return np.array([], dtype=int)
    frame_labels = np.zeros(tstamps[-1]//10, dtype=int)
    start_frame = 0
    for end_time, label in zip(tstamps, labels):
        end_frame = end_time // 10
        frame_labels[start_frame:end_frame] = label
        start_frame = end_frame
    return frame_labels


def cache_file(func):
    """
    Decorator to cache the output of a function to a file.
    Args:
        save_path: path to save the cached file (*.pkl.gz)
        overwrite: whether to overwrite the cached file if it exists
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        save_path = kwargs.get('save_path', None)
        overwrite = kwargs.get('overwrite', False)

        if save_path and os.path.exists(save_path) and not overwrite:
            print(f"Load cached file from {save_path}")
            with gzip.open(save_path, 'rb') as f: return pickle.load(f)

        result = func(*args, **kwargs)

        if save_path:
            print(f"Save cache file to {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with gzip.open(save_path, 'wb') as f: pickle.dump(result, f)

        return result
    return wrapper





#NOTE: To remove dependency on installed location of data_dir, load_{dataset} functions don't include data_dir in output data, and generate_dataset functions take data_dir as input.

#---------------------LibriSpeech loaders---------------------#

@cache_file
def load_librispeech(data_dir:str, algn_dir:str, splits:list, **kwargs):
    """
    Returns:
        data[spk_idx]=[(wav_path, label)]
    """
    if isinstance(splits, str): splits = [splits]
    
    def scan_librispeech_dir(root_dir, splits, target_ext, desc=""):
        for split in splits:
            split_dir = pjoin(root_dir,split)
            for spk_id in tqdm(os.listdir(split_dir), desc=f"{desc} ({split})"):
                spk_dir = pjoin(split_dir, spk_id)
                if not os.path.isdir(spk_dir): continue
                for chp_id in os.listdir(spk_dir):
                    chp_dir = pjoin(spk_dir, chp_id)
                    if not os.path.isdir(chp_dir): continue
                    for filename in os.listdir(chp_dir):
                        if filename.endswith(target_ext):
                            utt_id = filename[:-len(target_ext)]
                            yield pjoin(split,spk_id), chp_id, utt_id

    def get_label(algn_path):
        # get label and end times(in milliseconds) from TextGrid file
        try: tg = TextGrid.fromFile(algn_path)
        except Exception as e: raise RuntimeError(f"Failed to open file: [{type(e).__name__}] {e}")
        word_tier = tg.getFirst("words")
        label = np.array([(1 if interval.mark else 0) for interval in word_tier.intervals])
        tstamp = np.array([int(Decimal(str(interval.maxTime))*1000) for interval in word_tier.intervals])
        if tstamp[-1]!=int(Decimal(str(tg.maxTime))*1000): raise ValueError(f"Max time label is inconsistent in the file {algn_path}: {tstamp[-1]}!={tg.maxTime}")
        tstamp = tstamp//10*10  # ensure the timestamp is 10ms unit for the 10ms unit label
        return tstamp2framelabel(tstamp, label)
    
    spks = set()
    miss_utts = set()
    for spk_id, chp_id, utt_id in scan_librispeech_dir(data_dir, splits, '.flac', "Collecting audio files"):
        spks.add(spk_id)
        miss_utts.add(pjoin(spk_id,chp_id,utt_id))
    spk2idx = {spk: i for i,spk in enumerate(sorted(spks))}

    miss_labels = set()
    crash_labels = set()
    data = [[] for _ in range(len(spk2idx))]  # data[spk_idx]=[(utt_path, label)]
    for spk_id, chp_id, utt_id in scan_librispeech_dir(algn_dir, splits, '.TextGrid', "Processing alignments"):        
        file_id = pjoin(spk_id,chp_id,utt_id)
        if file_id not in miss_utts: miss_labels.add(file_id); continue
        else: miss_utts.remove(file_id)
        spk_idx = spk2idx[spk_id]
        try: label = get_label(pjoin(algn_dir,file_id+'.TextGrid'))
        except Exception as e: print("skipped error:",e); crash_labels.add(file_id); continue
        label = np.where(label==1, spk_idx, -1)
        data[spk_idx].append((file_id+'.flac', label))
    
    print(f'{len(miss_utts)} missing utts:', *miss_utts)
    print(f'{len(miss_labels)} missing labels:', *miss_labels)
    print(f'{len(crash_labels)} crached labels:', *crash_labels)

    return data


@cache_file
def simulate(data:tuple[tuple[np.array, np.array]], sample_num=None,
             min_spk=1, max_spk=3, no_ts_prob=0.2,
             samesex=False, male_spks=None, female_spks=None,
             rng=None, **kwargs):
    """
    Args:
        data: individual utt data[spk_idx]=[(wav_path or spec, label)]
        sample_num : max if sample_num=None
        
    Returns:
        simul_info : simulated conversations [(enroll_idx, [conv_idx])]
        if sample_num==1: return single (enroll_idx, [conv_idx]) pair
            where idx = (spk_idx, utt_idx)
    """
    if rng is None: rng = np.random.default_rng()

    if sample_num==1:
        # optimized for speed to train
        spk_num = rng.integers(min_spk,max_spk+1)
        spk_pool = rng.choice([male_spks, female_spks]) if samesex else np.arange(len(data)) 
        sampled_spks = rng.choice(spk_pool, spk_num+1, replace=False) # last idx for enroll
        if no_ts_prob==0 or rng.random()>no_ts_prob: sampled_spks[-1] = rng.choice(sampled_spks[:-1])
        sampled_len = np.array([len(data[spk]) for spk in sampled_spks])
        sampled_idx = rng.integers(0,sampled_len)
        data_idx = list(zip(sampled_spks.tolist(), sampled_idx.tolist()))
        enroll_idx = data_idx[-1]
        conv_idx = data_idx[:-1]
        return enroll_idx, conv_idx
    else:
        # not implemented samesex version yet
        data_indices = [rng.permutation(len(utts)).tolist() for utts in data]  # shuffle for randomness
        spk_pool1 = [i for i, utts in enumerate(data_indices) if len(utts)>=1]
        spk_pool2 = [i for i, utts in enumerate(data_indices) if len(utts)>=2]
        def extract_utt_and_update_pool(spk_idx):
            utt = data_indices[spk_idx].pop()
            if len(data_indices[spk_idx])<1 and spk_idx in spk_pool1: spk_pool1.remove(spk_idx)
            if len(data_indices[spk_idx])<2 and spk_idx in spk_pool2: spk_pool2.remove(spk_idx)
            return utt

        simul_info = []
        while sample_num is None or len(simul_info)<sample_num:
            if len(spk_pool2)<1 or len(spk_pool1)<2: break
            spk_num = min(rng.integers(min_spk,max_spk+1), len(spk_pool1)-1)  # -1 for oversampling  
            if no_ts_prob==0 or rng.random()>no_ts_prob:
                enroll_spk = rng.choice(spk_pool2)
                sampled_spks = rng.choice(spk_pool1, spk_num, replace=False)
                if enroll_spk not in sampled_spks: sampled_spks[rng.integers(0,spk_num)] = enroll_spk
            else:
                sampled_spks = rng.choice(spk_pool1, spk_num+1, replace=False)  # last idx for enroll
                enroll_spk = sampled_spks[-1]
                sampled_spks = sampled_spks[:-1]
            enroll_idx = (int(enroll_spk), extract_utt_and_update_pool(enroll_spk))
            conv_idx = [(int(spk), extract_utt_and_update_pool(spk)) for spk in sampled_spks]
            simul_info.append((enroll_idx, conv_idx))
        print(f"Generated {len(simul_info)} simulated conversations.")

        return simul_info








#---------------------AMI/CHiME loaders---------------------#

ami_valid_convs = {"ES2003a", "ES2003b", "ES2003c", "ES2003d", "ES2011a", "ES2011b", "ES2011c", "ES2011d", "IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011", "IS1008a", "IS1008b", "IS1008c", "IS1008d", "TS3004a", "TS3004b", "TS3004c", "TS3004d", "TS3006a", "TS3006b", "TS3006c", "TS3006d"}
ami_test_convs = {"EN2002a", "EN2002b", "EN2002c", "EN2002d", "ES2004a", "ES2004b", "ES2004c", "ES2004d", "ES2014a", "ES2014b", "ES2014c", "ES2014d", "IS1009a", "IS1009b", "IS1009c", "IS1009d", "TS3003a", "TS3003b", "TS3003c", "TS3003d", "TS3007a", "TS3007b", "TS3007c", "TS3007d"}
ami_valid_and_test_convs = ami_valid_convs|ami_test_convs
@cache_file
def load_ami(data_dir:str=None, algn_dir:str=None, conv_filter='train', **kwargs):
    """
    Returns:
        data[spk_idx]=(wav_path, label, utt_spans)
    """
    nospeak_spks = {'EN2009c_D', 'EN2002c_D', 'EN2003a_D', 'IN1001_D', 'EN2009b_D'}

    def get_label_ami(algn_path):
        conv, spk = os.path.basename(algn_path).split('.')[-4:-2]
        spk = f"{conv}_{spk}"
        label = []
        tstamp = []
        utt_spans = []

        word_tree = ET.parse(algn_path)
        word_root = word_tree.getroot()
        for seg in word_root.findall("w"):
            if seg.get('punc') == 'true': continue
            try:
                start_ms = int(Decimal(seg.attrib['starttime'])*100)*10
                end_ms = int(Decimal(seg.attrib['endtime'])*100)*10
            except: continue
            label.append(0)
            tstamp.append(start_ms)
            label.append(1)
            tstamp.append(end_ms)

        seg_tree = ET.parse(algn_path.replace('words', 'segments'))
        seg_root = seg_tree.getroot()
        for seg in seg_root.findall("segment"):
            try:
                u_start = int(Decimal(seg.attrib['transcriber_start'])*100)
                u_end = int(Decimal(seg.attrib['transcriber_end'])*100)
            except: continue
            utt_spans.append((u_start, u_end))

        return tstamp2framelabel(tstamp, label), utt_spans

    spks = set()
    for conv_id in tqdm(os.listdir(data_dir), desc="Collecting audio files"):
        conv_dir = pjoin(data_dir,conv_id)
        if not os.path.isdir(conv_dir): continue
        if conv_filter=='train' and conv_id in ami_valid_and_test_convs: continue
        elif conv_filter!='train' and conv_id not in conv_filter: continue
        for utt_id in os.listdir(pjoin(conv_dir,"audio")):
            if utt_id.endswith('.wav'):
                spk_id = conv_id+'_'+chr(ord('A')+int(utt_id.split('.')[1][8:]))
                if spk_id not in nospeak_spks: spks.add(spk_id)
    spk2idx = {spk: i for i,spk in enumerate(sorted(spks))}
    
    data = [None]*len(spk2idx)  # data[spk_idx]=(conv_path, label, utt_spans)
    for utt_id in tqdm(os.listdir(algn_dir), desc="Processing alignments"):
        if not utt_id.endswith('.words.xml'): continue
        conv_id, spk_chr, *_ = utt_id.split('.')
        spk_id = f"{conv_id}_{spk_chr}"
        if spk_id not in spk2idx: continue
        spk_idx = spk2idx[spk_id]
        label, utt_spans = get_label_ami(pjoin(algn_dir,utt_id))
        if len(label) == 0 or len(utt_spans) == 0:
            print('no speak:', utt_id)
            continue
        label = np.where(label==1, spk_idx, -1)
        conv_path = pjoin(conv_id,"audio",f"{conv_id}.Headset-{ord(spk_chr)-ord('A')}.wav")
        data[spk_idx] = (conv_path, label, utt_spans)

    return data


@cache_file
def load_chime(data_dir:str=None, algn_dir:str=None, **kwargs):
    """
    Returns:
        data[spk_idx]=(wav_path, label, utt_spans)
    """

    def get_label_chime(algn_path):
        conv_id = os.path.basename(algn_path)[:-5]
        spk2label = defaultdict(list)
        spk2tstamp = defaultdict(list)
        spk2utt = defaultdict(list)

        with open(algn_path, "r", encoding="utf-8") as f: words = json.load(f)
        words.sort(key=lambda x: (x["speaker"], float(x["start_time"])))
        for seg in words:
            spk = f"{conv_id}_{seg['speaker']}"
            start_ms = int(Decimal(seg["start_time"])*100)*10
            end_ms = int(Decimal(seg["end_time"])*100)*10
            spk2label[spk].append(0)
            spk2tstamp[spk].append(start_ms)
            spk2label[spk].append(1)
            spk2tstamp[spk].append(end_ms)
        
        time_to_seconds = lambda t: sum(c*Decimal(x) for c,x in zip([3600,60,1],t.split(':')))
        with open(algn_path.replace("align","transcriptions"), "r", encoding="utf-8") as f: utts = json.load(f)
        utts.sort(key=lambda x: (x["speaker"], time_to_seconds(seg["start_time"])))
        for seg in utts:
            spk = spk = f"{conv_id}_{seg['speaker']}"
            start_ms = int(time_to_seconds(seg["start_time"])*100)
            end_ms = int(time_to_seconds(seg["end_time"])*100)
            spk2utt[spk].append((start_ms, end_ms))

        return {spk: (tstamp2framelabel(spk2tstamp[spk],spk2label[spk]), spk2utt[spk]) for spk in spk2label.keys()}

    spks = set()
    for utt_id in tqdm(os.listdir(data_dir), desc="Collecting audio files"):
        if not utt_id.endswith('.wav') or '_P' not in utt_id: continue
        spk_id = utt_id[:-4]
        spks.add(spk_id)
    spk2idx = {spk: i for i,spk in enumerate(sorted(spks))}

    spk2label = {}
    for conv_id in tqdm(os.listdir(algn_dir), desc="Processing alignments"):
        if not conv_id.endswith('.json'): continue
        conv_labels = get_label_chime(pjoin(algn_dir,conv_id))
        spk2label.update(conv_labels)
        spks |= set(conv_labels.keys())
    
    data = [None]*len(spk2idx)  # data[spk_idx]=(conv_path, label, utt_spans)
    for spk, (label, utt_spans) in spk2label.items():
        if spk not in spk2idx: continue
        if len(label) == 0 or len(utt_spans) == 0:
            print('no speak!')
            continue
        spk_idx = spk2idx[spk]
        label = np.where(label==1, spk_idx, -1)
        data[spk_idx] = (spk+'.wav', label, utt_spans)

    return data









#---------------------dataset generators---------------------#

@cache_file
def generate_online_dataset_from_utt(
        data_dir, data,
        feat_args:dict=None, aug_args:dict=None,
        **kwargs):
    """
    Args:
        data: individual utt data[spk_idx]=[(wav_path, label)]
    
    Returns:
        feats[spk_idx]=[(spec, label)]
        where spec[time,freq]
    """
    if aug_args is None : aug_args = {}
    if feat_args is None : feat_args = {}
    augmentor = Augmentor(**aug_args)
    extractor = Extractor(**feat_args)
    feats = [[] for _ in range(len(data))]
    for spk_idx in tqdm(range(len(data)), desc="Generating online dataset"):
        for utt_path, label in data[spk_idx]:
            wav, sr = torchaudio.load(pjoin(data_dir,utt_path))  # (1,T)
            wav = augmentor(wav)
            feat = extractor(wav, len(label))  # (T,F)
            feats[spk_idx].append((feat, torch.Tensor(label)))
    return feats


@cache_file
def generate_offline_dataset_from_utt(
        data_dir, data, simul_info,
        feat_args:dict=None, aug_args:dict=None,
        **kwargs):
    """
    Args:
        data: individual utt data[spk_idx]=[(wav_path, label)]
        simul_info : simulated conversations [(enroll_idx, [conv_idx])]
            where idx = (spk_idx, utt_idx)
    
    Returns:
        feats = [(spec, label, enroll_spans)]
            where spec[time,freq], enroll_spans = ((spk_idx, utt_start, utt_end),)
    """
    if aug_args is None : aug_args = {}
    if feat_args is None : feat_args = {}
    augmentor = Augmentor(**aug_args)
    extractor = Extractor(**feat_args)
    feats = []
    for enroll_idx, conv_idx in tqdm(simul_info, desc="Generating offline dataset"):
        enroll_spk, enroll_utt = enroll_idx
        enroll_path, enroll_label = data[enroll_spk][enroll_utt]
        enroll_wav = mono_resample(*torchaudio.load(pjoin(data_dir,enroll_path)))  # (1,T)
        enroll_wav = augmentor.augment(enroll_wav)
        enroll_feat = extractor.extract(enroll_wav, len(enroll_label))

        conv_wavs = []
        conv_labels = []
        for conv_spk, conv_utt in conv_idx:
            conv_path, conv_label = data[conv_spk][conv_utt]
            conv_wav = mono_resample(*torchaudio.load(pjoin(data_dir,conv_path)))
            conv_wavs.append(conv_wav)
            conv_labels.append(torch.Tensor(conv_label))
        conv_labels = torch.cat(conv_labels, dim=0)
        conv_wavs = torch.cat(conv_wavs, dim=-1)
        conv_wavs = augmentor(conv_wavs)
        conv_feat = extractor(conv_wavs, len(conv_labels))

        feats.append((torch.cat([enroll_feat, conv_feat], dim=0),
                     torch.cat([torch.Tensor(enroll_label), conv_labels], dim=0),
                     ((enroll_spk,0,len(enroll_feat)),)))

    return feats


@cache_file
def generate_offline_dataset_from_conv(
        data_dir, data, select_enroll=False, min_enroll_len=5,
        feat_args:dict=None, aug_args:dict=None,
        **kwargs):
    """
    Args:
        data: individual conv data[spk_idx]=(conv_path, label, utt_spans)
    
    Returns:
        feats = [(spec, label, enroll_spans)]
            where spec[time,freq], enroll_spans = ((spk_idx, utt_start, utt_end),)
    """
    if aug_args is None : aug_args = {}
    if feat_args is None : feat_args = {}
    min_enroll_len = min_enroll_len*100  # second to frame
    augmentor = Augmentor(**aug_args)
    extractor = Extractor(**feat_args)
    feats = []
    if select_enroll:
        # select only one enrollment utterance randomly
        for spk_idx, (conv_path, conv_label, utt_spans) in enumerate(tqdm(data, desc="Generating offline dataset")):
            conv_label = torch.Tensor(conv_label)
            valid_spans = [span for span in utt_spans if (span[1]-span[0])>=min_enroll_len]
            if valid_spans: s, e = random.choice(valid_spans)
            else: s, e = max(utt_spans, key=lambda x: x[1]-x[0])
            conv_wav = mono_resample(*torchaudio.load(pjoin(data_dir,conv_path)))  # (1,T)
            enroll_wav = conv_wav[:,s*160:e*160]  # 10ms frame -> 160 sample
            enroll_wav = augmentor.augment(enroll_wav)
            enroll_feat = extractor.extract(enroll_wav, e-s)
            conv_wav  = torch.cat([conv_wav[:,:s*160], conv_wav[:,e*160:]], dim=1)
            conv_wav = augmentor.augment(conv_wav)
            conv_feat = extractor.extract(conv_wav, len(conv_label)-(e-s))
            feats.append((torch.cat([enroll_feat, conv_feat], dim=0),
                         torch.cat([conv_label[s:e], conv_label[:s], conv_label[e:]], dim=0),
                         ((spk_idx,0,len(enroll_feat)),)))
    else:
        for spk_idx, (conv_path, conv_label, utt_spans) in enumerate(tqdm(data, desc="Generating offline dataset")):
            conv_wav = mono_resample(*torchaudio.load(pjoin(data_dir,conv_path)))
            conv_wav = augmentor.augment(conv_wav)
            conv_feat = extractor.extract(conv_wav, len(conv_label))
            feats.append((conv_feat, torch.Tensor(conv_label), tuple((spk_idx,s,e) for s,e in utt_spans)))
        
    return feats








#---------------------Multiprocessing versions---------------------#

global_augmentor = None
global_extractor = None
global_rng = None
def init_worker(feat_args, aug_args):
    global global_augmentor, global_extractor, global_rng
    torch.set_num_threads(1) 
    if aug_args is None: aug_args = {}
    if feat_args is None: feat_args = {}
    global_augmentor = Augmentor(**aug_args)
    global_extractor = Extractor(**feat_args)
    global_rng = np.random.default_rng(None)


def _worker_online_utt(args):
    spk_idx, spk_utts, data_dir = args
    spk_feats = []
    for utt_path, label in spk_utts:
        wav, sr = torchaudio.load(os.path.join(data_dir, utt_path))
        wav = global_augmentor(wav)
        feat = global_extractor(wav, len(label))
        spk_feats.append((feat, torch.Tensor(label)))
    return spk_idx, spk_feats


def _worker_offline_utt(args):
    simul_idx, enroll_req, conv_reqs, data_dir = args
    enroll_spk, enroll_path, enroll_label = enroll_req
    enroll_wav = mono_resample(*torchaudio.load(os.path.join(data_dir, enroll_path)))
    enroll_wav = global_augmentor.augment(enroll_wav)
    enroll_feat = global_extractor.extract(enroll_wav, len(enroll_label))

    conv_wavs = []
    conv_labels = []
    for conv_path, conv_label in conv_reqs:
        conv_wav = mono_resample(*torchaudio.load(os.path.join(data_dir, conv_path)))
        conv_wavs.append(conv_wav)
        conv_labels.append(torch.Tensor(conv_label))
    conv_labels = torch.cat(conv_labels, dim=0)
    conv_wavs = torch.cat(conv_wavs, dim=-1)
    conv_wavs = global_augmentor(conv_wavs)
    conv_feat = global_extractor(conv_wavs, len(conv_labels))

    final_feat = torch.cat([enroll_feat, conv_feat], dim=0)
    final_label = torch.cat([torch.Tensor(enroll_label), conv_labels], dim=0)
    spans = ((enroll_spk, 0, len(enroll_feat)),)
    return simul_idx, (final_feat, final_label, spans)


def _worker_offline_conv(args):
    spk_idx, conv_path, conv_label, utt_spans, data_dir, select_enroll = args    
    conv_label_tensor = torch.Tensor(conv_label)
    conv_wav_raw = mono_resample(*torchaudio.load(os.path.join(data_dir, conv_path)))
    if select_enroll:
        span_idx = global_rng.choice(len(utt_spans))
        s, e = utt_spans[span_idx]
        enroll_wav = conv_wav_raw[:, s*160 : e*160]
        enroll_wav = global_augmentor.augment(enroll_wav)
        enroll_feat = global_extractor.extract(enroll_wav, e-s)
        
        conv_wav = torch.cat([conv_wav_raw[:, :s*160], conv_wav_raw[:, e*160:]], dim=1)
        conv_wav = global_augmentor.augment(conv_wav)
        conv_feat = global_extractor.extract(conv_wav, len(conv_label_tensor) - (e-s))
        
        final_feat = torch.cat([enroll_feat, conv_feat], dim=0)
        final_label = torch.cat([conv_label_tensor[s:e], conv_label_tensor[:s], conv_label_tensor[e:]], dim=0)
        spans = ((spk_idx, 0, len(enroll_feat)),)
        return spk_idx, (final_feat, final_label, spans)
    else:
        conv_wav = global_augmentor.augment(conv_wav_raw)
        conv_feat = global_extractor.extract(conv_wav, len(conv_label_tensor))
        spans = tuple((spk_idx, s, e) for s, e in utt_spans)
        return spk_idx, (conv_feat, conv_label_tensor, spans)



@cache_file
def generate_online_dataset_from_utt_parallel(data_dir, data, feat_args=None, aug_args=None, num_workers=None, **kwargs):
    if num_workers is None: num_workers = cpu_count()-1
    worker_args = [(spk_idx, utts, data_dir) for spk_idx, utts in enumerate(data)]
    feats = [[] for _ in range(len(data))]
    with Pool(num_workers, initializer=init_worker, initargs=(feat_args, aug_args)) as p:
        for spk_idx, spk_feats in tqdm(p.imap_unordered(_worker_online_utt, worker_args), total=len(worker_args), desc="Generating online dataset"):
            feats[spk_idx] = spk_feats
    return feats


@cache_file
def generate_offline_dataset_from_utt_parallel(data_dir, data, simul_info, feat_args=None, aug_args=None, num_workers=None, **kwargs):
    if num_workers is None: num_workers = cpu_count()-1
    # extract information from data before multiprocessing
    worker_args = []
    for simul_idx, (enroll_idx, conv_idx) in enumerate(simul_info):
        enroll_spk, enroll_utt = enroll_idx
        enroll_path, enroll_label = data[enroll_spk][enroll_utt]
        enroll_req = (enroll_spk, enroll_path, enroll_label)
        conv_reqs = []
        for conv_spk, conv_utt in conv_idx:
            conv_path, conv_label = data[conv_spk][conv_utt]
            conv_reqs.append((conv_path, conv_label))
        worker_args.append((simul_idx, enroll_req, conv_reqs, data_dir))
    feats = [None] * len(simul_info)
    with Pool(num_workers, initializer=init_worker, initargs=(feat_args, aug_args)) as p:
        for simul_idx, feat_data in tqdm(p.imap_unordered(_worker_offline_utt, worker_args), total=len(worker_args), desc="Generating offline dataset (utt)"):
            feats[simul_idx] = feat_data  
    return feats


@cache_file
def generate_offline_dataset_from_conv_parallel(data_dir, data, select_enroll=False, feat_args=None, aug_args=None, num_workers=None, **kwargs):
    if num_workers is None: num_workers = cpu_count()-1
    worker_args = [
        (spk_idx, conv_path, conv_label, utt_spans, data_dir, select_enroll) 
        for spk_idx, (conv_path, conv_label, utt_spans) in enumerate(data)
    ]
    feats = [None] * len(data)
    with Pool(num_workers, initializer=init_worker, initargs=(feat_args, aug_args)) as p:
        for spk_idx, feat_data in tqdm(p.imap_unordered(_worker_offline_conv, worker_args), total=len(worker_args), desc="Generating offline dataset (conv)"):
            feats[spk_idx] = feat_data
    return feats








""" examples

feat_args = {'n_mels':24}
aug_args = {'rir_paths':..., 
            'noise_paths':..., 
            'rir_prob':0.5, 'noise_prob':0.5}

            
            
# AMI dataset
ami_data_dir = ...  # 'AMI/raw/headset'
ami_align_dir = ...  # 'AMI/ami_manual_1.6.1/words'
ami_processed_dir = ...
splits = ["train","valid","test"]
for split in splits:
  if split == 'train': conv_filter='train'
  elif split == 'valid': conv_filter = ami_valid_convs
  elif split == 'test': conv_filter = ami_test_convs
  ami_data = load_ami(ami_data_dir, ami_align_dir, conv_filter=conv_filter, save_path=pjoin(ami_processed_dir,split,'data.pkl.gz'))
  if split=='train':
    generate_offline_dataset_from_conv_parallel(ami_data_dir, ami_data,
        feat_args=feat_args, aug_args=aug_args, save_path=pjoin(ami_processed_dir,split,'feat_24dim_aug.pkl.gz'), overwrite=True)
  else:
    generate_offline_dataset_from_conv_parallel(ami_data_dir, ami_data, select_enroll=True,
        feat_args=feat_args, save_path=pjoin(ami_processed_dir,split,'feat_24dim_noaug_selectEnroll.pkl.gz'), overwrite=True)

        
        
# CHiME dataset
chime_data_dir = ... # 'CHiME/raw'
chime_align_dir = ... # 'CHiME/align'
chime_processed_dir = ...
splits = ["eval","dev","train"]
for split in splits:
    data_dir = pjoin(chime_data_dir,f"CHiME6_{split}",'audio',split) if split!='train' else pjoin(chime_data_dir,f"CHiME6_{split}",'CHiME6','audio',split)
    algn_dir = pjoin(chime_align_dir, split)
    chime_data = load_chime(data_dir, algn_dir, save_path=pjoin(chime_processed_dir,split,'data.pkl.gz'), overwrite=True)
    if split=='train':
      generate_offline_dataset_from_conv_parallel(data_dir, chime_data,
          feat_args=feat_args, aug_args=aug_args, save_path=pjoin(chime_processed_dir,split,'feat_24dim_aug.pkl.gz'), overwrite=True)
    else:
      generate_offline_dataset_from_conv_parallel(data_dir, chime_data, select_enroll=True,
          feat_args=feat_args, save_path=pjoin(chime_processed_dir,split,'feat_24dim_noaug_selectEnroll.pkl.gz'), overwrite=True)

          

# LibriSpeech dataset
libri_data_dir = ... # "LibriSpeech"
libri_algn_dir = ... # "librispeech_alignments"
libri_processed_dir = ...
for split in ['dev', 'test']:
  libri_data = load_librispeech(libri_data_dir, libri_algn_dir,
      splits=[f"{split}-clean",f"{split}-other"], save_path=pjoin(libri_processed_dir,f'{split}','data.pkl.gz'))
  simul_info = simulate(libri_data, save_path=pjoin(libri_processed_dir,f'{split}','simul.pkl.gz'))
  generate_offline_dataset_from_utt_parallel(libri_data_dir, libri_data, simul_info,
      feat_args=feat_args, save_path=pjoin(libri_processed_dir,f'{split}','feat_24dim_noaug_selectEnroll.pkl.gz'))
# train
libri_data = load_librispeech(libri_data_dir, libri_algn_dir,
    splits=["train-clean-100", "train-clean-360", "train-other-500"], save_path=pjoin(libri_processed_dir,'train','data.pkl.gz'))
NUM_CHUNKS = 4
chunk_size = math.ceil(len(libri_data) / NUM_CHUNKS)
data_chunks = [libri_data[i : i + chunk_size] for i in range(0, len(libri_data), chunk_size)]
for i, chunk_data in enumerate(data_chunks):
    save_path = pjoin(libri_processed_dir, 'train', f'feat_24dim_aug_part_{i}.pkl.gz')
    if os.path.exists(save_path): continue
    generate_online_dataset_from_utt_parallel(libri_data_dir, chunk_data,
        feat_args=feat_args, aug_args=aug_args, save_path=save_path)
    gc.collect()
merged_data = []
for i in range(NUM_CHUNKS):
    part_path = pjoin(libri_processed_dir, 'train', f'feat_24dim_aug_part_{i}.pkl.gz')
    with gzip.open(part_path, 'rb') as f:
        chunk = pickle.load(f)
        merged_data.extend(chunk)
    del chunk
    gc.collect()
final_save_path = pjoin(libri_processed_dir, 'train', 'feat_24dim_aug.pkl.gz')
with gzip.open(final_save_path, 'wb') as f: pickle.dump(merged_data, f)
"""