# %%
import matplotlib.pyplot as plt
import pickle 
import numpy as np
import os 
import random
from global_config import globals
from tqdm import tqdm
import torch
import sys
sys.path.append('../')
from deepspt_src import (add_features, load_3D_data, curate_3D_data_to_tracks, 
                  handle_compound_tracks, fuse_tracks, flatten_list)
from deepspt_src.coloc_helpers import *
from glob import glob
import time 
from joblib import Parallel, delayed


# global config variables
globals._parse({})

# set seed
seed = globals.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Prep data

# Define where to find experiments

# '20230413_p5_p55_sCMOS_Jacob_Mari_Anand_Ricky'
# '20230415_p5_p55_sCMOS_Jacob_Mari_Gu_Ricky' - # CS6+ has more laser power

savepath = '../deepspt_results/EEA1_NPC1_results/precomputed_files/coloc_results/coloc_rota'

main_dirs = ['../_Data/LLSM_data/EEA1_NPC1_Rotavirus/20230413_p5_p55_sCMOS_Jacob_Mari_Anand_Ricky',
             '../_Data/LLSM_data/EEA1_NPC1_Rotavirus/20230415_p5_p55_sCMOS_Jacob_Mari_Gu_Ricky',
             '../_Data/LLSM_data/EEA1_NPC1_Rotavirus/20230419_p5_p55_sCMOS_Jacob_Mari_Anand_Ricky']

SEARCH_PATTERN = '{}/**/ProcessedTracks.mat'
OUTPUT_NAME = 'rotavirus'

files = []
for main_dir in main_dirs:
    _input = SEARCH_PATTERN.format(main_dir)

    files.append(sorted(glob(_input, recursive=True)))
files = np.concatenate(files)

dont_use_experiments = []

files_curated = []
for file in files:
    # temperature issues due to breakdown of aircon at Harvard Medical School 2023/04/13
    if 'CS8' in file or 'CS9' in file: 
        continue
    # memory issue stopped video
    if 'Ex33' in file: 
        continue
    if 'Ex34' in file: 
        continue
    else:
        files_curated.append(file)

xy_to_um, z_to_um = 0.1, 0.25
min_trace_length = 20
min_brightness = 0
max_brightness = np.inf
do_fuse_ = False
dont_use_high_power = True
use_20230415_high_power = False # if false these are not included, if true these are included
use_20230419_high_power = False

# True, True, False: 0.6-1.2mW + 2mW
# True, False, False: 0.6-1.2mW
# False, False, False: all: <= 4mW

# %%
# EEA1 data

def use_file_eea1(file, dont_use_high_power, 
                use_20230415_high_power, use_20230419_high_power):
    
    if 'ch560nmCamA' not in file:
        return False
    if 'EX_xyProj' in file or 'EX_yzProj' in file or 'EX_xzProj' in file:
        return False
    if '/Nalysis' in file or '/nalysis' in file:
        return False
    if '20230415' in file and 'Ex12' in file: # doesnt want to track
        return False
    if '20230419' in file and ('CS2' in file or 'CS3' in file):
        return False
    #print('EEA1',file)

    high_power_cs = ['CS6','CS7','CS8','CS9']
    high_power_exp = ['Ex02','Ex03','Ex23','Ex24','Ex29','Ex30','Ex36',
                        'Ex37','Ex38','Ex39','Ex46']
    if dont_use_high_power:
        if not use_20230415_high_power and any([exp in file for exp in high_power_cs]) and '20230415' in file:
            return False
        if not use_20230419_high_power and any([exp in file for exp in high_power_exp]) and '20230419' in file:
            return False
    
    return True


no_tracks = []
num_t_eea1_exp = []

# EEA1 data
no_tracks = []
num_t_eea1_exp = []

def parallel_loader_eea1(file):
    
    EEA1_df, seqOfEvents_list, keep_idx_len,\
        keep_idx_dim, keep_idx_bri,\
        original_idx = load_3D_data(
                        file.split('ProcessedTracks.mat')[0], 
                        '{}ProcessedTracks.mat', 
                        OUTPUT_NAME,
                        min_trace_length,
                        min_brightness,
                        max_brightness,
                        dont_use=dont_use_experiments)
    seqOfEvents_list_curated = []
    for i, idx in enumerate(original_idx):
        if idx not in keep_idx_len:
            continue
        if idx not in keep_idx_dim:
            continue
        if idx not in keep_idx_bri:
            continue
        seqOfEvents_list_curated.append(seqOfEvents_list[i])

    if len(EEA1_df)>=1:
        
        EEA1_tracks, EEA1_timepoints, EEA1_frames,\
        EEA1_track_ids, EEA1_amplitudes,\
        EEA1_amplitudes_S, EEA1_amplitudes_SM,\
        EEA1_amplitudes_sig, EEA1_amplitudes_bg,\
        EEA1_catIdx = curate_3D_data_to_tracks(EEA1_df, xy_to_um, z_to_um)
        EEA1_compound_idx = np.array(list(range(len(EEA1_tracks))))[EEA1_catIdx>4]
        num_t_eea1_exp.append(len(EEA1_tracks))
        handle_compound_tracks(EEA1_compound_idx, seqOfEvents_list_curated, EEA1_tracks, EEA1_timepoints, EEA1_frames,
                            EEA1_track_ids, EEA1_amplitudes, EEA1_amplitudes_S,
                            EEA1_amplitudes_SM, EEA1_amplitudes_sig, EEA1_amplitudes_bg,
                            min_trace_length, min_brightness)

        if do_fuse_:
            EEA1_tracks, EEA1_timepoints, EEA1_frames,\
            EEA1_track_ids, EEA1_amplitudes,\
            EEA1_amplitudes_S, EEA1_amplitudes_SM,\
            EEA1_amplitudes_sig, EEA1_amplitudes_bg, = fuse_tracks(
                                            EEA1_compound_idx, EEA1_tracks, 
                                            EEA1_timepoints, EEA1_frames,
                                            EEA1_track_ids, EEA1_amplitudes, 
                                            EEA1_amplitudes_S, EEA1_amplitudes_SM, 
                                            EEA1_amplitudes_sig, EEA1_amplitudes_bg,
                                            min_trace_length, min_brightness,
                                            blinking_forgiveness=1)
            
        files_to_save = [file.split('ProcessedTracks.mat')[0] for i in range(len(EEA1_tracks))] 
        exp_to_save = [file.split('ch560nmCamA')[0] for i in range(len(EEA1_tracks))]
        return files_to_save, exp_to_save,\
            EEA1_tracks, EEA1_timepoints, EEA1_frames,\
            EEA1_track_ids, EEA1_amplitudes,\
            EEA1_amplitudes_S, EEA1_amplitudes_SM,\
            EEA1_amplitudes_sig, EEA1_amplitudes_bg
    else:
        return [file.split('ch560nmCamA')[0]], [file.split('ch560nmCamA')[0]], [np.ones((1,3))*-1],\
                [np.ones((1,3))*-1], [np.ones((1,3))*-1], [np.ones((1,3))*-1],\
                    [np.ones((1,3))*-1], [np.ones((1,3))*-1], [np.ones((1,3))*-1]

t = time.time()
t1 = datetime.datetime.now()
print(t1)
results = Parallel(n_jobs=30)(
                delayed(parallel_loader_eea1)(file) for file in files_curated
                if use_file_eea1(file, dont_use_high_power, use_20230415_high_power, 
                            use_20230419_high_power))
print(datetime.datetime.now(), time.time()-t)
print('load')

EEA1_filenames_all = np.array(flatten_list([r[0] for r in results]))
EEA1_expname_all = np.array(flatten_list([r[1] for r in results]))
EEA1_tracks_all = np.array(flatten_list([r[2] for r in results]), dtype=object)
EEA1_timepoints_all = np.array(flatten_list([r[3] for r in results]), dtype=object)
EEA1_frames_all = np.array(flatten_list([r[4] for r in results]), dtype=object)
EEA1_track_ids_all = np.array(flatten_list([r[5] for r in results]), dtype=object)
EEA1_amplitudes_all = np.array(flatten_list([r[6] for r in results]), dtype=object)
EEA1_amplitudes_sig_all = np.array(flatten_list([r[7] for r in results]), dtype=object)
EEA1_amplitudes_bg_all = np.array(flatten_list([r[8] for r in results]), dtype=object)

len(EEA1_filenames_all), len(EEA1_tracks_all)

# %%

print(np.unique([e.split('Ex')[0] for e in EEA1_expname_all]))
np.unique(EEA1_expname_all).shape

# %%
# NPC1 data


def use_file_npc1(file, dont_use_high_power, 
                use_20230415_high_power, use_20230419_high_power):
    
    if 'ch642nmCamB' not in file:
        return False
    if 'EX_xyProj' in file or 'EX_yzProj' in file or 'EX_xzProj' in file:
        return False
    if '/Nalysis' in file or '/nalysis' in file:
        return False
    if '20230415' in file and 'Ex12' in file: # doesnt want to track
        return False
    if '20230419' in file and ('CS2' in file or 'CS3' in file):
        return False

    high_power_cs = ['CS6','CS7','CS8','CS9']
    high_power_exp = ['Ex02','Ex03','Ex23','Ex24','Ex29','Ex30','Ex36',
                        'Ex37','Ex38','Ex39','Ex46']
    if dont_use_high_power:
        if not use_20230415_high_power and any([exp in file for exp in high_power_cs]) and '20230415' in file:
            return False
        if not use_20230419_high_power and any([exp in file for exp in high_power_exp]) and '20230419' in file:
            return False
    
    return True

num_t_npc1_exp = []

def parallel_loader_npc1(file):

    NPC1_df, seqOfEvents_list, keep_idx_len,\
        keep_idx_dim, keep_idx_bri,\
        original_idx = load_3D_data(
                        file.split('ProcessedTracks.mat')[0], 
                        '{}ProcessedTracks.mat', 
                        OUTPUT_NAME,
                        min_trace_length,
                        min_brightness,
                        max_brightness,
                        dont_use=dont_use_experiments)
    seqOfEvents_list_curated = []
    for i, idx in enumerate(original_idx):
        if idx not in keep_idx_len:
            continue
        if idx not in keep_idx_dim:
            continue
        if idx not in keep_idx_bri:
            continue
        seqOfEvents_list_curated.append(seqOfEvents_list[i])

    NPC1_tracks, NPC1_timepoints, NPC1_frames,\
    NPC1_track_ids, NPC1_amplitudes,\
    NPC1_amplitudes_S, NPC1_amplitudes_SM,\
    NPC1_amplitudes_sig, NPC1_amplitudes_bg,\
    NPC1_catIdx = curate_3D_data_to_tracks(NPC1_df, xy_to_um, z_to_um)
    NPC1_compound_idx = np.array(list(range(len(NPC1_tracks))))[NPC1_catIdx>4]
    num_t_npc1_exp.append(len(NPC1_tracks))
    handle_compound_tracks(NPC1_compound_idx, seqOfEvents_list_curated, NPC1_tracks, NPC1_timepoints, NPC1_frames,
                            NPC1_track_ids, NPC1_amplitudes, NPC1_amplitudes_S,
                                NPC1_amplitudes_SM, NPC1_amplitudes_sig, NPC1_amplitudes_bg,
                                min_trace_length, min_brightness)

    
    if do_fuse_:
        NPC1_tracks, NPC1_timepoints, NPC1_frames,\
        NPC1_track_ids, NPC1_amplitudes,\
        NPC1_amplitudes_S, NPC1_amplitudes_SM,\
        NPC1_amplitudes_sig, NPC1_amplitudes_bg, = fuse_tracks(
                                        NPC1_compound_idx, NPC1_tracks, 
                                        NPC1_timepoints, NPC1_frames,
                                        NPC1_track_ids, NPC1_amplitudes, 
                                        NPC1_amplitudes_S, NPC1_amplitudes_SM, 
                                        NPC1_amplitudes_sig, NPC1_amplitudes_bg,
                                        min_trace_length, min_brightness,
                                        blinking_forgiveness=1)
    
    files_to_save = [file.split('ProcessedTracks.mat')[0] for i in range(len(NPC1_tracks))] 
    exp_to_save = [file.split('ch642nmCamB')[0] for i in range(len(NPC1_tracks))]
    return files_to_save, exp_to_save,\
            NPC1_tracks, NPC1_timepoints, NPC1_frames,\
            NPC1_track_ids, NPC1_amplitudes,\
            NPC1_amplitudes_S, NPC1_amplitudes_SM,\
            NPC1_amplitudes_sig, NPC1_amplitudes_bg
    
t = time.time()
t1 = datetime.datetime.now()
print(t1)
results = Parallel(n_jobs=30)(
                delayed(parallel_loader_npc1)(file) for file in files_curated
                if use_file_npc1(file, dont_use_high_power, use_20230415_high_power, 
                            use_20230419_high_power))
print(datetime.datetime.now(), time.time()-t)
print('load')

NPC1_filenames_all = np.array(flatten_list([r[0] for r in results]))
NPC1_expname_all = np.array(flatten_list([r[1] for r in results]))
NPC1_tracks_all = np.array(flatten_list([r[2] for r in results]), dtype=object)
NPC1_timepoints_all = np.array(flatten_list([r[3] for r in results]), dtype=object)
NPC1_frames_all = np.array(flatten_list([r[4] for r in results]), dtype=object)
NPC1_track_ids_all = np.array(flatten_list([r[5] for r in results]), dtype=object)
NPC1_amplitudes_all = np.array(flatten_list([r[6] for r in results]), dtype=object)
NPC1_amplitudes_sig_all = np.array(flatten_list([r[7] for r in results]), dtype=object)
NPC1_amplitudes_bg_all = np.array(flatten_list([r[8] for r in results]), dtype=object)

len(NPC1_filenames_all), len(NPC1_tracks_all)

# %%
# Rota data
def filter_tracks(tracks_filter, Rota_tracks,
                    Rota_timepoints, Rota_frames,
                    Rota_track_ids, Rota_amplitudes,
                    Rota_amplitudes_S, Rota_amplitudes_SM,
                    Rota_amplitudes_sig, Rota_amplitudes_bg):
    Rota_tracks = [Rota_tracks[i] for i in tracks_filter]
    Rota_timepoints = [Rota_timepoints[i] for i in tracks_filter]
    Rota_frames = [Rota_frames[i] for i in tracks_filter]
    Rota_track_ids = [Rota_track_ids[i] for i in tracks_filter]
    Rota_amplitudes = [Rota_amplitudes[i] for i in tracks_filter]
    Rota_amplitudes_S = [Rota_amplitudes_S[i] for i in tracks_filter]
    Rota_amplitudes_SM = [Rota_amplitudes_SM[i] for i in tracks_filter]
    Rota_amplitudes_sig = [Rota_amplitudes_sig[i] for i in tracks_filter]
    Rota_amplitudes_bg = [Rota_amplitudes_bg[i] for i in tracks_filter]
    return Rota_tracks, Rota_timepoints, Rota_frames,\
            Rota_track_ids, Rota_amplitudes,\
            Rota_amplitudes_sig, Rota_amplitudes_bg

multiplier_nominator = (1.2*10)
num_t_rota_exp = []
from joblib import Parallel, delayed
import time

def use_file_rota(file, dont_use_high_power, 
                use_20230415_high_power, use_20230419_high_power):
    
    if 'ch488nmCamB' not in file:
        return False
    if '/Nalysis' in file or '/nalysis' in file:
        return False
    if 'Analysis_wrong' in file:
        return False
    if 'EX_xyProj' in file or 'EX_yzProj' in file or 'EX_xzProj' in file:
        return False
    if '20230413' in file and 'ch488nmCamB_edge' not in file and 'CS1' not in file:
        return False
    if '20230415' in file and 'Ex12' in file: # doesnt want to track
        return False
    if '20230419' in file and ('CS2' in file or 'CS3' in file):
        return False
    if '20230413' in file and 'Ex05' in file:
        print('E and N not tracked in ex05')
        return False
    
    # high power experiments:
    high_power_cs = ['CS6','CS7','CS8','CS9']
    high_power_exp = ['Ex02','Ex03','Ex23','Ex24','Ex29','Ex30','Ex36',
                        'Ex37','Ex38','Ex39','Ex46']
    if dont_use_high_power:
        if not use_20230415_high_power and any([exp in file for exp in high_power_cs]) and '20230415' in file:
            return False
        if not use_20230419_high_power and any([exp in file for exp in high_power_exp]) and '20230419' in file:
            return False
    
    return True


def parallel_loader_rota(file, multiplier_nominator):
    if 'CS1' in file and '20230413' in file:
        ampli_multiplier = multiplier_nominator/(0.8*10) 
    elif 'CS1' not in file and '20230413' in file:
        ampli_multiplier = multiplier_nominator/(1.2*10) 

    elif 'CS1' in file and '20230415' in file:
        ampli_multiplier = multiplier_nominator/(1.2*10) 
    elif 'CS2' in file and '20230415' in file:
        ampli_multiplier = multiplier_nominator/(1.2*10)  

    high_power_cs = ['CS6','CS7','CS8','CS9']
    if any([exp in file for exp in high_power_cs]) and '20230415' in file:
        ampli_multiplier = multiplier_nominator/(2*20) 
    
    low_power_exp = ['Ex14','Ex15','Ex16','Ex17','Ex18','Ex19',
                        'Ex20','Ex21','Ex22','Ex25','Ex26','Ex27','Ex28','Ex31',
                        'Ex32','Ex33','Ex34','Ex35','Ex40','Ex41','Ex42','Ex43',
                        'Ex44','Ex45']
    high_power_exp = ['Ex02','Ex03','Ex23','Ex24','Ex29','Ex30','Ex36',
                        'Ex37','Ex38','Ex39','Ex46']
    
    if any([exp in file for exp in low_power_exp]) and '20230419' in file:
        ampli_multiplier = multiplier_nominator/(0.6*10) 
    
    elif any([exp in file for exp in high_power_exp]) and '20230419' in file:
        ampli_multiplier = multiplier_nominator/(4*10) 

    Rota_df, seqOfEvents_list, keep_idx_len,\
        keep_idx_dim, keep_idx_bri,\
        original_idx = load_3D_data(
                        file.split('ProcessedTracks.mat')[0], 
                        '{}ProcessedTracks.mat', 
                        OUTPUT_NAME,
                        min_trace_length,
                        min_brightness,
                        max_brightness,
                        dont_use=dont_use_experiments)

    seqOfEvents_list_curated = []
    for i, idx in enumerate(original_idx):
        if idx not in keep_idx_len:
            continue
        if idx not in keep_idx_dim:
            continue
        if idx not in keep_idx_bri:
            continue
        seqOfEvents_list_curated.append(seqOfEvents_list[i])
    
    Rota_tracks, Rota_timepoints, Rota_frames,\
    Rota_track_ids, Rota_amplitudes,\
    Rota_amplitudes_S, Rota_amplitudes_SM,\
    Rota_amplitudes_sig, Rota_amplitudes_bg,\
    Rota_catIdx = curate_3D_data_to_tracks(Rota_df, xy_to_um, z_to_um)
    Rota_compound_idx = np.array(list(range(len(Rota_tracks))))[Rota_catIdx>4]
    handle_compound_tracks(Rota_compound_idx, seqOfEvents_list_curated, Rota_tracks, Rota_timepoints, Rota_frames,
                            Rota_track_ids, Rota_amplitudes, Rota_amplitudes_S,
                                Rota_amplitudes_SM, Rota_amplitudes_sig, Rota_amplitudes_bg,
                                min_trace_length, min_brightness)
    num_t_rota_exp.append(len(Rota_tracks))
    if do_fuse_:
        Rota_tracks, Rota_timepoints, Rota_frames,\
        Rota_track_ids, Rota_amplitudes,\
        Rota_amplitudes_S, Rota_amplitudes_SM,\
        Rota_amplitudes_sig, Rota_amplitudes_bg, = fuse_tracks(
                                        Rota_compound_idx, Rota_tracks, 
                                        Rota_timepoints, Rota_frames,
                                        Rota_track_ids, Rota_amplitudes, 
                                        Rota_amplitudes_S, Rota_amplitudes_SM, 
                                        Rota_amplitudes_sig, Rota_amplitudes_bg,
                                        min_trace_length, min_brightness,
                                        blinking_forgiveness=1)
    

    Rota_amplitudes = [r*ampli_multiplier for r in Rota_amplitudes]
    files_to_save = [file.split('ProcessedTracks.mat')[0] for i in range(len(Rota_tracks))] 
    exp_to_save = [file.split('ch488nmCamB')[0] for i in range(len(Rota_tracks))]
    return files_to_save, exp_to_save,\
            Rota_tracks ,Rota_timepoints, Rota_frames,\
                Rota_track_ids, Rota_amplitudes,\
            Rota_amplitudes_sig, Rota_amplitudes_bg,\

t = time.time()
t1 = datetime.datetime.now()
print(t1)
results = Parallel(n_jobs=30)(
                delayed(parallel_loader_rota)(file, multiplier_nominator) for file in files_curated
                if use_file_rota(file, dont_use_high_power, use_20230415_high_power, 
                            use_20230419_high_power))
print(datetime.datetime.now(), time.time()-t)
print('load')

Rota_filenames_all = np.array(flatten_list([r[0] for r in results]))
Rota_expname_all = np.array(flatten_list([r[1] for r in results]))
Rota_tracks_all = np.array(flatten_list([r[2] for r in results]), dtype=object)
Rota_timepoints_all = np.array(flatten_list([r[3] for r in results]), dtype=object)
Rota_frames_all = np.array(flatten_list([r[4] for r in results]), dtype=object)
Rota_track_ids_all = np.array(flatten_list([r[5] for r in results]), dtype=object)
Rota_amplitudes_all = np.array(flatten_list([r[6] for r in results]), dtype=object)
Rota_amplitudes_sig_all = np.array(flatten_list([r[7] for r in results]), dtype=object)
Rota_amplitudes_bg_all = np.array(flatten_list([r[8] for r in results]), dtype=object)

# %%
s_npc1 = "".join(s for s in np.unique(NPC1_expname_all))
s_eea1 = "".join(s for s in np.unique(EEA1_expname_all))
s_rota = "".join(s for s in np.unique(Rota_expname_all))
assert s_npc1 == s_eea1 == s_rota

# %%
print(np.unique(NPC1_expname_all).shape)
print(np.unique(EEA1_expname_all).shape)
print(np.unique(Rota_expname_all).shape)

set(NPC1_expname_all)-set(EEA1_expname_all)
set(Rota_expname_all)-set(EEA1_expname_all)

# %%
EEA1val = 0
NPC1val = 1
Rotaval = 2
tracks = np.array(
    list(EEA1_tracks_all) + list(NPC1_tracks_all) + list(Rota_tracks_all), 
        dtype=object)
track_labels = np.concatenate([np.ones(len(EEA1_tracks_all))*EEA1val,
                               np.ones(len(NPC1_tracks_all))*NPC1val,
                               np.ones(len(Rota_tracks_all))*Rotaval])
experiments = np.unique(list(Rota_expname_all))

print(len(tracks), len(track_labels), len(experiments))
print(np.unique(track_labels, return_counts=True))
print(len(EEA1_tracks_all), len(NPC1_tracks_all), len(Rota_tracks_all))

# %%

# %%

"""
distthreshold = .75
min_coloc_len = 5
blinkinglength_threshold = 5

postprocess_min_coloc_len = 5
postprocess_min_norm2 = .75
corr_threshold = 0.8
"""

distthreshold = .75
min_coloc_len = 5
blinkinglength_threshold = 5

postprocess_min_coloc_len = 5
postprocess_min_norm2 = .75
corr_threshold = 0.8

total_Rota_tracks = []
total_endo_tracks = []
number_coloc_tracks = []
tracks_coloc_info_sum = 0
timepoints_annotated_by_colocsum = 0

tracks_seen_already = 0
for exp_i, exp in enumerate(np.sort(experiments)):
    print('Experiment:', exp)
    # if 'Ex34' not in exp or '20230419' not in exp:
    #     continue

    # Load POI tracks (Rota)
    POI_tracks = Rota_tracks_all[Rota_expname_all==exp]
    POI_frames = Rota_frames_all[Rota_expname_all==exp]
    POI_track_ids = Rota_track_ids_all[Rota_expname_all==exp]
    total_tracks = len(POI_tracks)
    print('ROTA tracks:', len(POI_tracks))
    # Load coloc tracks (NPC1, EEA1)x
    coloc_types = ['EEA1', 'NPC1'] # ['EEA1'] # ['EEA1', 'NPC1']
    for coloc_type in coloc_types:
        total_Rota_tracks.append(total_tracks)
        if coloc_type == 'EEA1':
            coloc_tracks = EEA1_tracks_all[EEA1_expname_all==exp]
            coloc_frames = EEA1_frames_all[EEA1_expname_all==exp]
            coloc_track_ids = EEA1_track_ids_all[EEA1_expname_all==exp]
        elif coloc_type == 'NPC1':
            coloc_tracks = NPC1_tracks_all[NPC1_expname_all==exp]
            coloc_frames = NPC1_frames_all[NPC1_expname_all==exp]
            coloc_track_ids = NPC1_track_ids_all[NPC1_expname_all==exp]
        total_endo_tracks.append(len(coloc_tracks))
        print('Coloc type:', coloc_type, len(coloc_tracks))

        if False:#os.path.exists(datapath+'/potential_coloc'+str(bio_repli)+'.pkl'):
            potential_coloc = pickle.load(open(datapath+'/potential_coloc'+str(bio_repli)+'.pkl', 'rb'))
            potential_coloc_info = pickle.load(open(datapath+'/potential_coloc_info'+str(bio_repli)+'.pkl', 'rb'))

        else:
            promising_idx, changepoint_list, seglens_list,\
            poi_idx_list, col_idx_list,\
            accepted_blinking_list, rejected_blinking_list = generate_coloc_output(
                                                                POI_tracks, POI_frames, 
                                                                coloc_tracks, coloc_frames, 
                                                                threshold=distthreshold, 
                                                                min_coloc_len=min_coloc_len,
                                                                blinkinglength_threshold=blinkinglength_threshold)                               

        
            print('Number of promising idx:', len(promising_idx))
            potential_coloc = []
            potential_coloc_info = []
            for i,(poi_idx, col_idx) in enumerate(tqdm(zip(poi_idx_list, col_idx_list))):
                changepoint = changepoint_list[i]
                seglens = seglens_list[i]

                accepted_blinking = accepted_blinking_list[i]
                rejected_blinking = rejected_blinking_list[i]
                
                for cp, segl in zip(changepoint, seglens):
                    start, end = cp, cp+segl
                    # get frames
                    frames_for_poi = np.array(POI_frames[poi_idx])[start:end]
                    frames_for_col = coloc_frames[col_idx]

                    # ensure matching frames (could be one or the other isnt there for a frame or so)
                    idx_in_COL = np.in1d(frames_for_col, frames_for_poi)
                    idx_in_POI = np.in1d(frames_for_poi, frames_for_col)
                    
                    assert np.mean(frames_for_poi[idx_in_POI]==frames_for_col[idx_in_COL])==1
                    
                    POI = POI_tracks[poi_idx][start:end][idx_in_POI] # particle of interest
                    COL = coloc_tracks[col_idx][idx_in_COL] # coloc particle
                    potential_coloc.append((POI, COL))
                    potential_coloc_info.append((poi_idx + tracks_seen_already,col_idx,cp,segl))

            tracks_coloc = np.array(potential_coloc, dtype=object)
            tracks_coloc_info = np.array(potential_coloc_info)
            if len(tracks_coloc)>0:

                """ FILTER PREDICTIONS BY length """
                lengths = np.array([len(x) for x in tracks_coloc[:,0]])
                filtering = np.hstack(np.where(lengths>postprocess_min_coloc_len))
                tracks_coloc = tracks_coloc[filtering]
                tracks_coloc_info = np.array(tracks_coloc_info)[filtering]
                og_potential_over_minlen = len(tracks_coloc)

                if len(tracks_coloc)>0:
                    """ FILTER PREDICTIONS BY 2. NORM DISTANCE """
                    ###### Prep found potential coloc
                    norm2_dist = np.array([np.median(
                                            np.sqrt(
                                            (np.sum((POI-COL)**2, axis=1).astype(float)))) 
                                        for POI,COL in tracks_coloc])
                    
                    norm2_filter = np.hstack(np.where(norm2_dist<postprocess_min_norm2))
                    tracks_coloc = tracks_coloc[norm2_filter]
                    tracks_coloc_info = np.array(tracks_coloc_info)[norm2_filter]

                if len(tracks_coloc)>0:
                    """ FILTER PREDICTIONS BY PEARSON CORRELATION """
                    from scipy import stats
                    xcorrelation = np.array([stats.pearsonr(POI[:,0], COL[:,0])[0] for POI,COL in tracks_coloc])
                    ycorrelation = np.array([stats.pearsonr(POI[:,1], COL[:,1])[0] for POI,COL in tracks_coloc])
                    zcorrelation = np.array([stats.pearsonr(POI[:,2], COL[:,2])[0] for POI,COL in tracks_coloc])
                    corr_filter = (xcorrelation>corr_threshold)*(ycorrelation>corr_threshold)*(zcorrelation>corr_threshold)

                    tracks_coloc = tracks_coloc[corr_filter]
                    tracks_coloc_info = np.array(tracks_coloc_info)[corr_filter]

                tracks_coloc_info_sum += len(tracks_coloc_info)
                pickle.dump(tracks_coloc, open(savepath+'/'+exp.replace('/','_')+'v1ROTAcoloc_'+coloc_type+'.pkl', 'wb'))
                pickle.dump(tracks_coloc_info, open(savepath+'/'+exp.replace('/','_')+'v1ROTAcoloc_info_'+coloc_type+'.pkl', 'wb'))

            number_coloc_tracks.append(len(tracks_coloc))
            """ SAVE ANNOTATED TIMEPOINTS AND FULL COLOC PARTNER TRACES """
            POItracks_full = np.array(POI_tracks, dtype=object)
            COLOCtracks_full = np.array(coloc_tracks, dtype=object)
            timepoints_annotated_by_coloc = []
            coloc_tracks_full = []
            for i in range(len(POItracks_full)):
                if len(tracks_coloc)>0:
                    if i + tracks_seen_already in np.array(tracks_coloc_info[:,0]):
                        for dup in range(np.sum(np.array(tracks_coloc_info[:,0])==i+tracks_seen_already)):
                            idx = list(tracks_coloc_info[:,0]).index(i + tracks_seen_already)
                            nc = tracks_coloc_info[idx,1] #- tracks_seen_already
                            cp = tracks_coloc_info[idx,2]
                            segl = tracks_coloc_info[idx,3]
                            start, end = cp, cp+segl
                            # get frames
                            frames_for_coloc = np.array(POI_frames[i])[start:end]
                            # ensure matching frames (could be one or the other isnt there for a frame or so)
                            idx_in_COL = np.in1d(coloc_frames[nc], frames_for_coloc)
                            idx_in_POI = np.in1d(frames_for_coloc, coloc_frames[nc])

                            assert np.mean(frames_for_coloc[idx_in_POI]==coloc_frames[nc][idx_in_COL])

                            poi_annot = np.in1d(POI_frames[i], frames_for_coloc).astype(int)     
                            col_annot = idx_in_COL.astype(int)

                            timepoints_annotated_by_coloc.append((poi_annot,col_annot))
                            coloc_tracks_full.append((POItracks_full[i], COLOCtracks_full[nc]))
                            timepoints_annotated_by_colocsum += 1
                    else:
                        timepoints_annotated_by_coloc.append((np.zeros(len(tracks[i])), None))
                else:
                    timepoints_annotated_by_coloc.append((np.zeros(len(tracks[i])), None))

            
            pickle.dump(timepoints_annotated_by_coloc, open(savepath+'/'+exp.replace('/','_')+'v1ROTAtimepoints_annotated_by_coloc'+coloc_type+'.pkl', 'wb'))
            pickle.dump(coloc_tracks_full, open(savepath+'/'+exp.replace('/','_')+'v1ROTAcoloc_tracks_full'+coloc_type+'.pkl', 'wb'))
    tracks_seen_already += len(POI_tracks)
