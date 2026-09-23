# %%
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull
import pickle 
import numpy as np
from global_config import globals
import matplotlib.pyplot as plt
from sklearn.decomposition import *
import tifffile as tiff
import glob
from scipy.ndimage import gaussian_filter, label
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import numpy as np
from scipy.spatial import ConvexHull
from pygel3d import hmesh
import datetime
#import ipyvolume as ipv
from scipy.ndimage import label, generate_binary_structure
from skimage.segmentation import find_boundaries
from glob import glob
import os
from joblib import Parallel, delayed
import time
import random
import pickle 
import os 
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from global_config import globals
from glob import glob
import alphashape
from scipy.optimize import minimize
import trimesh
import time
from joblib import Parallel, delayed

# change path of import with sys
import sys
sys.path.append('../')
from deepspt_src import *
from deepspt_src.coloc_helpers import *


def use_file(file, ch_cam_name='ch488nmCamB'):
    
    if ch_cam_name not in file:
        return False
    if 'EX_xyProj' in file or 'EX_yzProj' in file or 'EX_xzProj' in file:
        return False
    
    return True


def parallel_loader(file, multiplier_nominator=1.2*10, ch_cam_name='ch488nmCamB'):
    _df, seqOfEvents_list, keep_idx_len,\
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

    if len(_df)>=1:
        
        tracks, timepoints, frames,\
        track_ids, amplitudes,\
        amplitudes_S, amplitudes_SM,\
        amplitudes_sig, amplitudes_bg,\
        catIdx = curate_3D_data_to_tracks(_df, xy_to_um, z_to_um)
        compound_idx = np.array(list(range(len(tracks))))[catIdx>4]
        handle_compound_tracks(compound_idx, seqOfEvents_list_curated, 
                                tracks, timepoints, frames,
                            track_ids, amplitudes, amplitudes_S,
                            amplitudes_SM, amplitudes_sig, amplitudes_bg,
                            min_trace_length, min_brightness)

        if do_fuse_:
            tracks, timepoints, frames,\
            track_ids, amplitudes,\
            amplitudes_S, amplitudes_SM,\
            amplitudes_sig, amplitudes_bg, = fuse_tracks(
                                            compound_idx, tracks, 
                                            timepoints, frames,
                                            track_ids, amplitudes, 
                                            amplitudes_S, amplitudes_SM, 
                                            amplitudes_sig, amplitudes_bg,
                                            min_trace_length, min_brightness,
                                            blinking_forgiveness=1)
            
        files_to_save = [file.split('ProcessedTracks.mat')[0] for i in range(len(tracks))] 
        exp_to_save = [file.split(ch_cam_name)[0] for i in range(len(tracks))]
        return files_to_save, exp_to_save,\
            tracks, timepoints, frames,\
            track_ids, amplitudes,\
            amplitudes_S, amplitudes_SM,\
            amplitudes_sig, amplitudes_bg
    else:
        return [file.split(ch_cam_name)[0]], [file.split(ch_cam_name)[0]], [np.ones((1,3))*-1],\
                [np.ones((1,3))*-1], [np.ones((1,3))*-1], [np.ones((1,3))*-1],\
                    [np.ones((1,3))*-1], [np.ones((1,3))*-1], [np.ones((1,3))*-1]


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
main_dir = '../_Data/LLSM_data/Duallabeled_Rotavirus/20220615_p5_p55_sCMOS/CS16_SVGA_488soluble_rcTLPAtto565Atto642/'
SEARCH_PATTERN = '{}/**/ProcessedTracks.mat'
OUTPUT_NAME = 'rotavirus'
_input = SEARCH_PATTERN.format(main_dir)
files = sorted(glob(_input, recursive=True))
dont_use_experiments = []

files_curated = []
for file in files:
    if 'ch488nmCamB' in file:
        continue
    if 'ch488nmCamA' in file:
        continue
    if 'Analysis_Mari' in file:
        continue
    if '3Channel_Detection' in file or '3channel detection' in file:
        continue
    if 'Analysis642_3ch' in file or 'Analysis642 3ch' in file or 'Analysis_3ch' in file:
        continue
    if 'EX_xyProj' in file or 'EX_yzProj' in file or 'EX_xzProj' in file:
        continue
    files_curated.append(file)
xy_to_um, z_to_um = 0.1, 0.25
min_trace_length = 20
min_brightness = 0
max_brightness = np.inf
do_fuse_ = False
files_curated


# %%
print(np.unique([e.split('Ex')[0] for e in files_curated]))
print(np.unique([e.split('Ex')[0] for e in files_curated]))

# %%

t = time.time()
t1 = datetime.datetime.now()
print(t1)

multiplier_nominator = (1.2*10)
ch_cam_name = 'ch560nmCamB'
results = Parallel(n_jobs=30)(
                delayed(parallel_loader)(file, ch_cam_name=ch_cam_name, 
                                         multiplier_nominator=multiplier_nominator) 
                                         for file in files_curated
                if use_file(file, ch_cam_name=ch_cam_name))
print(datetime.datetime.now(), time.time()-t)
print('load')

VP7_filenames_all = np.array(flatten_list([r[0] for r in results]))
VP7_expname_all = np.array(flatten_list([r[1] for r in results]))
VP7_tracks_all = np.array(flatten_list([r[2] for r in results]), dtype=object)
VP7_timepoints_all = np.array(flatten_list([r[3] for r in results]), dtype=object)
VP7_frames_all = np.array(flatten_list([r[4] for r in results]), dtype=object)
VP7_track_ids_all = np.array(flatten_list([r[5] for r in results]), dtype=object)
VP7_amplitudes_all = np.array(flatten_list([r[6] for r in results]), dtype=object)
VP7_amplitudes_sig_all = np.array(flatten_list([r[7] for r in results]), dtype=object)
VP7_amplitudes_bg_all = np.array(flatten_list([r[8] for r in results]), dtype=object)


# %%
t = time.time()
t1 = datetime.datetime.now()
print(t1)

multiplier_nominator = (1.2*10)
ch_cam_name = 'ch642nmCamA'
results = Parallel(n_jobs=30)(
                delayed(parallel_loader)(file, ch_cam_name=ch_cam_name, 
                                         multiplier_nominator=multiplier_nominator) 
                                         for file in files_curated
                if use_file(file, ch_cam_name=ch_cam_name))
print(datetime.datetime.now(), time.time()-t)
print('load')

DLP_filenames_all = np.array(flatten_list([r[0] for r in results]))
DLP_expname_all = np.array(flatten_list([r[1] for r in results]))
DLP_tracks_all = np.array(flatten_list([r[2] for r in results]), dtype=object)
DLP_timepoints_all = np.array(flatten_list([r[3] for r in results]), dtype=object)
DLP_frames_all = np.array(flatten_list([r[4] for r in results]), dtype=object)
DLP_track_ids_all = np.array(flatten_list([r[5] for r in results]), dtype=object)
DLP_amplitudes_all = np.array(flatten_list([r[6] for r in results]), dtype=object)
DLP_amplitudes_sig_all = np.array(flatten_list([r[7] for r in results]), dtype=object)
DLP_amplitudes_bg_all = np.array(flatten_list([r[8] for r in results]), dtype=object)

# %%
print(len(VP7_tracks_all), len(DLP_tracks_all))
len(np.unique(DLP_expname_all)), len(np.unique(VP7_expname_all))

# %%

"""
initial thresholds to set to derive xyz chromatic offset
distthreshold = .75
min_coloc_len = 5
blinkinglength_threshold = 3
postprocess_min_coloc_len = 20
postprocess_min_norm2 = 0.5
corr_threshold = 0.9
array([0.00895479, 0.04941534, 0.37934114])

distthreshold = .4
min_coloc_len = 3
blinkinglength_threshold = 2
postprocess_min_coloc_len = 20
postprocess_min_norm2 = 0.6
corr_threshold = 0.9
min_frames_from_edge = 2
"""

# offset chromatic aberrations
# VP7_tracks_all_curated = VP7_tracks_all.copy()
VP7_tracks_all_curated = np.array(
    [v + np.array([0.009, 0.049, 0.38]) for v in VP7_tracks_all], dtype=object)

distthreshold = .4
min_coloc_len = 3
blinkinglength_threshold = 2
postprocess_min_coloc_len = 20
postprocess_min_norm2 = 0.6
corr_threshold = 0.9

total_DLP_tracks = []
total_VP7_tracks = []
number_coloc_tracks = []
tracks_coloc_info_sum = 0
timepoints_annotated_by_colocsum = 0

savepath = '../deepspt_results/analytics/duallabelled_results'

experiments = np.unique(DLP_expname_all)
tracks_seen_already = 0
coloctracks_seen_already = 0
timepoints_annotated_by_coloc_all = []
tracks_coloc_info_all = []
idx_dup_tracks = []
length_dup_all = []
xcorr_dup_all = []
ycorr_dup_all = []
zcorr_dup_all = []
norm2_dup_all = []
has_dupli_all = []

timepoints_annotated_by_coloc = []
COLtimepoints_annotated_by_coloc = []
original_trackidx = []
original_coloctrackidx = []
length_coloc = []
norm2_dist_coloc = []
signed_dist_coloc = []
xcorrelation_coloc = []
zcorrelation_coloc = []
ycorrelation_coloc = []
for exp_i, exp in enumerate(np.sort(experiments)):
    print('Experiment:', exp)

    # Load POI tracks (Rota)
    POI_tracks = DLP_tracks_all[DLP_expname_all==exp]
    POI_frames = DLP_frames_all[DLP_expname_all==exp]
    POI_track_ids = DLP_track_ids_all[DLP_expname_all==exp]
    total_tracks = len(POI_tracks)
    total_VP7_tracks.append(total_tracks)

    # Load coloc tracks
    coloc_tracks = VP7_tracks_all_curated[VP7_expname_all==exp]
    coloc_frames = VP7_frames_all[VP7_expname_all==exp]
    coloc_track_ids = VP7_track_ids_all[VP7_expname_all==exp]

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
        original_trackidx_interrim = []
        for i in range(len(POI_tracks)):
            if i in poi_idx_list:
                for ix in np.where(np.array(poi_idx_list)==i)[0]:
                    poi_idx, col_idx = i, col_idx_list[ix]
                    changepoint = changepoint_list[ix]
                    seglens = seglens_list[ix]
                    accepted_blinking = accepted_blinking_list[ix]
                    rejected_blinking = rejected_blinking_list[ix]
                    
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

                        """ FILTER PREDICTIONS BY LENGTH """
                        if len(POI)<postprocess_min_coloc_len:
                            COLtimepoints_annotated_by_coloc.append(np.zeros(len(POI_tracks[i])))
                            timepoints_annotated_by_coloc.append(np.zeros(len(POI_tracks[i])))
                            original_trackidx.append(i+tracks_seen_already)
                            length_coloc.append(np.nan)
                            norm2_dist_coloc.append(np.nan)
                            signed_dist_coloc.append(np.nan)
                            xcorrelation_coloc.append(np.nan)
                            zcorrelation_coloc.append(np.nan)
                            ycorrelation_coloc.append(np.nan)
                            original_coloctrackidx.append(np.nan)
                            continue

                        """ FILTER PREDICTIONS BY 2. NORM DISTANCE """
                        norm2_dist = np.median(
                                        np.sqrt(
                                            (np.sum((POI-COL)**2, axis=1).astype(float)))) 
                        signed_dist = np.median(POI-COL, axis=0).astype(float)

                        if norm2_dist>postprocess_min_norm2:
                            COLtimepoints_annotated_by_coloc.append(np.zeros(len(POI_tracks[i])))
                            timepoints_annotated_by_coloc.append(np.zeros(len(POI_tracks[i])))
                            original_trackidx.append(i+tracks_seen_already)
                            length_coloc.append(np.nan)
                            norm2_dist_coloc.append(np.nan)
                            signed_dist_coloc.append(np.nan)
                            xcorrelation_coloc.append(np.nan)
                            zcorrelation_coloc.append(np.nan)
                            ycorrelation_coloc.append(np.nan)
                            original_coloctrackidx.append(np.nan)
                            continue
            
                        """ FILTER PREDICTIONS BY PEARSON CORRELATION """
                        from scipy import stats
                        xcorrelation = stats.pearsonr(POI[:,0], COL[:,0])[0]
                        if xcorrelation<corr_threshold:
                            COLtimepoints_annotated_by_coloc.append(np.zeros(len(POI_tracks[i])))
                            timepoints_annotated_by_coloc.append(np.zeros(len(POI_tracks[i])))
                            original_trackidx.append(i+tracks_seen_already)
                            length_coloc.append(np.nan)
                            norm2_dist_coloc.append(np.nan)
                            signed_dist_coloc.append(np.nan)
                            xcorrelation_coloc.append(np.nan)
                            zcorrelation_coloc.append(np.nan)
                            ycorrelation_coloc.append(np.nan)
                            original_coloctrackidx.append(np.nan)
                            continue

                        ycorrelation = stats.pearsonr(POI[:,1], COL[:,1])[0]
                        if ycorrelation<corr_threshold:
                            COLtimepoints_annotated_by_coloc.append(np.zeros(len(POI_tracks[i])))
                            timepoints_annotated_by_coloc.append(np.zeros(len(POI_tracks[i])))
                            original_trackidx.append(i+tracks_seen_already)
                            length_coloc.append(np.nan)
                            norm2_dist_coloc.append(np.nan)
                            signed_dist_coloc.append(np.nan)
                            xcorrelation_coloc.append(np.nan)
                            zcorrelation_coloc.append(np.nan)
                            ycorrelation_coloc.append(np.nan)
                            original_coloctrackidx.append(np.nan)
                            continue

                        zcorrelation = stats.pearsonr(POI[:,2], COL[:,2])[0]
                        if zcorrelation<corr_threshold:
                            COLtimepoints_annotated_by_coloc.append(np.zeros(len(POI_tracks[i])))
                            timepoints_annotated_by_coloc.append(np.zeros(len(POI_tracks[i])))
                            original_trackidx.append(i+tracks_seen_already)
                            length_coloc.append(np.nan)
                            norm2_dist_coloc.append(np.nan)
                            signed_dist_coloc.append(np.nan)
                            xcorrelation_coloc.append(np.nan)
                            zcorrelation_coloc.append(np.nan)
                            ycorrelation_coloc.append(np.nan)
                            original_coloctrackidx.append(np.nan)
                            continue
                        
                        length_coloc.append(len(POI))
                        norm2_dist_coloc.append(norm2_dist)
                        signed_dist_coloc.append(signed_dist)
                        xcorrelation_coloc.append(xcorrelation)
                        ycorrelation_coloc.append(ycorrelation)
                        zcorrelation_coloc.append(zcorrelation)
                        
                        tp = np.zeros(len(POI_tracks[poi_idx]))
                        tp[start:end] = 1
                        timepoints_annotated_by_coloc.append(tp)


                        COLtimepoints_annotated_by_coloc.append(np.in1d(frames_for_col, frames_for_col[idx_in_COL]).astype(int))

                        original_trackidx.append(i+tracks_seen_already)
                        original_coloctrackidx.append(col_idx+coloctracks_seen_already)
                        original_trackidx_interrim.append(i)

            else:
                timepoints_annotated_by_coloc.append(np.zeros(len(POI_tracks[i])))
                COLtimepoints_annotated_by_coloc.append(np.zeros(len(POI_tracks[i])))
                original_trackidx.append(i+tracks_seen_already)
                length_coloc.append(np.nan)
                norm2_dist_coloc.append(np.nan)
                signed_dist_coloc.append(np.nan)
                xcorrelation_coloc.append(np.nan)
                zcorrelation_coloc.append(np.nan)
                ycorrelation_coloc.append(np.nan)
                original_coloctrackidx.append(np.nan)
    tracks_seen_already += len(POI_tracks)
    coloctracks_seen_already += len(coloc_tracks)

timepoints_annotated_by_coloc = np.array(timepoints_annotated_by_coloc)
COLtimepoints_annotated_by_coloc = np.array(COLtimepoints_annotated_by_coloc)
original_trackidx = np.array(original_trackidx)
length_coloc = np.array(length_coloc)
norm2_dist_coloc = np.array(norm2_dist_coloc)
signed_dist_coloc = np.array(signed_dist_coloc)
xcorrelation_coloc = np.array(xcorrelation_coloc)
zcorrelation_coloc = np.array(zcorrelation_coloc)
ycorrelation_coloc = np.array(ycorrelation_coloc)
original_coloctrackidx = np.array(original_coloctrackidx)


print(original_trackidx.shape, original_coloctrackidx.shape, timepoints_annotated_by_coloc.shape)
print(xcorrelation_coloc.shape, zcorrelation_coloc.shape, ycorrelation_coloc.shape)
print(DLP_tracks_all.shape, VP7_tracks_all.shape)


# if any filter below threshold remove coloc track
xcorrelation_coloc_filter = xcorrelation_coloc>corr_threshold
zcorrelation_coloc_filter = zcorrelation_coloc>corr_threshold
ycorrelation_coloc_filter = ycorrelation_coloc>corr_threshold
norm2_dist_coloc_filter = norm2_dist_coloc<postprocess_min_norm2
length_coloc_filter = length_coloc>postprocess_min_coloc_len
print(xcorrelation_coloc_filter)

from copy import deepcopy
# if original_trackidx is duplicated take the ones without nan in original_coloctrackidx
dup_idx = np.unique(original_trackidx)[np.unique(original_trackidx, return_counts=True)[1]>1]

DLP_track_ids_all.shape, 

timepoints_annotated_by_coloc_curated = []
COLtimepoints_annotated_by_coloc_curated = []
original_trackidx_curated = []
length_coloc_curated = []
norm2_dist_coloc_curated = []
signed_dist_coloc_curated = []
xcorrelation_coloc_curated = []
zcorrelation_coloc_curated = []
ycorrelation_coloc_curated = []
original_coloctrackidx_curated = []
original_experiment_curated = []
idx_kept_from_all = []

# cool original id 183, 182

tmp = deepcopy(timepoints_annotated_by_coloc)
direct_idx = np.array(range(len(original_trackidx)))
seen_already = []
for i,idx in enumerate(original_trackidx):
    if idx in seen_already:
        continue
    if idx in dup_idx:
        filtering = original_trackidx==idx
        non_nan_filter = ~np.isnan(original_coloctrackidx[filtering])
        non_nan_idx = direct_idx[filtering][non_nan_filter]
        if len(non_nan_idx)>1:
            selection_crit = []
            selection_crit2 = []
            length_of_coloc = []
            length_of_coloctrack = []
            color_list = ['darkorange', 'red', 'purple', 'pink']
            fig = go.Figure()
            for ni in non_nan_idx:
                coloc_idx_to_check_i = int(original_coloctrackidx[ni])
                length_of_coloc.append(int(length_coloc[ni]))
                length_of_coloctrack.append(int(len(VP7_tracks_all[coloc_idx_to_check_i])))
                selection_crit.append(
                    (xcorrelation_coloc[ni]+zcorrelation_coloc[ni]+ycorrelation_coloc[ni]
                     )*length_coloc[ni])
                selection_crit2.append(
                    (xcorrelation_coloc[ni]+zcorrelation_coloc[ni]+ycorrelation_coloc[ni]
                     )*len(VP7_tracks_all[coloc_idx_to_check_i]))

                DLP_to_plot = DLP_tracks_all[original_trackidx[ni]]
            
            if len(np.unique(length_of_coloc))>1:
                argmax_idx = np.argmax(selection_crit)
                argmax_idx_ni = non_nan_idx[np.argmax(selection_crit)]
            else:
                argmax_idx = np.argmax(selection_crit2)
                argmax_idx_ni = non_nan_idx[np.argmax(selection_crit2)]

            idx_kept_from_all.append(argmax_idx_ni)
            timepoints_annotated_by_coloc_curated.append(tmp[argmax_idx_ni])
            COLtimepoints_annotated_by_coloc_curated.append(COLtimepoints_annotated_by_coloc[argmax_idx_ni])
            original_trackidx_curated.append(original_trackidx[argmax_idx_ni])
            length_coloc_curated.append(length_coloc[argmax_idx_ni])
            norm2_dist_coloc_curated.append(norm2_dist_coloc[argmax_idx_ni])
            signed_dist_coloc_curated.append(signed_dist_coloc[argmax_idx_ni])
            xcorrelation_coloc_curated.append(xcorrelation_coloc[argmax_idx_ni])
            zcorrelation_coloc_curated.append(zcorrelation_coloc[argmax_idx_ni])
            ycorrelation_coloc_curated.append(ycorrelation_coloc[argmax_idx_ni])
            original_coloctrackidx_curated.append(original_coloctrackidx[argmax_idx_ni])
            original_experiment_curated.append(DLP_expname_all[original_trackidx[argmax_idx_ni]])

        elif len(non_nan_idx)==1:
            ni = non_nan_idx[0]
            idx_kept_from_all.append(ni)
            timepoints_annotated_by_coloc_curated.append(tmp[ni])
            COLtimepoints_annotated_by_coloc_curated.append(COLtimepoints_annotated_by_coloc[ni])
            original_trackidx_curated.append(original_trackidx[ni])
            length_coloc_curated.append(length_coloc[ni])
            norm2_dist_coloc_curated.append(norm2_dist_coloc[ni])
            signed_dist_coloc_curated.append(signed_dist_coloc[ni])
            xcorrelation_coloc_curated.append(xcorrelation_coloc[ni])
            zcorrelation_coloc_curated.append(zcorrelation_coloc[ni])
            ycorrelation_coloc_curated.append(ycorrelation_coloc[ni])
            original_coloctrackidx_curated.append(original_coloctrackidx[ni])
            original_experiment_curated.append(DLP_expname_all[original_trackidx[ni]])
    else:
        idx_kept_from_all.append(i)
        timepoints_annotated_by_coloc_curated.append(tmp[i])
        COLtimepoints_annotated_by_coloc_curated.append(COLtimepoints_annotated_by_coloc[i])
        original_trackidx_curated.append(original_trackidx[i])
        length_coloc_curated.append(length_coloc[i])
        norm2_dist_coloc_curated.append(norm2_dist_coloc[i])
        signed_dist_coloc_curated.append(signed_dist_coloc[i])
        xcorrelation_coloc_curated.append(xcorrelation_coloc[i])
        zcorrelation_coloc_curated.append(zcorrelation_coloc[i])
        ycorrelation_coloc_curated.append(ycorrelation_coloc[i])
        original_coloctrackidx_curated.append(original_coloctrackidx[i])
        original_experiment_curated.append(DLP_expname_all[original_trackidx[i]])

    seen_already.append(idx)
timepoints_annotated_by_coloc_curated = np.array(timepoints_annotated_by_coloc_curated, dtype=object)
COLtimepoints_annotated_by_coloc_curated = np.array(COLtimepoints_annotated_by_coloc_curated, dtype=object)
original_trackidx_curated = np.array(original_trackidx_curated)
length_coloc_curated = np.array(length_coloc_curated)
norm2_dist_coloc_curated = np.array(norm2_dist_coloc_curated)
signed_dist_coloc_curated = np.array(signed_dist_coloc_curated)
xcorrelation_coloc_curated = np.array(xcorrelation_coloc_curated)
zcorrelation_coloc_curated = np.array(zcorrelation_coloc_curated)
ycorrelation_coloc_curated = np.array(ycorrelation_coloc_curated)
original_coloctrackidx_curated = np.array(original_coloctrackidx_curated)
original_experiment_curated = np.array(original_experiment_curated)
idx_kept_from_all = np.array(idx_kept_from_all)


uncoating_value = 2
colocalization_label = []
for tp in timepoints_annotated_by_coloc_curated:
    if np.sum(tp)>0:
        segl,cp,val = find_segments(tp)
        if len(val)==1:
            colocalization_label.append(1)
        elif len(val)==2:
            if val[-1]==1 and val[0]==0:
                colocalization_label.append(2) # ends coloc but coloc start missed 
            elif val[-1]==0 and val[0]==1:
                colocalization_label.append(3)
        else:
            colocalization_label.append(4) # transient loss of coloc 
    else:
        colocalization_label.append(0)
colocalization_label = np.array(colocalization_label)

timepoints_annotated_by_coloc_curated_v2 = deepcopy(timepoints_annotated_by_coloc_curated)
for i,tp in enumerate(timepoints_annotated_by_coloc_curated_v2):
    if colocalization_label[i]==4: # fill transient loss of coloc
        segl,cp,val = find_segments(tp)
        if val[0]==0: 
            timepoints_annotated_by_coloc_curated_v2[i][cp[0]:cp[1]] = 1
    if colocalization_label[i]==2:
        segl,cp,val = find_segments(tp)
        if val[0]==0: # if start un-coloc but ends coloc it is coloc
            timepoints_annotated_by_coloc_curated_v2[i][cp[0]:cp[1]] = 1

colocalization_label_v2 = []
for tp in timepoints_annotated_by_coloc_curated_v2:
    if np.sum(tp)>0:
        segl,cp,val = find_segments(tp)
        if len(val)==1:
            colocalization_label_v2.append(1)
        else:
            colocalization_label_v2.append(uncoating_value)
    else:
        colocalization_label_v2.append(0)
colocalization_label_v2 = np.array(colocalization_label_v2)


# %%
print(np.unique(colocalization_label_v2, return_counts=True), np.sum(np.unique(colocalization_label_v2, return_counts=True)[1]), len(timepoints_annotated_by_coloc_curated))


# %%

tracks = DLP_tracks_all[original_trackidx_curated]

X = [x-x[0] for x in tracks]
print(len(X), 'len X', len(np.vstack(X)))
features = ['XYZ', 'SL', 'DP']
X_to_eval = add_features(X, features)
y_to_eval = [np.ones(len(x))*0.5 for x in X_to_eval]

# define dataset and method that model was trained on to find the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
datasets = ['SimDiff_dim3_ntraces300000_Drandom0.0001-0.5_dt1.0e+00_N5-600_B0.05-0.25_R5-25_subA0-0.7_superA1.3-2_Q1-16']
methods = ['XYZ_SL_DP']
dim = 3 if 'dim3' in datasets[0] else 2
# find the model
dir_name = ''
modelpath = 'mlruns/'
modeldir = '36'
use_mlflow = False
if use_mlflow:
    import mlflow
    mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("mlruns")))
    best_models_sorted = find_models_for(datasets, methods)
else:
    # not sorted tho
    path = '../mlruns/{}'.format(modeldir)
    best_models_sorted = find_models_for_from_path(path)
    print(best_models_sorted)

# model params
min_max_len = 601
X_padtoken = 0
y_padtoken = 10
batch_size = 32 # use if using ensemble_score but doesnt change argmax anyways

savename_score = '../deepspt_results/analytics/RotaEEA1NPC1_ensemble_score.pkl'
savename_pred = '../deepspt_results/analytics/RotaEEA1NPC1_ensemble_pred.pkl'
rerun_segmentaion = True
ensemble_score, ensemble_pred = run_temporalsegmentation(
                                best_models_sorted, 
                                X_to_eval, y_to_eval,
                                use_mlflow=use_mlflow,  
                                dir_name=dir_name, 
                                device=device, 
                                dim=dim, 
                                min_max_len=min_max_len, 
                                X_padtoken=X_padtoken, 
                                y_padtoken=y_padtoken,
                                batch_size=batch_size,
                                rerun_segmentaion=rerun_segmentaion,
                                savename_score=savename_score,
                                savename_pred=savename_pred,)

# %%

idx = 2
experiments_unique = np.unique(DLP_expname_all)
DLP_filenames_exp = DLP_filenames_all[original_trackidx_curated][DLP_expname_all[original_trackidx_curated]==experiments_unique[idx]]
DLP_expname_exp = DLP_expname_all[original_trackidx_curated][DLP_expname_all[original_trackidx_curated]==experiments_unique[idx]]
DLP_trackidx_exp = DLP_track_ids_all[original_trackidx_curated][DLP_expname_all[original_trackidx_curated]==experiments_unique[idx]]
DLP_tracks_exp = DLP_tracks_all[original_trackidx_curated][DLP_expname_all[original_trackidx_curated]==experiments_unique[idx]]
ensemble_pred_exp = ensemble_pred[DLP_expname_all[original_trackidx_curated]==experiments_unique[idx]]

angle_degrees = 30
DLP_tracks_exp_rotated = []
for t in DLP_tracks_exp:
    tmp = []
    for p in t:
        tmp.append(rotate_around_y(p, angle_degrees))
    DLP_tracks_exp_rotated.append(np.vstack(tmp))


for i, dte in enumerate(DLP_trackidx_exp):
    if 'p21' in dte:
        print(i, dte)

fig = go.Figure()
# Define a color scale
diffusion_list = DLP_tracks_exp_rotated
diffusion_labels = ensemble_pred_exp
traceidx = None
diff_types = ['Normal', 'Directed', 'Confined', 'Subdiffusive', 'Superdiffusive']
color_dict = {0:'#1f77b4', 1: '#d62728', 2:'#2ca02c', 3:'#ff7f0e', 4:'purple'}
print(len(diffusion_list), len(diffusion_labels))
width = 2

for i in tqdm(np.arange(0,len(diffusion_list),1)):
    fig.add_trace(go.Scatter3d(
            x=diffusion_list[i][:,0], 
            y=diffusion_list[i][:,1], 
            z=diffusion_list[i][:,2],
            mode='lines',
            showlegend=False,
            line=dict(
                color='black',
                width=width
            ),
    ))

DLP_tracks_exp_rotated
# change aspect ratio per axis
fig.update_layout(scene_aspectmode='manual',
                scene_aspectratio=dict(x=1, y=1.5, z=0.25))

# change margins
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

# # remove background
fig.update_layout(scene=dict(xaxis=dict(showbackground=False,
                                        backgroundcolor='white',
                                        showticklabels=False,
                                        showgrid=True,
                                        gridcolor='lightgrey',
                                        zeroline=False,
                                        zerolinecolor='black'),
                            yaxis=dict(showbackground=False,
                                        backgroundcolor='white',
                                        showticklabels=False,
                                        showgrid=True,
                                        gridcolor='lightgrey',
                                        zeroline=False,
                                        zerolinecolor='black'),
                            zaxis=dict(showbackground=False,
                                        backgroundcolor='white',
                                        showticklabels=False,
                                        showgrid=True,
                                        gridcolor='lightgrey',
                                        zeroline=False,
                                        zerolinecolor='black'),
                                        ))
# camera
fig.update_layout(scene_camera=dict(eye=dict(x=2, y=-0.1, z=0.2)))

from pathlib import Path
# save as pdf
fig.write_image('../deepspt_results/figures/Rota_diffcoloured_alltracks3D.pdf')
fig.write_image('../deepspt_results/figures/Rota_black_alltracks3D.pdf')

fig.show()
    

# %%

fp_datapath = '../_Data/Simulated_diffusion_tracks/'
hmm_filename = 'simulated2D_HMM.json'
dim = 3
dt = 4.2
selected_features = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,
                              19,20,21,23,24,25,27,28,29,30,31,
                              32,33,34,35,36,37,38,39,40,41,42])

# # track_uncoats_idx split train and test
track_uncoats_idx = colocalization_label_v2==uncoating_value

print(DLP_tracks_all.shape, VP7_tracks_all.shape)
# uncoating_tracks

uncoating_tracks = np.array(DLP_tracks_all[original_trackidx_curated], dtype=object)[track_uncoats_idx]
uncoating_frames = np.array(DLP_frames_all[original_trackidx_curated], dtype=object)[track_uncoats_idx]
uncoating_timepoints = timepoints_annotated_by_coloc_curated_v2[track_uncoats_idx]
uncoating_files = np.array(DLP_filenames_all[original_trackidx_curated], dtype=object)[track_uncoats_idx]
uncoating_trackids = np.array(DLP_track_ids_all[original_trackidx_curated], dtype=object)[track_uncoats_idx]
uncoating_pred = np.array(ensemble_pred, dtype=object)[track_uncoats_idx]
coloc_idx_vp7 = original_coloctrackidx_curated[colocalization_label_v2==uncoating_value]

uncoatingVP7_files = []
uncoatingVP7_tracks = []
uncoatingVP7_trackids = []
for idx in coloc_idx_vp7:
    i = int(idx)
    uncoatingVP7_tracks.append(VP7_tracks_all[i])
    uncoatingVP7_files.append(VP7_filenames_all[i])
    uncoatingVP7_trackids.append(VP7_track_ids_all[i])
uncoatingVP7_tracks = np.array(uncoatingVP7_tracks, dtype=object)
uncoatingVP7_files = np.array(uncoatingVP7_files, dtype=object)
uncoatingVP7_trackids = np.array(uncoatingVP7_trackids, dtype=object)

frame_change = []
for i in range(timepoints_annotated_by_coloc_curated_v2.shape[0]):
    if colocalization_label_v2[i]!=uncoating_value:
        continue
    segl,cp,val = find_segments(timepoints_annotated_by_coloc_curated_v2[i])
    frame_change.append(cp[-2])
frame_change = np.array(frame_change)

print(len(coloc_idx_vp7), len(frame_change), len(uncoating_timepoints),)

keep_idx = []
min_frames_from_edge = 2
for f,t in zip(frame_change,uncoating_tracks):
    if len(t[:f])>=min_frames_from_edge and len(t[f:])>=min_frames_from_edge:
        keep_idx.append(True)
    else:
        keep_idx.append(False)
keep_idx = np.array(keep_idx)

uncoating_tracks_pruned = uncoating_tracks[keep_idx]
uncoating_frames_pruned = uncoating_frames[keep_idx]
uncoating_timepoints_pruned = uncoating_timepoints[keep_idx]
uncoating_files_pruned = uncoating_files[keep_idx]
uncoating_trackids_pruned = uncoating_trackids[keep_idx]
uncoatingVP7_tracks_pruned = uncoatingVP7_tracks[keep_idx]
uncoatingVP7_files_pruned = uncoatingVP7_files[keep_idx]
uncoatingVP7_trackids_pruned = uncoatingVP7_trackids[keep_idx]
uncoating_pred_pruned = uncoating_pred[keep_idx]
frame_change_pruned = frame_change[keep_idx]

window_size = 30

starttime = time.time()
results2 = Parallel(n_jobs=6)(
        delayed(make_tracks_into_FP_timeseries)(
            track, pred_track, window_size=window_size, selected_features=selected_features,
            fp_datapath=fp_datapath, hmm_filename=hmm_filename, dim=dim, dt=dt)
        for track, pred_track in zip(uncoating_tracks_pruned, uncoating_pred_pruned))
timeseries_clean = np.array([r[0] for r in results2])
print('time taken, using parallel??', (time.time()-starttime)/len(uncoating_tracks_pruned))

length_track = np.hstack([len(t) for t in uncoating_tracks_pruned])
pickle.dump(timeseries_clean, open('../deepspt_results/analytics/timeseries_clean.pkl', 'wb'))
pickle.dump(frame_change_pruned, open('../deepspt_results/analytics/frame_change_pruned.pkl', 'wb'))
pickle.dump(length_track, open('../deepspt_results/analytics/length_track.pkl', 'wb'))
pickle.dump(uncoating_tracks_pruned, open('../deepspt_results/analytics/escape_tracks_all.pkl', 'wb'))
pickle.dump(uncoatingVP7_tracks_pruned, open('../deepspt_results/analytics/VP7escape_tracks_all.pkl', 'wb'))

print(uncoatingVP7_tracks_pruned.shape, uncoating_tracks_pruned.shape, uncoating_pred_pruned.shape)
print(timeseries_clean[0].shape, timeseries_clean.shape)

threshold = .6

i_idx = []
j_idx = []
like_pairs = []
rmsd = []
seen_idx = []
for i in range(len(uncoating_tracks_pruned)):
    for j in range(len(uncoating_tracks_pruned)):
        if i==j:
            continue
        # if i in seen_idx:
        #     continue

        f1 = uncoating_frames_pruned[i]
        f2 = uncoating_frames_pruned[j]

        # where f1 and f2 are identical
        f1_idx = np.in1d(f1, f2)
        f2_idx = np.in1d(f2, f1)
        if np.sum(f1_idx)<2 or np.sum(f2_idx)<2:
            continue

        t1 = uncoating_tracks_pruned[i][f1_idx]
        t2 = uncoating_tracks_pruned[j][f2_idx]

        r = np.sqrt(np.mean((t1-t2)**2))


        if r<threshold:
            i_idx.append(i)
            j_idx.append(j)
            seen_idx.append(i)
            seen_idx.append(j)
            rmsd.append(r)
            min_pi = np.min([i,j])
            max_pi = np.max([i,j])
            like_pairs.append([min_pi,max_pi])

rmsd = np.array(rmsd)
i_idx = np.array(i_idx)
j_idx = np.array(j_idx)

i_idx_under1 = i_idx[rmsd<threshold]
j_idx_under1 = j_idx[rmsd<threshold]
rmsd_under1 = rmsd[rmsd<threshold] 
len(j_idx_under1), i_idx_under1, j_idx_under1, rmsd_under1

uniq_like_pairs = np.unique(like_pairs, axis=0, return_index=True)[0]
uniq_like_pairs_idx = np.unique(like_pairs, axis=0, return_index=True)[1]
uniq_like_rmsd = rmsd[uniq_like_pairs_idx]
pickle.dump(uniq_like_pairs, open('../deepspt_results/analytics/uniq_like_pairs.pkl', 'wb'))

# %%

i,j = 2,3
print(timeseries_clean.shape, timeseries_clean[i].shape, np.sum(timeseries_clean[i]))
print(timeseries_clean.shape, timeseries_clean[j].shape, np.sum(timeseries_clean[j]))
globals.seed

# %%
np.unique(np.hstack(ensemble_pred), return_counts=True)


# %%