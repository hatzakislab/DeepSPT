# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle 
import pandas as pd
import similaritymeasures
from .deepspt_helper_functions import create_fingerprint_track, find_segments
import datetime
from collections import defaultdict


def confined_subdiffusive_from_standard_sim(output):
    clean = output[:int(0.2*len(output))]
    conf_sub = clean[len(clean)//2:]
    return conf_sub
    

def read_data(datapath, filename):
    if filename.split('.')[-1]=='csv':
        X = pd.read_csv( open( datapath + filename)).sort_values(by=['particle', 'frame']).reset_index(drop=True)
    elif filename.split('.')[-1]=='pkl':
        X = pickle.load(open(datapath + filename, 'rb'))
    return X


def find_nearest_coloc(tracks, track_frames, coloc_tracks, coloc_frames, 
                       threshold=0.3, min_coloc_len=20):
    def squared_dist_intersect(p,c,track_idx_intersect,coloc_idx_intersect):
        d = np.sqrt(np.sum((p[track_idx_intersect,:]-c[coloc_idx_intersect,:])**2, axis=1))
        return d
    poi_idx_list = []
    col_idx_list = []
    coloc_timelabels = []
    for i, p in enumerate(tqdm(tracks)):
        frames = track_frames[i]
        for j, c in enumerate(coloc_tracks):
            if len(np.intersect1d(frames,coloc_frames[j]))>0:
                track_idx_intersect = np.isin(frames,coloc_frames[j])
                coloc_idx_intersect = np.isin(coloc_frames[j],frames)
                d = squared_dist_intersect(p,c,track_idx_intersect,coloc_idx_intersect)
                close = np.where(d<threshold, 1, 0)
                if np.sum(close)>min_coloc_len:
                    timelabel = np.zeros(len(frames))
                    timelabel[track_idx_intersect] = close
                    coloc_timelabels.append(timelabel)
                    poi_idx_list.append(i)
                    col_idx_list.append(j) 
    return np.array(coloc_timelabels, dtype=object), np.array(poi_idx_list, dtype=object), np.array(col_idx_list, dtype=object)

def generate_coloc_output(tracks, track_frames, coloc_tracks, coloc_frames, threshold=0.4, min_coloc_len=20,
                          blinkinglength_threshold=3):
    coloc_timelabels, poi_idx_list, col_idx_list = find_nearest_coloc(
                                                    tracks, track_frames, 
                                                    coloc_tracks, coloc_frames, 
                                                    threshold=threshold, 
                                                    min_coloc_len=min_coloc_len)
    promising_idx = [i for i,c in enumerate(coloc_timelabels) if np.sum(c)>=min_coloc_len]

    coloc_timelabels = coloc_timelabels[promising_idx]
    poi_idx_list = poi_idx_list[promising_idx]
    col_idx_list = col_idx_list[promising_idx]

    changepoint_list, seglens_list = [], []
    accepted_blinking_list, rejected_blinking_list = [], []
    for i in range(len(coloc_timelabels)):

        coloc_timelabel = coloc_timelabels[i]

        # get the idx where there supposedly is coloc
        seglens, changepoint, label = find_segments(coloc_timelabel)
        changepoint = changepoint[np.where(label==1)]
        seglens = seglens[np.where(label==1)]

        coloc_timelabel, accepted_blinking, rejected_blinking = consider_merging_coloc_stretches(changepoint, 
                                                                seglens, coloc_timelabel, 
                                                                blinkinglength_threshold=blinkinglength_threshold)

        seglens, changepoint, label = find_segments(coloc_timelabel)

        changepoint = changepoint[np.where(label==1)]
        seglens = seglens[np.where(label==1)]

        changepoint_list.append(changepoint)
        seglens_list.append(seglens)
        accepted_blinking_list.append(accepted_blinking)
        rejected_blinking_list.append(rejected_blinking)

    return promising_idx, changepoint_list, seglens_list,poi_idx_list, col_idx_list,\
        accepted_blinking_list, rejected_blinking_list

# # threshold and filter colocs
# def create_coloc_label(dist_matrices, threshold=0.3, min_coloc_len=20):
#     coloc_timelabels = []
#     poi_idx_list = []
#     col_idx_list = []
#     for poi_idx,d_matrix in enumerate(dist_matrices):
#         for col_idx, d in enumerate(d_matrix):
#             close = np.where(d<threshold, 1, 0)
#             coloc_timelabels.append(close)
#             poi_idx_list.append(poi_idx)
#             col_idx_list.append(col_idx) 
#     return np.array(coloc_timelabels, dtype=object), np.array(poi_idx_list, dtype=object), np.array(col_idx_list, dtype=object)


def consider_merging_coloc_stretches(changepoint, seglens, coloc_timelabels, 
                                     blinkinglength_threshold=1):
    accepted_blinking = 0
    rejected_blinking = 0
    if len(changepoint)>1:
        for cp in range(len(changepoint[:-1])):
            cond1 = changepoint[cp+1]-(changepoint[cp]+seglens[cp]) <= blinkinglength_threshold
            if cond1:
                accepted_blinking +=1
                coloc_timelabels[changepoint[cp]:changepoint[cp+1]+seglens[cp+1]] = 1
            else:
                rejected_blinking +=1
    return coloc_timelabels, accepted_blinking, rejected_blinking




def similarity_measures(POI, COL, dist_matrix, start, end):
    # Partial Curve Mapping
    # Lower is better
    pcm = similaritymeasures.pcm(POI,COL)
    
    # Discrete Frechet distance
    # Lower is better
    df = similaritymeasures.frechet_dist(POI,COL)

    # area between two curves
    # Lower is better
    area = similaritymeasures.area_between_two_curves(POI,COL)

    # Curve Length based similarity measure
    # closer to 1 is better
    cl = 1-(np.sum((COL[1:]-COL[:-1])**2)/np.sum((POI[1:]-POI[:-1])**2))

    # Dynamic Time Warping distance
    # Lower is better
    dtw, _ = similaritymeasures.dtw(POI,COL)

    # Lock step Euclidean distance
    # Lower is better
    LSED = np.array(dist_matrix[start:end])/len(POI)

    # Cosine similarity
    # Higher is better
    from scipy.spatial.distance import cosine
    cos = [1-cosine(POI[v+1]-POI[v], COL[v+1]-COL[v]) 
            for v in range(len(POI)-1)]

    # Braycurtis
    # Higher is better
    from scipy.spatial.distance import braycurtis
    bra = [1-braycurtis(POI[v+1]-POI[v], COL[v+1]-COL[v])
            for v in range(len(POI)-1)]

    # Correlation
    # Higher is better
    a, b = POI[:,0], COL[:,0]
    a = (a - np.mean(a))/(np.std(a)*len(a))
    b = (b - np.mean(b))/(np.std(b))
    corr_x = np.correlate(a,b,mode='valid')[0]
    a, b = POI[:,1], COL[:,1]
    a = (a - np.mean(a))/(np.std(a)*len(a))
    b = (b - np.mean(b))/(np.std(b))
    corr_y = np.correlate(a,b,mode='valid')[0]
    return pcm/len(POI), df, area/len(POI), cl, dtw/len(POI), LSED/len(POI), cos, bra, corr_x, corr_y


def rotate(vector, theta, rotation_around=None) -> np.ndarray:
    """
    reference: https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    :param vector: list of length 2 OR
                   list of list where inner list has size 2 OR
                   1D numpy array of length 2 OR
                   2D numpy array of size (number of points, 2)
    :param theta: rotation angle in degree (+ve value of anti-clockwise rotation)
    :param rotation_around: "vector" will be rotated around this point, 
                    otherwise [0, 0] will be considered as rotation axis
    :return: rotated "vector" about "theta" degree around rotation
             axis "rotation_around" numpy array
    """
    vector = np.array(vector)
    if vector.ndim == 1:
        vector = vector[np.newaxis, :]
    if rotation_around is not None:
        vector = vector - rotation_around
    vector = vector.T
    theta = np.radians(theta)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    output: np.ndarray = (rotation_matrix @ vector).T
    if rotation_around is not None:
        output = output + rotation_around
    return output.squeeze()

def translate(vector, x_displace, y_displace) -> np.ndarray:
    output = np.zeros_like(vector)
    output[:,0] = vector[:,0] + x_displace
    output[:,1] = vector[:,1] + y_displace
    return output

def gaussian_noise(vector, noise_std) -> np.ndarray:
    output = np.zeros_like(vector)
    output[:,0] = vector[:,0] + np.random.normal(0, noise_std, size=vector[:,0].shape)
    output[:,1] = vector[:,1] + np.random.normal(0, noise_std, size=vector[:,1].shape)
    return output

def generate_pertubed_tracks(X, threshold=0.4, noise_level=1):
    altered_tracks_dict = defaultdict(list)

    for vec in X:
        theta = np.random.uniform(40,320) #random
        altered_tracks_dict['rotated'].append(rotate(vec, theta))
        
    for vec in X:
        x_displace = np.random.uniform(0,threshold)
        y_displace = np.random.uniform(0,threshold)
        while threshold<np.sqrt(x_displace**2 + y_displace**2):
            x_displace = np.random.uniform(0,threshold)
            y_displace = np.random.uniform(0,threshold)
        altered_tracks_dict['translated'].append(translate(vec, x_displace, y_displace))

    for vec in X:
        theta = np.random.uniform(0,360) #random
        out = rotate(vec, theta)

        x_displace = np.random.uniform(0,threshold)
        y_displace = np.random.uniform(0,threshold)
        while threshold<np.sqrt(x_displace**2 + y_displace**2):
            x_displace = np.random.uniform(0,threshold)
            y_displace = np.random.uniform(0,threshold)
        altered_tracks_dict['rotat_trans'].append(translate(out, x_displace, y_displace))

    for vec in X:
        altered_tracks_dict['gaussnoise'].append(gaussian_noise(vec, noise_level))

    for vec in X:
        noise_std = np.random.uniform(0.05, noise_level)
        x_displace = np.random.uniform(0,threshold)
        y_displace = np.random.uniform(0,threshold)
        while threshold<np.sqrt(x_displace**2 + y_displace**2):
            x_displace = np.random.uniform(0,threshold)
            y_displace = np.random.uniform(0,threshold)
        altered_tracks_dict['gaussnois_trans'].append(gaussian_noise(translate(vec, x_displace, y_displace), noise_std))

    return altered_tracks_dict

def metric_filtering(sim_measures, metric_thresholds):
    cond0 = sim_measures[0] < metric_thresholds['pcm']
    cond1 = sim_measures[1] < metric_thresholds['df'] 
    cond2 = sim_measures[2] < metric_thresholds['area'] 
    cond3 = sim_measures[3] < metric_thresholds['cl']
    cond4 = sim_measures[4] < metric_thresholds['dtw']
    cond5 = np.nanmean(sim_measures[5]) < metric_thresholds['LSED']
    cond6 = np.nanmean(sim_measures[6]) > metric_thresholds['cos']
    cond7 = np.nanmean(sim_measures[7]) > metric_thresholds['bra']
    cond8 = sim_measures[8] > metric_thresholds['corr_x']
    cond9 = sim_measures[9] > metric_thresholds['corr_y']
    cond = (cond0 and cond1 and cond2 and cond3 and cond4 and 
            cond5 and cond6 and cond7 and cond8 and cond9)
    if cond:
        return True
    else:
        return False


def fp_filtering(fp_measures, metric_thresholds):
    cond0 = fp_measures[list(fp_filter.keys())[0]] < list(metric_thresholds.values())[0]
    cond1 = fp_measures[list(fp_filter.keys())[1]] < list(metric_thresholds.values())[1] 
    cond2 = fp_measures[list(fp_filter.keys())[2]] < list(metric_thresholds.values())[2] 
    cond3 = fp_measures[list(fp_filter.keys())[3]] < list(metric_thresholds.values())[3]
    cond = cond0 and cond1 and cond2 and cond3
    if cond:
        return True
    else:
        return False