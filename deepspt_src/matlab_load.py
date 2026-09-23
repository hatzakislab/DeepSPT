# %%
import re
from glob import glob
import h5py
import scipy.io
import numpy as np
import pandas as pd
import pickle


def index(m, track_id, column):
    """
    Indexing in the matlab h5 file returns a reference only. This reference is
    then used to go back and find the values in the file.
    """
    ref = m[column][track_id][0]
    
    return np.array(m[ref][:])


def cme_tracks_to_pandas(mat_path, project_name):
    """
    Converts CME-derived ProcessedTracks.mat to Pandas DataFrame format.

    This version was specifically created for matlab 7.3 files.

    Add extra columns as required.
    """
    COLUMNS = "A", "x", "y", "z", "c", 'sigma_r', 't'

    h5file = h5py.File(mat_path, "r")
    m = h5file["tracks"]
    n_tracks = len(m["A"])
    df = []
    for i in range(n_tracks):
        # # Extract columns
        A, x, y, z, c, sig, t_in_seconds = [index(m=m, track_id=i, column=c) for c in COLUMNS]
        A_ch1, A_ch2, A_ch3 = A[:,0], A[:,1], A[:,2]
        A_sig1, A_sig2, A_sig3 = sig[:,0], sig[:,1], sig[:,2]
        x_ch1, x_ch2, x_ch3 = x[:,0], x[:,1], x[:,2]
        y_ch1, y_ch2, y_ch3 = y[:,0], y[:,1], y[:,2]
        z_ch1, z_ch2, z_ch3 = z[:,0], z[:,1], z[:,2]
        A_bg1, A_bg2, A_bg3 = c[:,0], c[:,1], c[:,2]
        t_in_seconds = t_in_seconds.flatten()

        track_len = len(A)
        # # Find out where parent dirs can be skipped
        real_dir = re.search(string=mat_path, pattern=project_name)
        end_dir =  re.search(string=mat_path, pattern="Analysis")
        
        # # Create path from actual directory
        filepath = mat_path[real_dir.end()+1:end_dir.start()-1]

        group = pd.DataFrame(
            {
                "file": np.repeat(filepath, track_len),
                "particle": np.repeat(i, track_len),
                "id": np.repeat('p'+str(i)+'_'+filepath, track_len),
                't_in_seconds': t_in_seconds,
                "A_ch1": A_ch1,
                "A_ch2": A_ch2,
                "A_ch3": A_ch3,
                "A_sig1": A_sig1,
                "A_sig2": A_sig2,
                "A_sig3": A_sig3,
                "x_ch1": x_ch1,
                "x_ch2": x_ch2,
                "x_ch3": x_ch3,
                "y_ch1": y_ch1,
                "y_ch2": y_ch2,
                "y_ch3": y_ch3,
                "z_ch1": z_ch1,
                "z_ch2": z_ch2,
                "z_ch3": z_ch3,
                "A_bg1": A_bg1,
                "A_bg2": A_bg2,
                "A_bg3": A_bg3
            }
        )

        df.append(group)
        group.fillna(method="ffill", inplace=True)

    return pd.concat(df)


# def extract_matlab_files_from_zip(name):
#     archive = ZipFile(name+'.zip','r')
#     member = []
#     for fp in archive.namelist():
#         if 'ProcessedTracks.mat' in fp:
#             member.append(fp)
#     ZipFile.extract(member, path='extracted', pwd=None)

def main(name, input, output):
    _input = input.format(name)
    _output = output.format(name)

    files = sorted(glob(_input, recursive=True))
    print("\nFound:", len(files), 'files')
    print()
    df = pd.concat(
        [cme_tracks_to_pandas(f, project_name=name) for f in files]
    )

    print("Number of files in df: {}".format(len(df["file"].unique())))
    print("Number of traces in df: {}".format(len(df["id"].unique())))

    # ALl traces
    #df.to_hdf(_output, key="df")

    print("Each trace will be tagged with 'file' like:")
    print(df["id"].values[0])
    print(df.columns)
    pickle.dump(df, open(_output, 'wb'))
    return df


if __name__ == "__main__":
    import datetime
    start = datetime.datetime.now()
    # Project name goes into curly path
    PROJECT_NAMES = "VirusVSVSARSCoV2"

    # Search for tracks in this path. ** means multiple wildcard subdirectories
    SEARCH_PATTERN = "{}/**/ProcessedTracks.mat"

    # Output to a file that also contains the project name in the curly bracket
    OUTPUT_NAME = "{}/processed_tracks_df.pkl"

    df = main(name=PROJECT_NAMES, input=SEARCH_PATTERN, output=OUTPUT_NAME)
    print('Time loading:', datetime.datetime.now()-start)