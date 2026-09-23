# %%
from torch.distributions import Normal, Independent, MultivariateNormal
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

from scipy.special import gamma, factorial
from scipy import stats

from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.metrics import *

from joblib import Parallel, delayed
from os.path import join
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import datetime
import inspect
import random
import pickle
import h5py
import math
import os
import re

try:  # only used by dead mlflow-lookup helpers; not on any live code path
    from mlflow.entities import ViewType
    import mlflow
except ImportError:  # pragma: no cover
    ViewType = None
    mlflow = None

from .hypopt_temporal_segmentation import hypoptUNet
from .temporal_segmentation import UNet
from .Fingerprint_functions import *
from iminuit.cost import LeastSquares
from .statbib import Chi2Fit
from iminuit import Minuit

from matplotlib.legend_handler import HandlerTuple
from matplotlib.collections import LineCollection
from plotly.subplots import make_subplots
import matplotlib.patches as patches
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import matplotlib


def load_val_test_from_idx(X, y, best_model_path):

    direc = "Unet_results/mlruns/"
    val_idx = torch.load(direc + best_model_path + "/CV_indices/CVfold0_D_val_idx.pt")
    test_idx = torch.load(direc + best_model_path + "/CV_indices/CVfold0_D_test_idx.pt")

    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_val, y_val, X_test, y_test


def load_Xy(datapath, filename_X, filename_y, features=["XYZ", "SL", "DP"]):
    X = np.array(
        pickle.load(open(datapath + filename_X, "rb")), dtype=object
    )  # .astype(np.float64)
    if len(filename_y) > 0:
        y = np.array(pickle.load(open(datapath + filename_y, "rb")), dtype=object)
    else:
        y = np.array([np.ones(len(xi)) for xi in X], dtype=object) * 0.5
    X = prep_tracks(X)
    X = add_features(X, features)

    return X, y[: len(X)]


def load_X(
    datapath,
    filename_X,
    identifiername="particle",
    timename="frame",
    features=["XYZ", "SL", "DP"],
):
    print(filename_X)
    if filename_X.split(".")[-1] == "csv":
        X = (
            pd.read_csv(open(datapath + filename_X))
            .sort_values(by=[identifiername, timename])
            .reset_index(drop=True)
        )
        print(X.columns)
    elif filename_X.split(".")[-1] == "pkl":
        X = pickle.load(open(datapath + filename_X, "rb"))
    X = prep_tracks(X)
    X = np.array(X, dtype=object)
    X = add_features(X, features)
    return X


def prep_csv_tracks_track(
    val,
    xname="x",
    yname="y",
    zname="",
    timename="A",
    identifiername="particle",
    center=False,
):
    if zname == "":
        X = np.vstack(val[[xname, yname]].values).astype(float)
        T = np.vstack(val[timename].values).astype(float)
        P = np.vstack(val[identifiername].values).astype(int)
    else:
        X = np.vstack(val[[xname, yname, zname]].values).astype(float)
        T = np.vstack(val[timename].values).astype(float)
        P = np.vstack(val[identifiername].values).astype(int)
    if center:
        X = [x - x[0] for x in X]
    return X, T, P


def read_data_csv(filepath, useful_col=["x", "y", "steplength", "particle", "frame"]):
    return (
        pd.read_csv(filepath)
        .sort_values(by=["particle", "frame"])
        .reset_index(drop=True)[useful_col]
    )


def ensemble_voting(r1, r2, r3):
    return np.array(
        [
            torch.stack(
                [
                    torch.from_numpy(r1[i]),
                    torch.from_numpy(r2[i]),
                    torch.from_numpy(r3[i]),
                ]
            )
            .mode(0)
            .values.numpy()
            for i in range(len(r1))
        ],
        dtype=object,
    )


def ensemble_scoring(r1, r2, r3):
    return [(r1[i] + r2[i] + r3[i]) / 3 for i in range(len(r1))]


def make_preds(
    model,
    X_to_eval,
    y_to_eval,
    min_max_len=601,
    device="cpu",
    X_padtoken=0,
    y_padtoken=10,
    batch_size=256,
    temperature=7.0,
):
    tmp_dict = {}
    max_len = np.max([len(x) for x in X_to_eval])
    max_len = max_len if max_len > min_max_len else min_max_len
    D = TestSetData(
        X_to_eval,
        y_to_eval,
        X_padtoken=X_padtoken,
        y_padtoken=y_padtoken,
        maxlens=max_len,
        device=device,
    )
    model.eval()
    model.to(device)
    test_loader = DataLoader(D, batch_size=batch_size)
    masked_pred, masked_score, masked_y = model.predict(test_loader)
    tmp_dict["masked_pred"] = masked_pred
    tmp_dict["masked_score"] = masked_score
    tmp_dict["masked_y"] = masked_y
    return tmp_dict


def make_temperature_pred(
    model,
    X_to_eval,
    y_to_eval,
    min_max_len=601,
    device="cpu",
    X_padtoken=0,
    y_padtoken=10,
    batch_size=256,
    temperature=3.8537957365297553,
):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.eval()
    calib_model = model
    calib_model.eval()

    tmp_dict = {}
    max_len = np.max([len(x) for x in X_to_eval])
    max_len = max_len if max_len > min_max_len else min_max_len
    D = TestSetData(
        X_to_eval,
        y_to_eval,
        X_padtoken=X_padtoken,
        y_padtoken=y_padtoken,
        maxlens=max_len,
        device=device,
    )
    val_loader = DataLoader(D, batch_size=batch_size)

    with torch.no_grad():
        masked_preds = []
        masked_ys = []
        masked_argmax = []
        for idx, xb in enumerate(tqdm(val_loader)):
            _, y = xb
            calib_model.module_list[-1].conv_softmax[1].register_forward_hook(
                get_activation(str(idx))
            )
            _ = calib_model(xb)
            pred = activation[str(idx)] / temperature
            pred = nn.Softmax(dim=1)(pred)

            for i in range(len(y)):
                mask_idx = sum(y[i].ge(y_padtoken))
                masked_y = y[i][mask_idx:].unsqueeze(0)
                masked_pred = pred[i][:, mask_idx:].unsqueeze(0)
                masked_ys.append(masked_y.cpu().squeeze(0).numpy())
                masked_preds.append(masked_pred.detach().cpu().squeeze(0).numpy())
                masked_argmax.append(
                    masked_pred.squeeze(0).detach().cpu().argmax(0).numpy()
                )

    tmp_dict["masked_pred"] = masked_argmax
    tmp_dict["masked_score"] = masked_preds
    tmp_dict["masked_y"] = masked_ys
    return tmp_dict


def temperature_pred(
    X_val, y_val, model, temperature=4, X_padtoken=10, y_padtoken=0, device="cpu"
):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.eval()
    calib_model = model
    calib_model.eval()

    D_val = CustomDataset(
        X_val, y_val, X_padtoken=X_padtoken, y_padtoken=y_padtoken, device=device
    )
    val_loader = DataLoader(D_val, batch_size=64)

    with torch.no_grad():
        masked_preds = []
        for idx, xb in enumerate(val_loader):
            _, y = xb
            calib_model.module_list[-1].conv_softmax[1].register_forward_hook(
                get_activation(str(idx))
            )
            _ = calib_model(xb)
            pred = activation[str(idx)] / temperature
            pred = nn.Softmax(dim=1)(pred)

            for i in range(len(y)):
                mask_idx = sum(y[i].ge(y_padtoken))
                masked_pred = pred[i][:, mask_idx:].unsqueeze(0)
                masked_preds.append(masked_pred.detach().cpu().numpy())
        masked_preds = [i for s in masked_preds for i in s]
    return masked_preds


def make_preds_parallel(model, xb):
    print(xb.shape)
    model.eval()
    masked_pred, masked_score, masked_y = model.simple_predict(xb)
    return masked_pred, masked_score, masked_y


def load_UnetModels_directly(
    best_model_path="Unet_results/mlruns/", device="cpu", dim=2
):
    model_state_dict = torch.load(
        best_model_path, map_location=torch.device(device), weights_only=False
    )["model_state_dict"]
    if dim == 2:
        # manually set hyperparams
        n_classes = 4
        n_features = 4
        init_channels = 130
        channel_multiplier = 2
        depth = 4
        dil_rate = 2
        kernelsize = 7
        outconv_kernel = 3
        enc_conv_nlayers = 2
        bottom_conv_nlayers = 3
        dec_conv_nlayers = 1
        out_nlayers = 4
        batchnorm = True
        batchnormfirst = True
        pooling = "max"
        pools = [2, 2, 2, 2, 2]
        X_padtoken = 0
        y_padtoken = 10

        model = hypoptUNet(
            n_features=n_features,
            init_channels=init_channels,
            n_classes=n_classes,
            depth=depth,
            enc_kernel=kernelsize,
            dec_kernel=kernelsize,
            outconv_kernel=outconv_kernel,
            dil_rate=dil_rate,
            pools=pools,
            pooling=pooling,
            enc_conv_nlayers=enc_conv_nlayers,
            bottom_conv_nlayers=bottom_conv_nlayers,
            dec_conv_nlayers=dec_conv_nlayers,
            out_nlayers=out_nlayers,
            X_padtoken=X_padtoken,
            y_padtoken=y_padtoken,
            channel_multiplier=channel_multiplier,
            batchnorm=batchnorm,
            batchnormfirst=batchnormfirst,
            device=device,
        )

        model.load_state_dict(model_state_dict)
    elif dim == 3:
        # manually set hyperparams
        n_classes = 4
        n_features = 5
        init_channels = 48
        channel_multiplier = 2
        depth = 3
        dil_rate = 2
        kernelsize = 5
        outconv_kernel = 3
        enc_conv_nlayers = 3
        bottom_conv_nlayers = 4
        dec_conv_nlayers = 4
        out_nlayers = 2
        batchnorm = True
        batchnormfirst = True
        pooling = "max"
        pools = [2, 2, 2]
        X_padtoken = 0
        y_padtoken = 10

        model = hypoptUNet(
            n_features=n_features,
            init_channels=init_channels,
            n_classes=n_classes,
            depth=depth,
            enc_kernel=kernelsize,
            dec_kernel=kernelsize,
            outconv_kernel=outconv_kernel,
            dil_rate=dil_rate,
            pools=pools,
            pooling=pooling,
            enc_conv_nlayers=enc_conv_nlayers,
            bottom_conv_nlayers=bottom_conv_nlayers,
            dec_conv_nlayers=dec_conv_nlayers,
            out_nlayers=out_nlayers,
            X_padtoken=X_padtoken,
            y_padtoken=y_padtoken,
            channel_multiplier=channel_multiplier,
            batchnorm=batchnorm,
            batchnormfirst=batchnormfirst,
            device=device,
        )

        model.load_state_dict(model_state_dict)
    return model


def load_UnetModels(best_model_path, dir="Unet_results/mlruns/", device="cpu"):
    files = sorted(glob(dir + best_model_path + "/*_UNETmodel.torch", recursive=True))
    model_state_dict = torch.load(
        files[0], map_location=torch.device(device), weights_only=False
    )["model_state_dict"]
    id = best_model_path.split("/")[1]

    n_classes = grab_metric(id, "Number of classes")
    n_features = grab_metric(id, "Number of input features")

    init_channels = grab_metric(id, "Initial Number of Channels")
    channel_multiplier = grab_metric(id, "Channel multiplier")

    depth = grab_metric(id, "Depth of model")
    dil_rate = grab_metric(id, "Dilation rate")

    kernelsize = grab_metric(id, "Unet kernel size")
    outconv_kernel = grab_metric(id, "Output block kernel size")

    enc_conv_nlayers = grab_metric(id, "Number encoder layers")
    bottom_conv_nlayers = grab_metric(id, "Number bottom layers")
    dec_conv_nlayers = grab_metric(id, "Number decoder layers")
    out_nlayers = grab_metric(id, "Number output layers")

    batchnorm = grab_params(best_model_path, "Batchnorm")
    batchnormfirst = grab_params(best_model_path, "BN before ReLU")

    pooling = grab_params(best_model_path, "Pooling type")
    pools = [
        grab_metric(id, "pools" + str(i))
        for i in range(grab_metric(id, "Depth of model"))
    ]
    X_padtoken = int(grab_params(best_model_path, "X padtoken"))
    y_padtoken = int(grab_params(best_model_path, "y padtoken"))

    model = hypoptUNet(
        n_features=n_features,
        init_channels=init_channels,
        n_classes=n_classes,
        depth=depth,
        enc_kernel=kernelsize,
        dec_kernel=kernelsize,
        outconv_kernel=outconv_kernel,
        dil_rate=dil_rate,
        pools=pools,
        pooling=pooling,
        enc_conv_nlayers=enc_conv_nlayers,
        bottom_conv_nlayers=bottom_conv_nlayers,
        dec_conv_nlayers=dec_conv_nlayers,
        out_nlayers=out_nlayers,
        X_padtoken=X_padtoken,
        y_padtoken=y_padtoken,
        channel_multiplier=channel_multiplier,
        batchnorm=batchnorm,
        batchnormfirst=batchnormfirst,
        device=device,
    )

    model.load_state_dict(model_state_dict)
    return model


def load_oldUnetModels(best_model_path, dir="Unet_results/mlruns/", device="cpu"):
    files = sorted(glob(dir + best_model_path + "/*_UNETmodel.torch", recursive=True))
    model_state_dict = torch.load(
        files[0], map_location=torch.device(device), weights_only=False
    )["model_state_dict"]
    id = best_model_path.split("/")[1]

    n_classes = grab_metric(id, "Number of classes")
    n_features = grab_metric(id, "Number of input features")

    init_channels = grab_metric(id, "Initial Number of Channels")

    depth = grab_metric(id, "Depth of model")
    dil_rate = grab_metric(id, "Dilation rate")

    kernelsize = grab_metric(id, "Encoder kernel size")

    pools = [
        grab_metric(id, "pools" + str(i))
        for i in range(grab_metric(id, "Depth of model"))
    ]
    X_padtoken = int(grab_params(best_model_path, "X padtoken"))
    y_padtoken = int(grab_params(best_model_path, "y padtoken"))

    model = UNet(
        n_features=n_features,
        init_channels=init_channels,
        n_classes=n_classes,
        depth=depth,
        enc_kernel=kernelsize,
        dil_rate=dil_rate,
        pools=pools,
        X_padtoken=X_padtoken,
        y_padtoken=y_padtoken,
        device=device,
    )

    model.load_state_dict(model_state_dict)
    return model


def timestamper():
    date = (
        str(datetime.datetime.now().year)
        + str(datetime.datetime.now().month)
        + str(datetime.datetime.now().day)
    )
    T = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute)
    return date + T


def make_experiment_name_from_tags(tags: dict):
    return "".join([t + "_" + tags[t] + "__" for t in tags.keys()])


def find_experiments_by_tags(tags: dict):
    exps = mlflow.tracking.MlflowClient().search_experiments()

    def all_tags_match(e):
        for tag in tags.keys():
            if tag not in e.tags:
                return False
            if e.tags[tag] != tags[tag]:
                return False
        return True

    return [e for e in exps if all_tags_match(e)]


def grab_metric(id, string):
    return int(mlflow.tracking.MlflowClient().get_metric_history(id, string)[0].value)


def grab_params(best_model_path, string):
    exp_id = best_model_path.split("/")[0]
    run_id = best_model_path.split("/")[1]
    runs = mlflow.search_runs(
        experiment_ids=[exp_id], run_view_type=ViewType.ACTIVE_ONLY
    )
    return runs[runs["run_id"] == run_id]["params." + string].values[0]


def prep_pools_string(pools_str):
    pools_str = pools_str.replace(",", "")
    pools_str = pools_str.replace("[", "")
    pools_str = pools_str.replace("]", "")
    pools_str = pools_str.split(" ")
    return [int(x) for x in list(pools_str)]


def find_models_for_from_path(path):
    # load from path
    files = sorted(glob(path + "/*/*_UNETmodel.torch", recursive=True))
    return files


def find_models_for(datasets, methods):
    # load using mlflow
    r = mlflow_to_resultsdict_v2(datasets, methods, modelname="Unet")
    DATASET = datasets[0]
    trained_models = list(r[DATASET].keys())
    print("trained_models", trained_models)
    models_max_val_acc = [
        np.max(r[DATASET][model]["VAL_ACCS"]) for model in trained_models
    ]
    best_model_idx = np.argmax(models_max_val_acc)
    best_models_sorted = np.array(trained_models)[np.argsort(models_max_val_acc)[::-1]]
    return best_models_sorted


def mlflow_to_resultsdict_v2(datasets, methods, modelname="Unet"):
    results_dict = {}
    for dataset in datasets:
        sub_results = {}
        for method in methods:
            tag = {"DATASET": dataset, "METHOD": method, "MODEL": modelname}
            exps = find_experiments_by_tags(tag)
            if len(exps) == 0:
                continue
            assert len(exps) == 1, "No experiment with this tag or two identical tags"

            runs = mlflow.search_runs(
                experiment_ids=[exps[0].experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
            )
            for i, id in enumerate(runs["run_id"].to_list()):
                tmp_dict = {}
                folds_train_loss = []
                folds_train_acc = []
                folds_val_loss = []
                folds_val_acc = []
                for fold in range(grab_metric(id, "Cross Validation Folds")):
                    train_loss = []
                    for r in mlflow.tracking.MlflowClient().get_metric_history(
                        id, "CVfold" + str(fold) + "_TRAIN_LOSS"
                    ):
                        train_loss.append(r.value)
                    folds_train_loss.append(train_loss)

                    train_acc = []
                    for r in mlflow.tracking.MlflowClient().get_metric_history(
                        id, "CVfold" + str(fold) + "_TRAIN_ACC"
                    ):
                        train_acc.append(r.value)
                    folds_train_acc.append(train_acc)

                    val_loss = []
                    for r in mlflow.tracking.MlflowClient().get_metric_history(
                        id, "CVfold" + str(fold) + "_VAL_LOSS"
                    ):
                        val_loss.append(r.value)
                    folds_val_loss.append(val_loss)

                    val_acc = []
                    for r in mlflow.tracking.MlflowClient().get_metric_history(
                        id, "CVfold" + str(fold) + "_VAL_ACC"
                    ):
                        val_acc.append(r.value)
                    folds_val_acc.append(val_acc)

                tmp_dict["TRAIN_LOSSES"] = folds_train_loss
                tmp_dict["TRAIN_ACCS"] = folds_train_acc
                tmp_dict["VAL_LOSSES"] = folds_val_loss
                tmp_dict["VAL_ACCS"] = folds_val_acc
                tmp_dict["TAG"] = tag
                sub_results[runs["experiment_id"][i] + "/" + str(id)] = tmp_dict
        results_dict[dataset] = sub_results
    return results_dict


def mlflow_to_resultsdict(
    datasets,
    epochs,
    methods,
    init_channels,
    pools,
    depths,
    dil_rates,
    enc_kernels,
    batch_sizes,
    lrs,
    segmentsize,
):
    results_dict = {}
    for dataset in datasets:
        sub_results = {}
        for epoch in epochs:
            for method in methods:
                for init_channel in init_channels:
                    for depth in depths:
                        for dil_rate in dil_rates:
                            for enc_kernel in enc_kernels:
                                for batch_size in batch_sizes:
                                    for lr in lrs:
                                        for pool in pools:
                                            dec_kernel = pool

                                            tag = {
                                                "DATASET": dataset,
                                                "METHOD": method,
                                                "INIT_CHANNELS": init_channel,
                                                "POOLS": pool,
                                                "DEPTH": depth,
                                                "DILATION_RATE": dil_rate,
                                                "ENCODER_KERNELSIZE": enc_kernel,
                                                "DECODER_KERNELSIZE": dec_kernel,
                                                "BATCHSIZE": batch_size,
                                                "EPOCHS": epoch,
                                                "LEARNING_RATE": lr,
                                            }

                                            exps = find_experiments_by_tags(tag)
                                            if len(exps) == 0:
                                                continue
                                            assert (
                                                len(exps) == 1
                                            ), "No experiment with this tag or two identical tags"

                                            runs = mlflow.search_runs(
                                                experiment_ids=[exps[0].experiment_id],
                                                run_view_type=ViewType.ACTIVE_ONLY,
                                            )

                                            tmp_dict = {}
                                            for i, id in enumerate(
                                                runs["run_id"].to_list()
                                            ):
                                                train_loss = []
                                                for (
                                                    r
                                                ) in mlflow.tracking.MlflowClient().get_metric_history(
                                                    id, "TRAIN_LOSS"
                                                ):
                                                    train_loss.append(r.value)

                                                train_acc = []
                                                for (
                                                    r
                                                ) in mlflow.tracking.MlflowClient().get_metric_history(
                                                    id, "TRAIN_ACC"
                                                ):
                                                    train_acc.append(r.value)

                                                val_loss = []
                                                for (
                                                    r
                                                ) in mlflow.tracking.MlflowClient().get_metric_history(
                                                    id, "VAL_LOSS"
                                                ):
                                                    val_loss.append(r.value)

                                                val_acc = []
                                                for (
                                                    r
                                                ) in mlflow.tracking.MlflowClient().get_metric_history(
                                                    id, "VAL_ACC"
                                                ):
                                                    val_acc.append(r.value)

                                                tmp_dict["TRAIN_LOSS"] = train_loss
                                                tmp_dict["TRAIN_ACC"] = train_acc
                                                tmp_dict["VAL_LOSS"] = val_loss
                                                tmp_dict["VAL_ACC"] = val_acc
                                                tmp_dict["TAG"] = tag

                                                sub_results[
                                                    runs["experiment_id"][i]
                                                    + "/"
                                                    + str(id)
                                                ] = tmp_dict

        results_dict[dataset] = sub_results
    return results_dict


class TestSetData(Dataset):
    def __init__(self, X, Y, X_padtoken=0, y_padtoken=10, maxlens=0, device="cpu"):
            
        X = [torch.tensor(x.astype(np.float32), device=device, dtype=torch.float32) for x in X]
        Y = [torch.tensor(y, device=device, dtype=torch.float32) for y in Y]
        self.X = [nn.ConstantPad1d((maxlens - len(x), 0), X_padtoken)(x.T) for x in X]
        y1 = [nn.ConstantPad1d((maxlens - len(y), 0), y_padtoken)(y) for y in Y]

        self.y = [
            nn.ConstantPad1d((maxlens - len(y), 0), y_padtoken)(
                y.permute(*torch.arange(y.ndim - 1, -1, -1))
            )
            for y in Y
        ]
        for i in range(len(self.y)):
            assert torch.all(torch.eq(self.y[i], y1[i])), "y not equal"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CustomDataset(Dataset):
    def __init__(self, X, Y, X_padtoken=0, y_padtoken=10, device="cpu"):
        maxlens = np.max([len(x) for x in X])
        X = [torch.tensor(x, device=device) for x in X]
        Y = [torch.tensor(y, device=device) for y in Y]
        # seems like pre-padding is smartest https://arxiv.org/abs/1903.07288,
        # but lstm should only see variable len input
        self.X = [
            nn.ConstantPad1d((maxlens - len(x), 0), X_padtoken)(
                x.permute(*torch.arange(x.ndim - 1, -1, -1))
            ).float()
            for x in X
        ]
        self.y = [
            nn.ConstantPad1d((maxlens - len(y), 0), y_padtoken)(
                y.permute(*torch.arange(y.ndim - 1, -1, -1))
            ).float()
            for y in Y
        ]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prep_tracks(df, xname="x", yname="y", identifiername="particle"):
    if type(df) == pd.core.frame.DataFrame:
        cols = (
            [xname, yname, "steplength", identifiername]
            if "steplength" in df.columns
            else [xname, yname, identifiername]
        )
        df = df[cols]
        df_by_particle = dict(tuple(df.groupby(identifiername)))
        D = [np.vstack(val[[xname, yname]].values) for val in df_by_particle.values()]
        X = [arr - arr[0] for arr in D]
    else:
        X = [arr - arr[0] for arr in df]
    return X


def center_to_origin(X):
    return [torch.from_numpy(arr - arr[0]).cpu() for arr in X]


def steplength_traces(data):
    """
    Computes the steplength in 2. norm for a trajectory (x,y)
    """

    def squaredist_full(x, y, z):
        return np.sqrt(
            (x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2 + (z[1:] - z[:-1]) ** 2
        )

    steplengths = []
    for trace in data:
        if trace.shape[1] == 2:
            x, y = trace[:, 0], trace[:, 1]
            z = np.zeros_like(x)
        if trace.shape[1] == 3:
            x, y, z = trace[:, 0], trace[:, 1], trace[:, 2]
        sl = squaredist_full(x, y, z)
        sl = np.append(0, sl)
        steplengths.append(sl.reshape(-1, 1))
    return np.array(steplengths, dtype=object)


def rollMSD(X, window=20):
    """
    Computes the mean squared displacement (msd) for a trajectory (x,y) up to
    window.
    """
    rolling_msds = []
    rolling_msd_ratios = []
    for trace in X:
        if trace.shape[1] == 2:
            x, y = trace[:, 0], trace[:, 1]
            z = np.zeros_like(x)
        if trace.shape[1] == 3:
            x, y, z = trace[:, 0], trace[:, 1], trace[:, 2]

        rolling_msd = []
        rolling_msd_ratio = []
        for j in range(len(x)):
            lower = max(0, j - window // 2)
            upper = min(len(x) - 1, j + window // 2)
            if window - (upper - lower):
                if lower == 0:
                    upper += window - (upper - lower)
                if upper == len(x) - 1:
                    lower -= window - (upper - lower)
            upper = min(len(x) - 1, upper)
            lower = max(0, lower)
            lag_msds = [
                squaredist(
                    x[lower:upper][:-lag],
                    x[lower:upper][lag:],
                    y[lower:upper][:-lag],
                    y[lower:upper][lag:],
                    z[lower:upper][:-lag],
                    z[lower:upper][lag:],
                )
                for lag in range(1, window)
            ]

            rolling_msd.append(np.mean(lag_msds))
            rolling_msd_ratio.append(
                np.mean(
                    [
                        lag_msds[i] / lag_msds[i + 1] - (i) / (i + 1)
                        for i in range(len(lag_msds) - 1)
                    ]
                )
            )

        rolling_msds.append(np.array(rolling_msd).reshape(-1, 1))
        rolling_msd_ratios.append(np.array(rolling_msd_ratio).reshape(-1, 1))
    return np.array(rolling_msds, dtype=object), np.array(
        rolling_msd_ratios, dtype=object
    )


def dotproduct_traces(data):
    """
    Computes the dotproduct for a trajectory (x,y)
    """
    dotproducts = []
    for trace in data:
        vecs = trace[1:] - trace[:-1]
        dots = np.dot(vecs[:-1], vecs[1:].T).diagonal()
        dots = np.append([0, 0], dots)
        dotproducts.append(dots.reshape(-1, 1))
    return np.array(dotproducts, dtype=object)


def rollEfficiency(X, window=20):
    """
    Computes the efficiency of a trajectory, logarithm of the ratio of squared end-to-end distance
    and the sum of squared distances.
    """

    rolling_effs = []
    for trace in X:
        if trace.shape[1] == 2:
            x, y = trace[:, 0], trace[:, 1]
            z = np.zeros_like(x)
        if trace.shape[1] == 3:
            x, y, z = trace[:, 0], trace[:, 1], trace[:, 2]

        rolling_eff = []
        for j in range(len(x)):
            lower = max(0, j - window // 2)
            upper = min(len(x) - 1, j + window // 2)
            if window - (upper - lower):
                if lower == 0:
                    upper += window - (upper - lower)
                if upper == len(x) - 1:
                    lower -= window - (upper - lower)
            upper = min(len(x) - 1, upper)
            lower = max(0, lower)
            top = squaredist(x[lower], x[upper], y[lower], y[upper], z[lower], z[upper])
            bottom = sum(
                [
                    squaredist(
                        x[lower + i],
                        x[lower + i + 1],
                        y[lower + i],
                        y[lower + i + 1],
                        z[lower + i],
                        z[lower + i + 1],
                    )
                    for i in range(min(len(x) - 1, window))
                ]
            )
            eff = np.sqrt((top) / ((window - 1) * bottom))
            rolling_eff.append(eff)
        rolling_effs.append(np.array(rolling_eff).reshape(-1, 1))
    return np.array(rolling_effs, dtype=object)


# def rollTrappedness(X, window=20):
#     """
#     Computes the efficiency of a trajectory, logarithm of the ratio of squared end-to-end distance
#     and the sum of squared distances.
#     """

#     rolling_trappednesses = []
#     for trace in X:
#         if trace.shape[1]==2:
#             x, y = trace[:,0], trace[:,1]
#             z = np.zeros_like(x)
#         if trace.shape[1]==3:
#             x, y, z = trace[:,0], trace[:,1], trace[:,2]

#         rolling_trappedness = []
#         for j in range(len(x)):
#             lower = max(0, j-window//2)
#             upper = min(len(x)-1, j+window//2)
#             if window-(upper-lower):
#                 if lower == 0:
#                     upper += window-(upper-lower)
#                 if upper == len(x)-1:
#                     lower -= window-(upper-lower)
#             upper = min(len(x)-1, upper)
#             lower = max(0, lower)
#             lag_msds = [squaredist(x[lower:upper][:-lag], x[lower:upper][lag:],
#                                    y[lower:upper][:-lag], y[lower:upper][lag:],
#                                    z[lower:upper][:-lag], z[lower:upper][lag:])
#                                         for lag in range(1, window)]
#             maxpair = GetMax(x, y, z)
#             r0 = np.sqrt(maxpair) / 2
#             D = lag_msds[1] - lag_msds[0]
#             trappedness = 1 - np.exp(0.2045 - 0.25117 * (D * len(x)) / r0 ** 2)

#             rolling_trappedness.append(trappedness)
#         rolling_trappednesses.append(np.array(rolling_trappedness).reshape(-1,1))
#     return np.array(rolling_trappednesses, dtype=object)


def GetMax(x, y, z):
    """Computes the maximum squared distance between all points in the (x,y,z) set.

    Parameters
    ----------
    x : list-like
        x-coordinates.
    y : list-like
        y-coordinates.
    z : list-like
        z-coordinates.

    Returns
    -------
    float
        Largest squared distance between any two points in the set.

    """
    from itertools import combinations
    from random import randint

    A = np.array([x, y, z]).T

    def square_distance(x, y):
        return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)])

    max_square_distance = 0
    for pair in combinations(A, 2):
        if square_distance(*pair) > max_square_distance:
            max_square_distance = square_distance(*pair)
            max_pair = pair
    return max_square_distance


def squaredist(x0, x1, y0, y1, z0, z1):
    return np.mean((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)


def combine_to_features(f1, f2):
    features = []
    for i in range(len(f1)):
        features.append(np.hstack([f1[i], f2[i]]))
    return features


def add_features(X, features_list: list):
    dim = X[0].shape[1]
    out = X
    if "SL" in features_list:
        steplengths = steplength_traces(X)
        out = [np.hstack([out[i], steplengths[i]]) for i in range(len(X))]
    if "DP" in features_list:
        dotproduct = dotproduct_traces(X)
        out = [np.hstack([out[i], dotproduct[i]]) for i in range(len(X))]
    if "MSD" in features_list:
        MSD, MSD_ratio = rollMSD(X)
        out = [np.column_stack([out[i], MSD[i]]) for i in range(len(X))]
    if "EFF" in features_list:
        EFF = rollEfficiency(X)
        out = [np.column_stack([out[i], EFF[i]]) for i in range(len(X))]
    # if 'TRAP' in features_list:
    #     TRAP = rollTrappedness(X)
    #     out = [np.hstack([out[i], TRAP[i]]) for i in range(len(X))]
    if "XYZ" not in features_list:
        out = [x[:, dim:] for x in out]

    return out


def labelsegment(l):
    uniq, idx, inv, cnts = np.unique(
        l, return_index=True, return_inverse=True, return_counts=True
    )
    # draw between two uniques:
    if len(np.unique(cnts)) == 1:
        out = l[-1]
    else:
        ids = []
        for i in range(len(cnts)):
            if cnts[i] == max(cnts):
                ids.append(i)
        if len(ids) == 1:
            out = uniq[ids[0]]
        else:
            idx_of_max_cnt = [idx[i] for i in ids]
            out = uniq[np.argmax(idx_of_max_cnt)]
    return out


def segment_y(y, segmentsize):
    """
    Segment 'y' into 'segmentsize' segments.
    label is given as major class in segment
    if its a draw the label for highest timestamp wins.
    eg. [0,1,1]->1, [0,1]->1, [0,1,1,0]->0
    """
    if segmentsize == 1:
        return y
    else:
        y_segmented = []
        for ys in y:
            y_segmented.append(
                [
                    labelsegment(ys[i : i + segmentsize])
                    for i in range(0, len(ys), segmentsize)
                ]
            )
        return y_segmented


def Chi2Fit(
    x,
    y,
    sy,
    f,
    plot=True,
    print_level=0,
    labels=None,
    ax=None,
    savefig=None,
    valpos=None,
    exponential=False,
    fitcol=None,
    markersize=5,
    plotcol=None,
    name=None,
    fontsize=15,
    linewidth=3,
    png=False,
    custom_cost=None,
    **guesses,
):
    """Function that peforms a Chi2Fit to data given function
    ----------
    Parameters
    ----------
    x: ndarray of shape for input in f
        - input values to fit
    y: ndarray of shape output from f
        - output values to fit
    sy: ndarray of length y
        - errors on the y values
    f: function
        - Function to fit, should be of form f(x,args), where args
          is a list of arguments
    **guesses: mappings ie. p0=0.1,p1=0.2
        - initial guesses for the fit parameters
    print_level: int 0,1
        - Wether to print output from chi2 ect.
    labels:
        - Mappable to pass to ax.set call to set labels on plot
    name: str
        -Label to call fit in legend
    fontsize: int
        - Size of font in plot
    linewidth: float
        - Width of line on data
    ---------
    Returns
    ---------
    params: length args
        - fit params
    errs: lenght args
        - errror on fit params
    Ndof: int
        - Number of  degrees of freedom for fit
    Chi2: float
        - Chi2 for fit
    pval: float
        -pvalue for the fit
    """
    xmin, xmax = np.min(x), np.max(x)
    names = inspect.getfullargspec(f)[0][1:]
    if custom_cost is None:
        chi2_object = LeastSquares(x, y, sy, f)
    else:
        chi2_object = custom_cost
    if len(guesses) != 0:
        paramguesses = {}
        lims = {}
        for key, value in guesses.items():
            if key.split("_")[0] == "limit":
                lims[key.split("_")[1]] = value
            else:
                paramguesses[key] = value
        minuit = Minuit(chi2_object, **paramguesses)
        if len(lims) > 0:
            for key, value in lims.items():
                minuit.limits[key] = value
        minuit.print_level = print_level
    else:
        minuit = Minuit(chi2_object)
    minuit.migrad()
    chi2 = minuit.fval
    # free parameters, not len(guesses): `guesses` also carries the
    # limit_<name> entries, which are bounds rather than fitted parameters
    Ndof = len(x) - minuit.nfit
    Pval = stats.chi2.sf(chi2, Ndof)
    params = minuit.values
    errs = minuit.errors

    if not exponential:
        dict = {"chi2": chi2, "Ndof": Ndof, "Pval": Pval}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.2f} +/- {py:4.2f}"
    else:
        dict = {"chi2": f"{chi2:4.4E}", "Ndof": f"{Ndof:4.4E}", "Pval": f"{Pval:4.4E}"}
        for n, p, py in zip(names, params, errs):
            dict[n] = f"{p:4.4E} +/- {py:4.4E}"
    return params, errs, Pval


def Scalings(msds):
    """Fit mean squared displacements to a power law.

    Parameters
    ----------
    msds : list-like
        mean squared displacenemts.

    Returns
    -------
    tuple of length 3
        The first index is the fitted generalized diffusion constant,
        the second is the scaling exponent alpha, and the final is the pvalue for the fit.

    """

    def power(x, D, alpha):
        return 4 * D * (x) ** alpha

    params, errs, Pval = Chi2Fit(
        np.arange(1, len(msds) + 1),
        msds,
        np.ones(len(msds)),
        power,
        plot=False,
        D=1,
        alpha=1,
        limit_alpha=(-10, 10),
    )
    sy = np.std(msds - power(np.arange(1, len(msds) + 1), *params))
    params, errs, Pval = Chi2Fit(
        np.arange(1, len(msds) + 1),
        msds,
        sy * np.ones(len(msds)),
        power,
        plot=False,
        D=1,
        alpha=1,
        limit_alpha=(-10, 10),
    )
    return params[0], params[1], Pval


def datasplitter(D, val_size, test_size, seed):
    lengths = [int(round(len(D) * (1 - test_size))), int(round(len(D) * (test_size)))]
    D_train, D_test = random_split(
        D, lengths, generator=torch.Generator().manual_seed(seed)
    )

    val_size = val_size / (1 - test_size)
    lengths = [
        int(round(len(D_train) * (1 - val_size))),
        int(round(len(D_train) * (val_size))),
    ]
    D_train, D_val = random_split(
        D_train, lengths, generator=torch.Generator().manual_seed(seed)
    )

    return D_train, D_val, D_test


def remove_short_traces3D(df, threshold=30):
    particles, counts = np.unique(df["id"], return_counts=True)
    keep_particles = [
        particles[i] for i in range(len(particles)) if counts[i] >= threshold
    ]
    return df[df["id"].isin(keep_particles)], keep_particles


def remove_dim_traces3D(df, threshold=40):
    traces = np.unique(df["id"])
    keep_particles = [
        idx for idx in traces if df[df["id"] == idx]["A_ch1"][0] >= threshold
    ]
    return df[df["id"].isin(keep_particles)], keep_particles


def remove_overlybright_traces3D(df, threshold=1000):
    traces = np.unique(df["id"])
    keep_particles = [
        idx for idx in traces if np.max(df[df["id"] == idx]["A_ch1"]) <= threshold
    ]
    return df[df["id"].isin(keep_particles)], keep_particles


def remove_irregular_sampling(times, keep_std_threshold=0.001):
    keep_idx = []
    for i in range(len(times)):
        sd = np.std((times[i][1:] - times[i][0]) - (times[i][:-1] - times[i][0]))
        if sd < keep_std_threshold:
            keep_idx.append(i)
    return keep_idx


def convert_matlabDF_to_arrays(dfg, xy_to_um, z_to_um):
    tracks = []
    timepoints = []
    frames = []
    amplitudes = []
    amplitudes_S = []
    amplitudes_SM = []
    amplitudes_sig = []
    amplitudes_bg = []
    catIdx = []
    track_ids = []
    for _, group in dfg:
        x = group["x_ch1"].values * xy_to_um
        y = group["y_ch1"].values * xy_to_um
        z = group["z_ch1"].values * z_to_um
        xy = np.column_stack([x, y])
        tracks.append(np.column_stack([xy, z]))
        timepoints.append(group["t_in_seconds"].values)
        frames.append(group["frames"].values)
        A12 = np.column_stack([group["A_ch1"].values, group["A_ch2"].values])
        amplitudes.append(np.column_stack([A12, group["A_ch3"].values]))

        AS12 = np.column_stack([group["AS_ch1"].values, group["AS_ch2"].values])
        amplitudes_S.append(np.column_stack([AS12, group["AS_ch3"].values]))

        ASM12 = np.column_stack([group["ASM_ch1"].values, group["ASM_ch2"].values])
        amplitudes_SM.append(np.column_stack([ASM12, group["ASM_ch3"].values]))

        A12_sig = np.column_stack([group["A_sig1"].values, group["A_sig2"].values])
        amplitudes_sig.append(np.column_stack([A12_sig, group["A_sig3"].values]))
        A12_bg = np.column_stack([group["A_bg1"].values, group["A_bg2"].values])
        amplitudes_bg.append(np.column_stack([A12_bg, group["A_bg3"].values]))
        catIdx.append(np.unique(group["catIdx"].values))
        track_ids.append(np.unique(group["id"].values))
    return (
        tracks,
        timepoints,
        frames,
        list(np.hstack(track_ids)),
        amplitudes,
        amplitudes_S,
        amplitudes_SM,
        amplitudes_sig,
        amplitudes_bg,
        np.hstack(catIdx),
    )


def keepidx_func(l, idx):
    return [l[i] for i in range(len(l)) if i in idx]


def load_or_create_resultsdict_for_rawdata(
    PROJECT_NAMES,
    SEARCH_PATTERN,
    OUTPUT_NAME,
    globals,
    datapath,
    save_dict_name,
    save_path,
    best_models_sorted,
    the_data_is="2D",
    modelpath="Unet_results/mlruns/",
    xy_to_um=0.104,
    z_to_um=0.25,
    min_trace_length=20,
    min_brightness=40,
    features=["XYZ", "SL", "DP"],
    device="cpu",
):
    if the_data_is == "3D":
        if os.path.exists(save_dict_name):
            results_dict = pickle.load(open(save_dict_name, "rb"))
        else:
            results_dict = {}
            for file in PROJECT_NAMES:
                if os.path.exists(OUTPUT_NAME.format(file)):
                    df = pickle.load(
                        open("{}/processed_tracks_df.pkl".format(file), "rb")
                    )
                else:
                    df = matlabloader(
                        name=file, input=SEARCH_PATTERN, output=OUTPUT_NAME
                    )

                df = remove_short_traces3D(df, threshold=min_trace_length)
                df = remove_dim_traces3D(df, threshold=min_brightness)

                # turn matlab df into useful numpy arrays
                dfg = df.groupby("id")
                (
                    tracks,
                    times,
                    track_ids,
                    videos,
                    amplitudes,
                    amplitudes_sig,
                    amplitudes_bg,
                ) = convert_matlabDF_to_arrays(dfg, xy_to_um, z_to_um)
                tracks = np.array(tracks, dtype=object)

                X = prep_tracks(tracks)

                X_to_eval = add_features(X, features)
                y_to_eval = [np.ones(len(x)) * 0.5 for x in X_to_eval]

                files_dict = {}
                for modelname in best_models_sorted:
                    model = load_UnetModels(modelname, dir=modelpath, device=device)
                    tmp_dict = make_preds(
                        model,
                        X_to_eval,
                        y_to_eval,
                        min_max_len=601,
                        X_padtoken=globals.X_padtoken,
                        y_padtoken=globals.y_padtoken,
                        batch_size=globals.batch_size,
                        device=device,
                    )
                    files_dict[modelname] = tmp_dict

                if len(list(files_dict.keys())) == 3:
                    files_dict["ensemble_score"] = ensemble_scoring(
                        files_dict[list(files_dict.keys())[0]]["masked_score"],
                        files_dict[list(files_dict.keys())[1]]["masked_score"],
                        files_dict[list(files_dict.keys())[2]]["masked_score"],
                    )
                    ensemble_pred = [
                        np.argmax(files_dict["ensemble_score"][i], axis=0)
                        for i in range(len(files_dict["ensemble_score"]))
                    ]
                    files_dict["ensemble"] = ensemble_pred

                files_dict["X_to_eval"] = X_to_eval
                files_dict["y_to_eval"] = y_to_eval
                files_dict["amplitudes"] = amplitudes
                files_dict["amplitudes_sig"] = amplitudes_sig
                files_dict["amplitudes_bg"] = amplitudes_bg
                files_dict["modelnames"] = best_models_sorted
                files_dict["info_dict"] = {
                    "times": times,
                    "track_ids": track_ids,
                    "video": videos,
                }
                results_dict[file] = files_dict

            pickle.dump(results_dict, open(save_dict_name, "wb"))

        return results_dict

    if the_data_is == "2D":
        if os.path.exists(save_path + save_dict_name):
            results_dict = pickle.load(open(save_path + save_dict_name, "rb"))
            assert len(PROJECT_NAMES) == 1, "PROJECT_NAMES can only be length 1"
            X_to_eval = load_X(datapath, PROJECT_NAMES[0], features=["XYZ", "SL", "DP"])
            y_to_eval = [np.ones(len(x)) * 0.5 for x in X_to_eval]
        else:
            results_dict = {}
            for file in PROJECT_NAMES:
                X_to_eval = load_X(datapath, file, features=["XYZ", "SL", "DP"])
                y_to_eval = [np.ones(len(x)) * 0.5 for x in X_to_eval]
                files_dict = {}
                for modelname in best_models_sorted:
                    model = load_UnetModels(modelname, dir=modelpath, device=device)
                    tmp_dict = make_preds(
                        model,
                        X_to_eval,
                        y_to_eval,
                        min_max_len=601,
                        X_padtoken=globals.X_padtoken,
                        y_padtoken=globals.y_padtoken,
                        batch_size=globals.batch_size,
                        device=device,
                    )
                    files_dict[modelname] = tmp_dict
                if len(list(files_dict.keys())) > 1:
                    files_dict["ensemble_score"] = ensemble_scoring(
                        files_dict[list(files_dict.keys())[0]]["masked_score"],
                        files_dict[list(files_dict.keys())[1]]["masked_score"],
                        files_dict[list(files_dict.keys())[2]]["masked_score"],
                    )
                    ensemble_pred = [
                        np.argmax(files_dict["ensemble_score"][i], axis=0)
                        for i in range(len(files_dict["ensemble_score"]))
                    ]
                    files_dict["ensemble"] = ensemble_pred

                files_dict["modelnames"] = best_models_sorted
                results_dict[file] = files_dict

            pickle.dump(results_dict, open(save_path + save_dict_name, "wb"))

        return results_dict, X_to_eval, y_to_eval, "put_times_here", "put_tracks_here"


def index(m, track_id, column):
    """
    Indexing in the matlab h5 file returns a reference only. This reference is
    then used to go back and find the values in the file.
    """
    ref = m[column][track_id][0]

    return np.array(m[ref][:])


# def extract_matlab_files_from_zip(name):
#     archive = ZipFile(name+'.zip','r')
#     member = []
#     for fp in archive.namelist():
#         if 'ProcessedTracks.mat' in fp:
#             member.append(fp)
#     ZipFile.extract(member, path='extracted', pwd=None)


def matlabloader(name, input, output, dont_use=[]):
    _input = input.format(name)
    _output = output.format(name)

    files = sorted(glob(_input, recursive=True))
    df = pd.concat([cme_tracks_to_pandas(f, project_name=name)[0] for f in files])
    seqOfEvents_list = [cme_tracks_to_pandas(f, project_name=name)[1] for f in files][0]
    original_idx = [cme_tracks_to_pandas(f, project_name=name)[2] for f in files][0]
    pickle.dump(df, open(_output, "wb"))
    return df, seqOfEvents_list, original_idx


def cme_tracks_to_pandas(mat_path, project_name):
    """
    Converts CME-derived ProcessedTracks.mat to Pandas DataFrame format.

    This version was specifically created for matlab 7.3 files.

    Add extra columns as required.
    """
    COLUMNS = (
        "A",
        "AS",
        "ASM",
        "x",
        "y",
        "z",
        "c",
        "sigma_r",
        "t",
        "f",
        "catIdx",
        "nSeg",
        "seqOfEvents",
    )

    import scipy.io

    try:
        scifile = scipy.io.loadmat(mat_path)
        m = scifile["tracks"]
        n_tracks = len(m["A"][0])

        df = []
        seqOfEvents_list = []
        original_idx = []
        for i in range(n_tracks):
            A = m["A"][0][i]
            try:
                AS = m["AS"][0][i]
                ASM = m["ASM"][0][i]
            except:
                AS = np.zeros(A.shape)
                ASM = np.zeros(A.shape)

            sig = m["sigma_r"][0][i]

            x = m["x"][0][i]
            y = m["y"][0][i]
            z = m["z"][0][i]

            seqOfEvents = m["seqOfEvents"][0][i]
            seqOfEvents_list.append(seqOfEvents)

            c = m["c"][0][i]
            t_in_seconds = m["t"][0][i].flatten()
            frames = m["f"][0][i].flatten()
            catIdx = m["catIdx"][0][i].flatten()
            nSeg = m["nSeg"][0][i]
            catIdx = np.repeat(catIdx, A.shape[1])
            nSeg = np.repeat(nSeg, A.shape[1])

            if A.shape[0] == 3:
                A_ch1, A_ch2, A_ch3 = A[0, :], A[1, :], A[2, :]
                AS_ch1, AS_ch2, AS_ch3 = AS[0, :], AS[1, :], AS[2, :]
                ASM_ch1, ASM_ch2, ASM_ch3 = ASM[0, :], ASM[1, :], ASM[2, :]
                A_sig1, A_sig2, A_sig3 = sig[0, :], sig[1, :], sig[2, :]
                x_ch1, x_ch2, x_ch3 = x[0, :], x[1, :], x[2, :]
                y_ch1, y_ch2, y_ch3 = y[0, :], y[1, :], y[2, :]
                z_ch1, z_ch2, z_ch3 = z[0, :], z[1, :], z[2, :]
                A_bg1, A_bg2, A_bg3 = c[0, :], c[1, :], c[2, :]
            if A.shape[0] == 2:
                A_ch1, A_ch2, A_ch3 = A[0, :], A[1, :], np.zeros_like(A[1, :])
                AS_ch1, AS_ch2, AS_ch3 = AS[0, :], AS[1, :], np.zeros_like(AS[1, :])
                ASM_ch1, ASM_ch2, ASM_ch3 = (
                    ASM[0, :],
                    ASM[1, :],
                    np.zeros_like(ASM[1, :]),
                )
                A_sig1, A_sig2, A_sig3 = sig[0, :], sig[1, :], np.zeros_like(A[1, :])
                x_ch1, x_ch2, x_ch3 = x[0, :], x[1, :], np.zeros_like(A[1, :])
                y_ch1, y_ch2, y_ch3 = y[0, :], y[1, :], np.zeros_like(A[1, :])
                z_ch1, z_ch2, z_ch3 = z[0, :], z[1, :], np.zeros_like(A[1, :])
                A_bg1, A_bg2, A_bg3 = c[0, :], c[1, :], np.zeros_like(A[1, :])
            if A.shape[0] == 1:
                A_ch1, A_ch2, A_ch3 = (
                    A[0, :],
                    np.zeros_like(A[0, :]),
                    np.zeros_like(A[0, :]),
                )
                AS_ch1, AS_ch2, AS_ch3 = (
                    AS[0, :],
                    np.zeros_like(AS[0, :]),
                    np.zeros_like(AS[0, :]),
                )
                ASM_ch1, ASM_ch2, ASM_ch3 = (
                    ASM[0, :],
                    np.zeros_like(ASM[0, :]),
                    np.zeros_like(ASM[0, :]),
                )
                A_sig1, A_sig2, A_sig3 = (
                    sig[0, :],
                    np.zeros_like(sig[0, :]),
                    np.zeros_like(sig[0, :]),
                )
                x_ch1, x_ch2, x_ch3 = (
                    x[0, :],
                    np.zeros_like(x[0, :]),
                    np.zeros_like(x[0, :]),
                )
                y_ch1, y_ch2, y_ch3 = (
                    y[0, :],
                    np.zeros_like(y[0, :]),
                    np.zeros_like(y[0, :]),
                )
                z_ch1, z_ch2, z_ch3 = (
                    z[0, :],
                    np.zeros_like(z[0, :]),
                    np.zeros_like(z[0, :]),
                )
                A_bg1, A_bg2, A_bg3 = (
                    c[0, :],
                    np.zeros_like(c[0, :]),
                    np.zeros_like(c[0, :]),
                )

            track_len = len(nSeg)
            # # Find out where parent dirs can be skipped

            # # Create path from actual directory

            try:
                real_dir = re.search(string=mat_path, pattern=project_name)
                end_dir = re.search(string=mat_path, pattern="Analysis")
                filepath = mat_path[real_dir.end() : end_dir.start() - 1]
            except:
                real_dir = re.search(string=mat_path, pattern=project_name)
                end_dir = re.search(string=mat_path, pattern="")
                filepath = mat_path[real_dir.end() : end_dir.start() - 1]

            group = pd.DataFrame(
                {
                    "file": np.repeat(filepath, track_len),
                    "particle": np.repeat(i, track_len),
                    "id": np.repeat("p" + str(i + 1) + "_" + filepath, track_len),
                    "t_in_seconds": t_in_seconds,
                    "frames": frames,
                    "A_ch1": A_ch1,
                    "A_ch2": A_ch2,
                    "A_ch3": A_ch3,
                    "AS_ch1": AS_ch1,
                    "AS_ch2": AS_ch2,
                    "AS_ch3": AS_ch3,
                    "ASM_ch1": ASM_ch1,
                    "ASM_ch2": ASM_ch2,
                    "ASM_ch3": ASM_ch3,
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
                    "A_bg3": A_bg3,
                    "catIdx": catIdx,
                    "nSeg": nSeg,
                }
            )
            group.fillna(np.nan)
            df.append(group)
            original_idx.append("p" + str(i + 1) + "_" + filepath)
    except:
        h5file = h5py.File(mat_path, "r")
        m = h5file["tracks"]
        n_tracks = len(m["A"])
        df = []
        seqOfEvents_list = []
        original_idx = []
        for i in range(n_tracks):
            # # Extract columns
            try:
                (
                    A,
                    AS,
                    ASM,
                    x,
                    y,
                    z,
                    c,
                    sig,
                    t_in_seconds,
                    frames,
                    catIdx,
                    nSeg,
                    seqOfEvents,
                ) = [index(m=m, track_id=i, column=c) for c in COLUMNS]
            except:
                COLUMNS = (
                    "A",
                    "x",
                    "y",
                    "z",
                    "c",
                    "sigma_r",
                    "t",
                    "f",
                    "catIdx",
                    "nSeg",
                    "seqOfEvents",
                )
                A, x, y, z, c, sig, t_in_seconds, frames, catIdx, nSeg, seqOfEvents = [
                    index(m=m, track_id=i, column=c) for c in COLUMNS
                ]
                AS, ASM = np.zeros_like(A), np.zeros_like(A)

            catIdx = np.repeat(catIdx[0], A.shape[0])
            nSeg = np.repeat(nSeg[0], A.shape[0])
            if A.shape[1] == 3:
                A_ch1, A_ch2, A_ch3 = A[:, 0], A[:, 1], A[:, 2]
                AS_ch1, AS_ch2, AS_ch3 = AS[:, 0], AS[:, 1], AS[:, 2]
                ASM_ch1, ASM_ch2, ASM_ch3 = ASM[:, 0], ASM[:, 1], ASM[:, 2]
                A_sig1, A_sig2, A_sig3 = sig[:, 0], sig[:, 1], sig[:, 2]
                x_ch1, x_ch2, x_ch3 = x[:, 0], x[:, 1], x[:, 2]
                y_ch1, y_ch2, y_ch3 = y[:, 0], y[:, 1], y[:, 2]
                z_ch1, z_ch2, z_ch3 = z[:, 0], z[:, 1], z[:, 2]
                A_bg1, A_bg2, A_bg3 = c[:, 0], c[:, 1], c[:, 2]
            if A.shape[1] == 2:
                A_ch1, A_ch2, A_ch3 = A[:, 0], A[:, 1], 0
                AS_ch1, AS_ch2, AS_ch3 = AS[:, 0], AS[:, 1], np.zeros_like(AS[:, 1])
                ASM_ch1, ASM_ch2, ASM_ch3 = (
                    ASM[:, 0],
                    ASM[:, 1],
                    np.zeros_like(ASM[:, 1]),
                )
                A_sig1, A_sig2, A_sig3 = sig[:, 0], sig[:, 1], 0
                x_ch1, x_ch2, x_ch3 = x[:, 0], x[:, 1], 0
                y_ch1, y_ch2, y_ch3 = y[:, 0], y[:, 1], 0
                z_ch1, z_ch2, z_ch3 = z[:, 0], z[:, 1], 0
                A_bg1, A_bg2, A_bg3 = c[:, 0], c[:, 1], 0
            if A.shape[1] == 1:
                A_ch1, A_ch2, A_ch3 = (
                    A[:, 0],
                    np.zeros_like(A[:, 0]),
                    np.zeros_like(A[:, 0]),
                )
                AS_ch1, AS_ch2, AS_ch3 = (
                    AS[:, 0],
                    np.zeros_like(AS[:, 0]),
                    np.zeros_like(AS[:, 0]),
                )
                ASM_ch1, ASM_ch2, ASM_ch3 = (
                    ASM[:, 0],
                    np.zeros_like(ASM[:, 0]),
                    np.zeros_like(ASM[:, 0]),
                )
                A_sig1, A_sig2, A_sig3 = (
                    sig[:, 0],
                    np.zeros_like(sig[:, 0]),
                    np.zeros_like(sig[:, 0]),
                )
                x_ch1, x_ch2, x_ch3 = (
                    x[:, 0],
                    np.zeros_like(x[:, 0]),
                    np.zeros_like(x[:, 0]),
                )
                y_ch1, y_ch2, y_ch3 = (
                    y[:, 0],
                    np.zeros_like(y[:, 0]),
                    np.zeros_like(y[:, 0]),
                )
                z_ch1, z_ch2, z_ch3 = (
                    z[:, 0],
                    np.zeros_like(z[:, 0]),
                    np.zeros_like(z[:, 0]),
                )
                A_bg1, A_bg2, A_bg3 = (
                    c[:, 0],
                    np.zeros_like(c[:, 0]),
                    np.zeros_like(c[:, 0]),
                )

            t_in_seconds = t_in_seconds.flatten()
            frames = frames.flatten()

            track_len = len(A)
            # # Find out where parent dirs can be skipped
            real_dir = re.search(string=mat_path, pattern=project_name)
            end_dir = re.search(string=mat_path, pattern=project_name.split("/")[-1])

            # # Create path from actual directory
            filepath = mat_path[real_dir.end() : end_dir.start() - 1]

            group = pd.DataFrame(
                {
                    "file": np.repeat(filepath, track_len),
                    "particle": np.repeat(i, track_len),
                    "id": np.repeat("p" + str(i + 1) + "_" + filepath, track_len),
                    "t_in_seconds": t_in_seconds,
                    "frames": frames,
                    "A_ch1": A_ch1,
                    "A_ch2": A_ch2,
                    "A_ch3": A_ch3,
                    "AS_ch1": AS_ch1,
                    "AS_ch2": AS_ch2,
                    "AS_ch3": AS_ch3,
                    "ASM_ch1": ASM_ch1,
                    "ASM_ch2": ASM_ch2,
                    "ASM_ch3": ASM_ch3,
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
                    "A_bg3": A_bg3,
                    "catIdx": catIdx,
                    "nSeg": nSeg,
                }
            )
            group.fillna(np.nan)
            df.append(group)
            seqOfEvents_list.append(seqOfEvents)
            original_idx.append("p" + str(i + 1) + "_" + filepath)

    return pd.concat(df), seqOfEvents_list, original_idx


def load_or_pred_for_simdata(
    X_to_eval,
    y_to_eval,
    globals,
    savepath,
    savename,
    best_models_sorted,
    filenames_X,
    filenames_y,
    min_max_len=601,
    use_mlflow=False,
    device="cuda",
    modelpath="Unet_results/mlruns/",
    dim=2,
):
    if False:  # os.path.exists(savepath+savename):
        results_dict = pickle.load(open(savepath + savename, "rb"))
    else:
        results_dict = {}
        for modelname in best_models_sorted:
            if use_mlflow:
                model = load_UnetModels(
                    modelname, dir=modelpath, device=device, dim=dim
                )
            else:
                model = load_UnetModels_directly(modelname, device=device, dim=dim)
            import time

            starttime = time.time()
            tmp_dict = make_preds(
                model,
                X_to_eval,
                y_to_eval,
                min_max_len=min_max_len,
                X_padtoken=globals.X_padtoken,
                y_padtoken=globals.y_padtoken,
                batch_size=globals.batch_size,
            )
            print("time per track", (time.time() - starttime) / len(X_to_eval))
            results_dict[modelname] = tmp_dict

        if len(best_models_sorted) > 1:
            ensemble_score = ensemble_scoring(
                results_dict[best_models_sorted[0]]["masked_score"],
                results_dict[best_models_sorted[1]]["masked_score"],
                results_dict[best_models_sorted[2]]["masked_score"],
            )
            results_dict["ensemble_score"] = ensemble_score

            ensemble_pred = [
                np.argmax(ensemble_score[i], axis=0) for i in range(len(ensemble_score))
            ]
            results_dict["ensemble"] = ensemble_pred

        results_dict["models"] = best_models_sorted
        results_dict["filenames_X"] = filenames_X
        results_dict["filenames_y"] = filenames_y

        print(
            np.mean(
                np.hstack(results_dict[best_models_sorted[0]]["masked_pred"])
                == np.hstack(y_to_eval)
            )
        )
        print(
            np.mean(
                np.hstack(results_dict[best_models_sorted[1]]["masked_pred"])
                == np.hstack(y_to_eval)
            )
        )
        print(
            np.mean(
                np.hstack(results_dict[best_models_sorted[2]]["masked_pred"])
                == np.hstack(y_to_eval)
            )
        )
        print(np.mean(np.hstack(ensemble_pred) == np.hstack(y_to_eval)))

        pickle.dump(results_dict, open(savepath + savename, "wb"))
    return results_dict


def global_transition_probs(pred_argmax):
    """
    returns dicts with the transistions

    output[0] = transistions normalized by the difftype, e.g. all transistions
    from normal are normalized by the amount of transistions from normal to the others
    output[1] = the counts of transistions
    output[2] = the transistions normalized to the global number of transistions
    """
    flat_pred_argmax = [p for subpred in pred_argmax for p in subpred]
    run_lens, run_pos, run_difftype = find_segments(flat_pred_argmax)
    before = run_difftype[:-1]
    after = run_difftype[1:]

    transistions = [str(b) + str(a) for b, a in zip(before, after)]
    uniq_transistions, count_per_transistions = np.unique(
        transistions, return_counts=True
    )

    trans_dict = {u: p for u, p in zip(uniq_transistions, count_per_transistions)}
    globalnormed_trans_dict = {
        u: p
        for u, p in zip(
            uniq_transistions, count_per_transistions / sum(count_per_transistions)
        )
    }

    possible_difftypes = 4
    counts_per_difftype = np.zeros(possible_difftypes)
    for tr in uniq_transistions:
        if tr[0] == "0":
            counts_per_difftype[0] += trans_dict[tr]
        if tr[0] == "1":
            counts_per_difftype[1] += trans_dict[tr]
        if tr[0] == "2":
            counts_per_difftype[2] += trans_dict[tr]
        if tr[0] == "3":
            counts_per_difftype[3] += trans_dict[tr]

    norm_trans_dict = {u: p for u, p in zip(uniq_transistions, count_per_transistions)}
    for key in trans_dict.keys():
        if key[0] == "0":
            norm_trans_dict[key] /= counts_per_difftype[int(key[0])]
        if key[0] == "1":
            norm_trans_dict[key] /= counts_per_difftype[int(key[0])]
        if key[0] == "2":
            norm_trans_dict[key] /= counts_per_difftype[int(key[0])]
        if key[0] == "3":
            norm_trans_dict[key] /= counts_per_difftype[int(key[0])]
    return norm_trans_dict, trans_dict, globalnormed_trans_dict


def lifetime_calculator(channels, subpaths, best_models_sorted, device="cpu"):
    lifetime_list = []
    for channel in channels:
        for subpath in subpaths:
            # Prep data
            the_data_is = "2D"
            datapath = "Insulin/"
            bio_repli = 0
            datapath = datapath + subpath
            save_dict_name = (
                datapath.replace("/", "_")
                + str(bio_repli)
                + "_"
                + channel
                + "_results_dict_mldir"
                + best_models_sorted[0].split("/")[0]
                + ".pkl"
            )
            save_path = "Insulin/predictions/"
            SEARCH_PATTERN = "not_used"
            OUTPUT_NAME = "not_used"
            print(save_path + save_dict_name)
            PROJECT_NAMES = [channel + "/bg_corr_all_tracked" + str(bio_repli) + ".csv"]
            file = channel + "/bg_corr_all_tracked" + str(bio_repli) + ".csv"
            xy_to_um = 1
            z_to_um = 1

            exp_name = (
                subpath.replace("/", "_")
                + PROJECT_NAMES[0].split("/")[0]
                + "_"
                + str(bio_repli)
            )
            results_dict, X_to_eval, y_to_eval, _, _ = (
                load_or_create_resultsdict_for_rawdata(
                    PROJECT_NAMES,
                    SEARCH_PATTERN,
                    OUTPUT_NAME,
                    globals,
                    datapath,
                    save_dict_name,
                    save_path,
                    best_models_sorted,
                    the_data_is,
                    xy_to_um=xy_to_um,
                    z_to_um=z_to_um,
                    device=device,
                )
            )
            if "ensemble" in (list(results_dict[file].keys())):
                ens_pred = results_dict[file]["ensemble"]
                ens_score = results_dict[file]["ensemble_score"]
            else:
                m1_pred = results_dict[file][list(results_dict[file].keys())[0]][
                    "masked_pred"
                ]
                m1_score = results_dict[file][list(results_dict[file].keys())[0]][
                    "masked_score"
                ]

            difftypes = ["Normal ", "Directed ", "Confined ", "Subdiffusive "]
            color_list = [
                sns.color_palette("colorblind")[1],
                sns.color_palette("dark")[3],
                sns.color_palette("colorblind")[-1],
                sns.color_palette("dark")[0],
            ]
            flat_pred_argmax = [p for subpred in ens_pred for p in subpred]

            tau = np.unique(flat_pred_argmax, return_counts=True)[1] / sum(
                np.unique(flat_pred_argmax, return_counts=True)[1]
            )
            difftype = np.unique(flat_pred_argmax, return_counts=True)[0]

            lifetime = np.zeros(len(difftypes))
            for i in range(len(tau)):
                lifetime[difftype[i]] = tau[i]

            lifetime_list.append(lifetime)
    return lifetime_list


def transistion_array_calculator(channels, subpaths, best_models_sorted, device="cpu"):
    list_trans_dict = []
    for channel in channels:
        for subpath in subpaths:
            # Prep data
            the_data_is = "2D"
            datapath = "Insulin/"
            bio_repli = 0
            datapath = datapath + subpath
            save_dict_name = (
                datapath.replace("/", "_")
                + str(bio_repli)
                + "_"
                + channel
                + "_results_dict_mldir"
                + best_models_sorted[0].split("/")[0]
                + ".pkl"
            )
            save_path = "Insulin/predictions/"
            SEARCH_PATTERN = "not_used"
            OUTPUT_NAME = "not_used"
            print(save_path + save_dict_name)
            PROJECT_NAMES = [channel + "/bg_corr_all_tracked" + str(bio_repli) + ".csv"]
            file = channel + "/bg_corr_all_tracked" + str(bio_repli) + ".csv"
            xy_to_um = 1
            z_to_um = 1

            exp_name = (
                subpath.replace("/", "_")
                + PROJECT_NAMES[0].split("/")[0]
                + "_"
                + str(bio_repli)
            )
            results_dict, X_to_eval, y_to_eval, _, _ = (
                load_or_create_resultsdict_for_rawdata(
                    PROJECT_NAMES,
                    SEARCH_PATTERN,
                    OUTPUT_NAME,
                    globals,
                    datapath,
                    save_dict_name,
                    save_path,
                    best_models_sorted,
                    the_data_is,
                    xy_to_um=xy_to_um,
                    z_to_um=z_to_um,
                    device=device,
                )
            )
            if "ensemble" in (list(results_dict[file].keys())):
                ens_pred = results_dict[file]["ensemble"]
                ens_score = results_dict[file]["ensemble_score"]
            else:
                m1_pred = results_dict[file][list(results_dict[file].keys())[0]][
                    "masked_pred"
                ]
                m1_score = results_dict[file][list(results_dict[file].keys())[0]][
                    "masked_score"
                ]

            if "ensemble" in (list(results_dict[file].keys())):
                m1_norm_trans_dict, _, _ = global_transition_probs(ens_pred)
                list_trans_dict.append(m1_norm_trans_dict)
            else:
                m1_norm_trans_dict, _, _ = global_transition_probs(m1_pred)
                list_trans_dict.append(m1_norm_trans_dict)

    return list_trans_dict


def FP_trace_segments(
    tracks, predictions, fp_datapath, hmm_filename, dim, dt, savename, threshold=5
):
    if os.path.exists(savename):
        traces_in_segments = pickle.load(open(savename, "rb"))
    else:
        difftypes = {0: "Normal", 1: "Directed", 2: "Confined", 3: "Subdiffusive"}
        traces_in_segments = {}
        for i in tqdm(range(len(predictions))):
            segment_trace = []
            segment_pred = []
            segment_fp = []
            tmp_dict = {}
            _, pred_changepoints, _ = find_segments(predictions[i])
            for idx in range(len(pred_changepoints) - 1):
                segment = predictions[i][
                    pred_changepoints[idx] : pred_changepoints[idx + 1]
                ]
                segment_xyz = tracks[i][:, :2][
                    pred_changepoints[idx] : pred_changepoints[idx + 1]
                ]

                # append stuff
                segment_trace.append(segment_xyz)
                segment_pred.append(segment)
                if len(segment_xyz) >= threshold:
                    difftype = difftypes[np.unique(segment)[0]]
                    segment_fp.append(
                        create_fingerprint_track(
                            segment_xyz, fp_datapath, hmm_filename, dim, dt, difftype
                        )
                    )
                else:
                    segment_fp.append(None)
            tmp_dict["segment_trace"] = segment_trace
            tmp_dict["segment_pred"] = segment_pred
            tmp_dict["segment_fp"] = segment_fp
            traces_in_segments[str(i)] = tmp_dict
        pickle.dump(traces_in_segments, open(savename, "wb"))
    return traces_in_segments


def get_FP_for_each_segment(channels, subpaths, best_models_sorted, device="cpu"):
    for channel in channels:
        for subpath in subpaths:
            # Prep data
            the_data_is = "2D"
            datapath = "Insulin/"
            bio_repli = 0
            datapath = datapath + subpath
            save_dict_name = (
                datapath.replace("/", "_")
                + str(bio_repli)
                + "_"
                + str(channel)
                + "_results_dict_mldir"
                + best_models_sorted[0].split("/")[0]
                + ".pkl"
            )
            save_path = "Insulin/predictions/"
            SEARCH_PATTERN = "not_used"
            OUTPUT_NAME = "not_used"
            PROJECT_NAMES = [
                str(channel) + "/bg_corr_all_tracked" + str(bio_repli) + ".csv"
            ]
            file = str(channel) + "/bg_corr_all_tracked" + str(bio_repli) + ".csv"

            xy_to_um = 1
            z_to_um = 1

            results_dict, X_to_eval, y_to_eval, _, _ = (
                load_or_create_resultsdict_for_rawdata(
                    PROJECT_NAMES,
                    SEARCH_PATTERN,
                    OUTPUT_NAME,
                    globals,
                    datapath,
                    save_dict_name,
                    save_path,
                    best_models_sorted,
                    the_data_is,
                    xy_to_um=xy_to_um,
                    z_to_um=z_to_um,
                    device=device,
                )
            )
            if "ensemble" in (list(results_dict[file].keys())):
                ens_pred = results_dict[file]["ensemble"]
                ens_score = results_dict[file]["ensemble_score"]
            else:
                m1_pred = results_dict[file][list(results_dict[file].keys())[0]][
                    "masked_pred"
                ]
                m1_score = results_dict[file][list(results_dict[file].keys())[0]][
                    "masked_score"
                ]

            exp_name = (
                subpath.replace("/", "_")
                + PROJECT_NAMES[0].split("/")[0]
                + "_"
                + str(bio_repli)
            )
            difftypes = {
                "0": "Normal",
                "1": "Directed",
                "2": "Confined",
                "3": "Subdiffusive",
            }
            fp_datapath = "Simulated_diffusion_tracks/"
            hmm_filename = "simulated2D_HMM.json"
            threshold = 5
            dim = 2
            dt = 0.036

            tracks = np.array(X_to_eval, dtype=object)
            predictions = np.array(ens_pred, dtype=object)
            assert len(tracks) == len(predictions)
            savename = datapath + "" + "_FP_trace_segments" + exp_name + ".pkl"

            traces_in_segments = FP_trace_segments(
                tracks,
                predictions,
                fp_datapath,
                hmm_filename,
                dim,
                dt,
                savename,
                threshold=threshold,
            )
    return traces_in_segments


def timestamper():
    date = (
        str(datetime.datetime.now().year)
        + str(datetime.datetime.now().month)
        + str(datetime.datetime.now().day)
    )
    T = str(datetime.datetime.now().hour) + str(datetime.datetime.now().minute)
    return date + T


def plot_diffusion(track, label_list, name="", savename=""):
    color_dict = {"0": "blue", "1": "red", "2": "green", "3": "darkorange"}
    plt.figure()
    x, y = track[:, 0], track[:, 1]
    c = [colors.to_rgba(color_dict[str(label)]) for label in label_list]

    lines = [
        ((x0, y0), (x1, y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])
    ]

    colored_lines = LineCollection(lines, colors=c, linewidths=(2,))
    markers = [
        plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
        for color in color_dict.values()
    ]
    diff_types = ["Norm", "Dir", "Conf", "Sub"]
    # plot data
    fig, ax = plt.subplots()
    ax.add_collection(colored_lines)
    ax.autoscale_view()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(markers, diff_types, numpoints=1, bbox_to_anchor=(1.33, 1.04))
    plt.title(name)
    if len(savename) > 0:
        plt.savefig(savename + ".pdf")
    plt.show()


def plot_diffusion_xy(x, y, label_list, name="", savename=""):
    color_dict = {"0": "blue", "1": "red", "2": "green", "3": "darkorange"}
    plt.figure()
    c = [colors.to_rgba(color_dict[str(label)]) for label in label_list]

    lines = [
        ((x0, y0), (x1, y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])
    ]

    colored_lines = LineCollection(lines, colors=c, linewidths=(2,))
    markers = [
        plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
        for color in color_dict.values()
    ]
    diff_types = ["Norm", "Dir", "Conf", "Sub"]
    # plot data
    fig, ax = plt.subplots()
    ax.add_collection(colored_lines)
    ax.autoscale_view()
    if len(savename) > 0:
        plt.savefig(savename + ".pdf")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(markers, diff_types, numpoints=1, bbox_to_anchor=(1.33, 1.04))
    plt.title(name)
    plt.show()


def timepoint_confidence_plot(pred, savename=""):
    colors_dict = {
        "Normal": "blue",
        "Directed": "red",
        "Confined": "green",
        "Sub": "darkorange",
    }
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.stackplot(
        list(range(len(pred[0]))),
        pred,
        colors=list(colors_dict.values()),
        labels=colors_dict.keys(),
    )
    plt.xlabel("Timestamp")
    plt.ylabel("Confidence")
    if len(savename) > 0:
        plt.savefig(savename + ".pdf")
    plt.show()


def plot_3Ddiffusion_nopred(diffusion_list, traceidx=None):
    fig = go.Figure()
    for i in range(len(diffusion_list)):
        x = diffusion_list[i][:, 0]
        y = diffusion_list[i][:, 1]
        z = diffusion_list[i][:, 2]
        names = []
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                showlegend=False,
                mode="lines",
                marker=dict(size=2),
                text=["t: {}".format(i) for i, x in enumerate(x)],
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=x[0:1],
                y=y[0:1],
                z=z[0:1],
                name="Start",
                mode="markers",
                marker=dict(color="yellowgreen", size=4),
            )
        )
        fig.update_layout(
            title="Trace length: " + str(len(x)) + " Trace idx: " + str(traceidx)
        )
    fig.show()


def load_or_create_resultsdict_for_rawdata(
    PROJECT_NAMES,
    SEARCH_PATTERN,
    OUTPUT_NAME,
    globals,
    datapath,
    save_dict_name,
    save_path,
    best_models_sorted,
    load_old=False,
    use_mlflow=True,
    the_data_is="2D",
    modelpath="Unet_results/mlruns/",
    xy_to_um=0.104,
    z_to_um=0.25,
    min_trace_length=20,
    min_brightness=40,
    min_prefusetrace_length=5,
    min_prefusebrightness=0,
    max_brightness=1000,
    do_fuse_=False,
    features=["XYZ", "SL", "DP"],
    device="cpu",
    min_max_len=601,
):
    if the_data_is == "3D":
        dim = 3
        results_dict = {}
        for file in PROJECT_NAMES:
            if os.path.exists(save_dict_name):
                print(
                    "PATH ALREADY EXITST SO SIMPLY LOAD OTHERWISE DELETE: ",
                    save_dict_name,
                )
                results_dict = pickle.load(open(save_dict_name, "rb"))
                df = load_3D_data(
                    file,
                    SEARCH_PATTERN,
                    OUTPUT_NAME,
                    min_trace_length,
                    min_brightness,
                    max_brightness,
                )
                (
                    tracks,
                    timepoints,
                    frames,
                    track_ids,
                    amplitudes,
                    amplitudes_sig,
                    amplitudes_bg,
                    catIdx,
                ) = curate_3D_data_to_tracks(df, xy_to_um, z_to_um)
                compound_idx = np.array(list(range(len(tracks))))[catIdx > 4]
                handle_compound_tracks(
                    compound_idx,
                    tracks,
                    timepoints,
                    frames,
                    track_ids,
                    amplitudes,
                    amplitudes_sig,
                    amplitudes_bg,
                    min_trace_length,
                    min_brightness,
                )
                if do_fuse_:
                    (
                        tracks,
                        timepoints,
                        track_ids,
                        amplitudes,
                        amplitudes_sig,
                        amplitudes_bg,
                    ) = fuse_tracks(
                        tracks,
                        timepoints,
                        frames,
                        track_ids,
                        amplitudes,
                        amplitudes_sig,
                        amplitudes_bg,
                        min_trace_length,
                        min_brightness,
                        blinking_forgiveness=1,
                    )

                # Prep for ML
                X = prep_tracks(tracks)
                X_to_eval = add_features(X, features)
                y_to_eval = [np.ones(len(x)) * 0.5 for x in X_to_eval]
            else:
                df = load_3D_data(
                    file,
                    SEARCH_PATTERN,
                    OUTPUT_NAME,
                    min_trace_length,
                    min_brightness,
                    max_brightness,
                )
                (
                    tracks,
                    timepoints,
                    frames,
                    track_ids,
                    amplitudes,
                    amplitudes_sig,
                    amplitudes_bg,
                    catIdx,
                ) = curate_3D_data_to_tracks(df, xy_to_um, z_to_um)
                compound_idx = np.array(list(range(len(tracks))))[catIdx > 4]
                handle_compound_tracks(
                    compound_idx,
                    tracks,
                    timepoints,
                    frames,
                    track_ids,
                    amplitudes,
                    amplitudes_sig,
                    amplitudes_bg,
                    min_trace_length,
                    min_brightness,
                )
                if do_fuse_:
                    (
                        tracks,
                        timepoints,
                        frames,
                        track_ids,
                        amplitudes,
                        amplitudes_sig,
                        amplitudes_bg,
                    ) = fuse_tracks(
                        tracks,
                        timepoints,
                        frames,
                        track_ids,
                        amplitudes,
                        amplitudes_sig,
                        amplitudes_bg,
                        min_trace_length,
                        min_brightness,
                        blinking_forgiveness=1,
                    )

                # Prep for ML
                X = prep_tracks(tracks)
                X_to_eval = add_features(X, features)
                y_to_eval = [np.ones(len(x)) * 0.5 for x in X_to_eval]

                print("Predicting on file: ", file)
                # Save a dict with predictions
                files_dict = {}
                for modelname in best_models_sorted:
                    if load_old:
                        if use_mlflow:
                            model = load_oldUnetModels(modelname, device=device)
                        else:
                            model = load_oldUnetModels_directly(
                                modelname, dir=modelpath, device=device
                            )
                    else:
                        if use_mlflow:
                            model = load_UnetModels(
                                modelname, dir=modelpath, device=device
                            )
                        else:
                            model = load_UnetModels_directly(
                                modelname, device=device, dim=dim
                            )

                    tmp_dict = make_preds(
                        model,
                        X_to_eval,
                        y_to_eval,
                        min_max_len=min_max_len,
                        X_padtoken=globals.X_padtoken,
                        y_padtoken=globals.y_padtoken,
                        batch_size=globals.batch_size,
                        device=device,
                    )
                    files_dict[modelname] = tmp_dict

                if len(list(files_dict.keys())) >= 3:
                    files_dict["ensemble_score"] = ensemble_scoring(
                        files_dict[list(files_dict.keys())[0]]["masked_score"],
                        files_dict[list(files_dict.keys())[1]]["masked_score"],
                        files_dict[list(files_dict.keys())[2]]["masked_score"],
                    )
                    ensemble_pred = [
                        np.argmax(files_dict["ensemble_score"][i], axis=0)
                        for i in range(len(files_dict["ensemble_score"]))
                    ]
                    files_dict["ensemble"] = ensemble_pred
                files_dict["X_to_eval"] = X_to_eval
                files_dict["y_to_eval"] = y_to_eval
                files_dict["modelnames"] = best_models_sorted
                results_dict[file] = files_dict

            pickle.dump(results_dict, open(save_dict_name, "wb"))
        return results_dict, X_to_eval, y_to_eval, "put_times_here", "put_tracks_here"

    if the_data_is == "2D":
        dim = 2
        if os.path.exists(save_path + save_dict_name):
            print(
                "\nPATH EXISTS SO SIMPLY LOADS - OTHERWISE DELETE: ",
                save_path + save_dict_name,
            )
            results_dict = pickle.load(open(save_path + save_dict_name, "rb"))
            assert len(PROJECT_NAMES) == 1, "PROJECT_NAMES can only be length 1"
            X_to_eval = np.array(
                load_X(datapath, PROJECT_NAMES[0], features=["XYZ", "SL", "DP"]),
                dtype=object,
            )
            y_to_eval = np.array(
                [np.ones(len(x)) * 0.5 for x in X_to_eval], dtype=object
            )
            print(X_to_eval.shape)
            filtering = np.array([len(t) for t in X_to_eval]) > min_trace_length
            X_to_eval = X_to_eval[filtering]
            y_to_eval = y_to_eval[filtering]
            print(X_to_eval.shape)
        else:
            results_dict = {}
            for file in PROJECT_NAMES:
                X_to_eval = np.array(
                    load_X(datapath, file, features=["XYZ", "SL", "DP"]), dtype=object
                )
                y_to_eval = np.array(
                    [np.ones(len(x)) * 0.5 for x in X_to_eval], dtype=object
                )
                print(X_to_eval.shape)
                filtering = np.array([len(t) for t in X_to_eval]) > min_trace_length
                X_to_eval = X_to_eval[filtering]
                y_to_eval = y_to_eval[filtering]
                print(X_to_eval.shape)
                files_dict = {}
                for modelname in best_models_sorted:
                    if use_mlflow:
                        model = load_UnetModels(modelname, dir=modelpath, device=device)
                    else:
                        model = load_UnetModels_directly(
                            modelname, device=device, dim=dim
                        )

                    tmp_dict = make_preds(
                        model,
                        X_to_eval,
                        y_to_eval,
                        min_max_len=min_max_len,
                        X_padtoken=globals.X_padtoken,
                        y_padtoken=globals.y_padtoken,
                        batch_size=globals.batch_size,
                        device=device,
                    )
                    files_dict[modelname] = tmp_dict
                if len(list(files_dict.keys())) > 1:
                    files_dict["ensemble_score"] = ensemble_scoring(
                        files_dict[list(files_dict.keys())[0]]["masked_score"],
                        files_dict[list(files_dict.keys())[1]]["masked_score"],
                        files_dict[list(files_dict.keys())[2]]["masked_score"],
                    )
                    ensemble_pred = [
                        np.argmax(files_dict["ensemble_score"][i], axis=0)
                        for i in range(len(files_dict["ensemble_score"]))
                    ]
                    files_dict["ensemble"] = ensemble_pred

                files_dict["modelnames"] = best_models_sorted
                results_dict[file] = files_dict

            pickle.dump(results_dict, open(save_path + save_dict_name, "wb"))
        return results_dict, X_to_eval, y_to_eval, "put_times_here", "put_tracks_here"


def plot_3Ddiffusion(diffusion_list, diffusion_labels, traceidx=None):
    diff_types = ["Normal", "Directed", "Confined", "Subdiffusive", "Superdiffusive"]
    color_dict = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c", 3: "#ff7f0e", 4: "purple"}
    for i in range(len(diffusion_list)):
        x = diffusion_list[i][:, 0]
        y = diffusion_list[i][:, 1]
        z = diffusion_list[i][:, 2]
        difftype = diffusion_labels[i]
        fig = go.Figure()
        _, pos, seg_difftype = find_segments(difftype)
        names = []
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z, showlegend=False, marker=dict(color="grey", size=2)
            )
        )
        for idx in range(len(pos) - 1):
            fig.add_trace(
                go.Scatter3d(
                    x=x[pos[idx] : pos[idx + 1] + 1],
                    y=y[pos[idx] : pos[idx + 1] + 1],
                    z=z[pos[idx] : pos[idx + 1] + 1],
                    mode="lines",
                    name=diff_types[seg_difftype[idx]],
                    showlegend=(
                        True if diff_types[seg_difftype[idx]] not in names else False
                    ),
                    line=dict(color=color_dict[seg_difftype[idx]], width=4),
                    text=[
                        "t: {}".format(i)
                        for i, x in enumerate(x[pos[idx] : pos[idx + 1] + 1])
                    ],
                )
            )
            names.append(diff_types[seg_difftype[idx]])
        fig.add_trace(
            go.Scatter3d(
                x=x[0:1],
                y=y[0:1],
                z=z[0:1],
                name="Start",
                mode="markers",
                marker=dict(color="yellowgreen", size=4),
            )
        )
        fig.update_layout(
            title="Trace length: " + str(len(x)) + " Trace idx: " + str(traceidx)
        )
        fig.show()


def global_difftype_occupancy_piechart(pred_argmax):
    difftypes = ["Normal", "Directed", "Confined", "Subdiffusive"]
    colors = ["blue", "red", "green", "darkorange"]
    flat_pred_argmax = [p for subpred in pred_argmax for p in subpred]
    lifetime = np.unique(flat_pred_argmax, return_counts=True)[1] / sum(
        np.unique(flat_pred_argmax, return_counts=True)[1]
    )
    labels = [
        difftypes[i] + str(np.round(np.around(lifetime[i], 3) * 100, 3)) + "%"
        for i in range(len(difftypes))
    ]
    fig = plt.figure()
    plt.pie(lifetime, labels=labels, colors=colors, labeldistance=1.15)
    plt.axis("equal")
    fig.patch.set_facecolor("xkcd:white")
    plt.show()


def global_transition_probs(pred_argmax):
    """
    returns dicts with the transistions

    output[0] = transistions normalized by the difftype, e.g. all transistions
    from normal are normalized by the amount of transistions from normal to the others
    output[1] = the counts of transistions
    output[2] = the transistions normalized to the global number of transistions
    """
    flat_pred_argmax = [p for subpred in pred_argmax for p in subpred]
    run_lens, run_pos, run_difftype = find_segments(flat_pred_argmax)
    before = run_difftype[:-1]
    after = run_difftype[1:]

    transistions = [str(b) + str(a) for b, a in zip(before, after)]
    uniq_transistions, count_per_transistions = np.unique(
        transistions, return_counts=True
    )

    trans_dict = {u: p for u, p in zip(uniq_transistions, count_per_transistions)}
    globalnormed_trans_dict = {
        u: p
        for u, p in zip(
            uniq_transistions, count_per_transistions / sum(count_per_transistions)
        )
    }

    possible_difftypes = 4
    counts_per_difftype = np.zeros(possible_difftypes)
    for tr in uniq_transistions:
        if tr[0] == "0":
            counts_per_difftype[0] += trans_dict[tr]
        if tr[0] == "1":
            counts_per_difftype[1] += trans_dict[tr]
        if tr[0] == "2":
            counts_per_difftype[2] += trans_dict[tr]
        if tr[0] == "3":
            counts_per_difftype[3] += trans_dict[tr]

    norm_trans_dict = {u: p for u, p in zip(uniq_transistions, count_per_transistions)}
    for key in trans_dict.keys():
        if key[0] == "0":
            norm_trans_dict[key] /= counts_per_difftype[int(key[0])]
        if key[0] == "1":
            norm_trans_dict[key] /= counts_per_difftype[int(key[0])]
        if key[0] == "2":
            norm_trans_dict[key] /= counts_per_difftype[int(key[0])]
        if key[0] == "3":
            norm_trans_dict[key] /= counts_per_difftype[int(key[0])]
    return norm_trans_dict, trans_dict, globalnormed_trans_dict


def accuracy_calc(pred, y):
    acc = []
    for i in range(len(pred)):
        acc.append(sum(pred[i] == y[i]) / len(pred[i]))
    return acc


def threshold_tracklength(y, comp="<", lower=0, upper=600):
    keep_idx = []
    if comp == "<":
        for i in range(len(y)):
            if len(y[i]) < upper:
                keep_idx.append(i)
    elif comp == ">":
        for i in range(len(y)):
            if len(y[i]) > lower:
                keep_idx.append(i)
    elif comp == "<<":
        for i in range(len(y)):
            if lower < len(y[i]) < upper:
                keep_idx.append(i)
    return keep_idx


def generate_confusion_matrix(
    cnf_matrix, classes, normalize=False, title="Confusion matrix"
):
    if normalize:
        cnf_matrix = cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cnf_matrix, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cnf_matrix.max() / 2.0

    for i, j in itertools.product(
        range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])
    ):
        plt.text(
            j,
            i,
            format(cnf_matrix[i, j], fmt),
            horizontalalignment="center",
            color="white" if cnf_matrix[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return cnf_matrix


def plot_confusion_matrix(
    predicted_labels_list: list,
    y_test_list: list,
    class_names: list,
    datapath,
    name,
    save: bool = False,
):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure(figsize=(6, 6))
    f1 = f1_score(y_test_list, predicted_labels_list, average="micro")
    recall = recall_score(y_test_list, predicted_labels_list, average="micro")
    precision = precision_score(y_test_list, predicted_labels_list, average="micro")
    generate_confusion_matrix(
        cnf_matrix,
        classes=class_names,
        normalize=True,
        title="Normalized confusion matrix \nF1-score = "
        + "%.3f" % f1
        + " recall "
        + "%.3f" % recall
        + " precision "
        + "%.3f" % precision,
    )
    if save:
        plt.savefig(datapath + "results/confusion_matrix_simulateddata_" + name)
    plt.show()


def load_fingerprints_file(datapath, X_test_filename):
    X_test = pickle.load(open(datapath + X_test_filename, "rb"))
    X_test = [
        arr for arr in X_test if type(arr) == np.ndarray and not np.sum(np.isnan(arr))
    ]
    return np.vstack(X_test)


def kld_weight_func(method: str, epoch: int, epochs: int):
    if method == "constant_zero":
        kld_weight = 0

    if method == "constant_one":
        kld_weight = 1

    if method == "linear":
        kld_weight = epoch / epochs

    return kld_weight


def prep_for_encoding(X, pad_token=0):
    xlens = np.array([x.size()[0] for x in X])
    X = [
        nn.ConstantPad2d((0, max(xlens) - x.size(0)), pad_token)(x.T).T.float()
        for x in X
    ]
    return (X, torch.from_numpy(xlens).cpu())


def remove_short_traces(df, threshold=30):
    particles, counts = np.unique(df["particle"], return_counts=True)
    keep_particles = [
        particles[i] for i in range(len(particles)) if counts[i] > threshold
    ]
    return df[df["particle"].isin(keep_particles)]


def steplength_calculator(t, dim=2):
    assert dim == 1 or dim == 2 or dim == 3
    if dim == 1:
        x = t[:, 0]
        steplength = np.sqrt((x[1:] - x[:-1]) ** 2)
    elif dim == 2:
        x, y = t[:, 0], t[:, 1]
        steplength = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
    elif dim == 3:
        x, y, z = t[:, 0], t[:, 1], t[:, 2]
        steplength = np.sqrt(
            (x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2 + (z[1:] - z[:-1]) ** 2
        )

    return steplength


def find_segments(inarray):
    """
    input: predicted labels, diffusion labels shape = (n,)
    output: segment run lengths, start positions of segments, difftypes of segments
    """
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))  # positions
        return (z, p, ia[i])




def create_fingerprints(datapath: str, filename: str, hmm_filename: str, dim: int):
    """
    input:
    datapath: string to where data is
    filename: filename for tracking df
    hmm_filename: name of hmm json file

    output:
    dumps array of fingerprints at "datapath + 'diff_fingerprints' + filename.split('.')[1]+'.pkl"
    """

    print("Generating fingerprints")
    traces = prep_for_fingerprints(datapath, filename)
    if not os.path.isfile(datapath + hmm_filename):
        steplength = []
        for t in traces:
            x, y = t[:, 0], t[:, 1]
            steplength.append(np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2))

        model = HiddenMarkovModel.from_samples(
            NormalDistribution, n_components=4, X=steplength
        )
        #
        print(model)
        model.bake()
        print("Saving HMM model")

        s = model.to_json()
        f = open(datapath + hmm_filename, "w")
        f.write(s)
        f.close()
    else:
        print("loading HMM model")
        s = datapath + hmm_filename
        file = open(s, "r")
        json_s = ""
        for line in file:
            json_s += line
        model = HiddenMarkovModel.from_json(json_s)
        print(model)
    d = []
    for t in traces:
        x, y = t[:, 0], t[:, 1]
        SL = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
        d.append((x, y, SL))
    import multiprocessing as mp
    from functools import partial

    p = mp.Pool(processes=mp.cpu_count())
    print("resulting")
    print(f"running {len(traces)} particles")

    train_result = []
    for di in d:
        if len(di[0]) < 20:
            continue
        train_result.append(ThirdAppender(di, model, dim=dim))

    assert len(train_result) == len(traces)

    pickle.dump(
        train_result,
        open(datapath + "diff_fingers_" + filename.split(".")[:-1] + ".pkl", "wb"),
    )
    return train_result


def create_fingerprint_track(track, datapath, hmm_filename, dim: int, dt, difftype):
    s = datapath + hmm_filename
    file = open(s, "r")
    json_s = ""
    for line in file:
        json_s += line
    model = HiddenMarkovModel.from_json(json_s)

    if dim == 2:
        x, y = track[:, 0], track[:, 1]
        SL = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
        d = (x, y, SL)
    if dim == 3:
        x, y, z = track[:, 0], track[:, 1], track[:, 2]
        SL = np.sqrt(
            (x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2 + (z[1:] - z[:-1]) ** 2
        )
        d = (x, y, z, SL)

    train_result = ThirdAppender(d, model, dim=dim, dt=dt, difftype=difftype)

    return train_result


def make_array(sorted_df):
    combined = []
    for x, y in zip(sorted_df["x_step"].tolist(), sorted_df["y_step"].tolist()):
        cor_list = []
        cor_list.append(x)
        cor_list.append(y)

        combined.append(cor_list)

    arr = np.array(combined)
    return arr


def prep_for_fingerprints(datapath, filename):
    if filename.split(".")[-1] == "csv":
        df = (
            pd.read_csv(open(datapath + filename))
            .sort_values(by=["particle", "frame"])
            .reset_index(drop=True)
        )
        r = df.groupby(["particle"]).apply(make_array)
        final = r.tolist()
    elif filename.split(".")[-1] == "pkl":
        final = pickle.load(open(datapath + filename, "rb"))
    elif filename.split(".")[-1] == "pt":
        final = torch.load(datapath + filename)
    else:
        raise Exception("load function cant deal with datatype -> bad coding by Jacob")
    return final


def behavior_TDP(input_pred, norm_trans_dict, mass=10000, savename=""):
    flat_pred = np.hstack(input_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    flat_pred = np.hstack(input_pred)
    lifetimes = np.unique(flat_pred, return_counts=True)[1] / sum(
        np.unique(flat_pred, return_counts=True)[1]
    )
    print(lifetimes)
    plt.scatter(
        [0.2, 0.2, 0.8, 0.8],
        [0.2, 0.8, 0.8, 0.2],
        s=[
            mass * lifetimes[0],
            mass * lifetimes[1],
            mass * lifetimes[2],
            mass * lifetimes[3],
        ],
        color=["blue", "red", "green", "darkorange"],
    )

    color_dict = {"0": "blue", "1": "red", "2": "green", "3": "darkorange"}
    markers = [
        plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
        for color in color_dict.values()
    ]
    diff_types = [
        "Normal: {:.2f}%".format(lifetimes[0] * 100),
        "Directed: {:.2f}%".format(lifetimes[1] * 100),
        "Confined: {:.2f}%".format(lifetimes[2] * 100),
        "Subdiffusive: {:.2f}%".format(lifetimes[3] * 100),
    ]
    plt.legend(markers, diff_types, numpoints=1, bbox_to_anchor=(1.55, 1.01))

    dis = 0.02
    dx1, dy1 = 0, 0.2
    dx2, dy2 = 0.2, 0
    dx3, dy3 = 0.33, 0.33

    arrowx1, arrowy1 = 0.20 - dis, 0.40
    arrowx2, arrowy2 = 0.20 + dis, 0.60
    arrowx3, arrowy3 = 0.80 - dis, 0.4
    arrowx4, arrowy4 = 0.80 + dis, 0.6

    arrowx5, arrowy5 = 0.40, 0.20 - dis
    arrowx6, arrowy6 = 0.60, 0.20 + dis
    arrowx7, arrowy7 = 0.40, 0.80 - dis
    arrowx8, arrowy8 = 0.60, 0.80 + dis

    arrowx9, arrowy9 = 0.35 - dis * 1.5, 0.35
    arrowx10, arrowy10 = 0.65 + dis * 1.5, 0.65
    arrowx11, arrowy11 = 0.65, 0.35 - dis * 1.5
    arrowx12, arrowy12 = 0.35, 0.65 + dis * 1.5

    plt.arrow(
        arrowx1, arrowy1, dx1, dy1, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx2, arrowy2, -dx1, -dy1, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx3, arrowy3, dx1, dy1, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx4, arrowy4, -dx1, -dy1, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx5, arrowy5, dx2, dy2, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx6, arrowy6, -dx2, -dy2, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx7, arrowy7, dx2, dy2, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx8, arrowy8, -dx2, -dy2, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx9, arrowy9, dx3, dy3, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx10,
        arrowy10,
        -dx3,
        -dy3,
        head_width=0.03,
        color="k",
        length_includes_head=1,
    )
    plt.arrow(
        arrowx11,
        arrowy11,
        -dx3,
        dy3,
        head_width=0.03,
        color="k",
        length_includes_head=1,
    )
    plt.arrow(
        arrowx12,
        arrowy12,
        dx3,
        -dy3,
        head_width=0.03,
        color="k",
        length_includes_head=1,
    )

    possible_trans = [
        "01",
        "02",
        "03",
        "10",
        "12",
        "13",
        "20",
        "21",
        "23",
        "30",
        "31",
        "32",
    ]
    for trans in possible_trans:
        try:
            norm_trans_dict[trans]
        except:
            norm_trans_dict[trans] = 0

    plt.annotate("{:.2f}".format(norm_trans_dict["01"]), (0.11, 0.5))
    plt.annotate("{:.2f}".format(norm_trans_dict["10"]), (0.23, 0.5))
    plt.annotate("{:.2f}".format(norm_trans_dict["32"]), (0.71, 0.5))
    plt.annotate("{:.2f}".format(norm_trans_dict["23"]), (0.83, 0.5))

    plt.annotate("{:.2f}".format(norm_trans_dict["21"]), (0.47, 0.83))
    plt.annotate("{:.2f}".format(norm_trans_dict["12"]), (0.47, 0.73))
    plt.annotate("{:.2f}".format(norm_trans_dict["30"]), (0.47, 0.23))
    plt.annotate("{:.2f}".format(norm_trans_dict["03"]), (0.47, 0.13))

    plt.annotate("{:.2f}".format(norm_trans_dict["13"]), (0.39, 0.65))
    plt.annotate("{:.2f}".format(norm_trans_dict["20"]), (0.65, 0.58))
    plt.annotate("{:.2f}".format(norm_trans_dict["02"]), (0.3, 0.4))
    plt.annotate("{:.2f}".format(norm_trans_dict["31"]), (0.55, 0.33))

    plt.ylim(0.1, 0.9)
    plt.xlim(0.1, 0.9)
    ax.axis("off")
    ax.set_aspect("equal")
    if len(savename) > 0:
        plt.savefig(savename + ".png", bbox_inches="tight", pad_inches=0.2)
        plt.savefig(savename + ".pdf", bbox_inches="tight", pad_inches=0.2)

    plt.show()


def global_difftype_occupancy_piechart(pred_argmax, savename=""):
    difftypes = ["Normal ", "Directed ", "Confined ", "Subdiffusive "]
    color_list = [
        sns.color_palette("colorblind")[1],
        sns.color_palette("dark")[3],
        sns.color_palette("colorblind")[-1],
        sns.color_palette("dark")[0],
    ]
    flat_pred_argmax = [p for subpred in pred_argmax for p in subpred]
    lifetime = np.unique(flat_pred_argmax, return_counts=True)[1] / sum(
        np.unique(flat_pred_argmax, return_counts=True)[1]
    )
    labels = [
        difftypes[i] + str(np.round(np.around(lifetime[i], 3) * 100, 3)) + "%"
        for i in range(len(difftypes))
    ]
    fig = plt.figure()
    plt.pie(lifetime, labels=labels, colors=color_list, labeldistance=1.15)

    # add a circle at the center to transform it in a donut chart
    my_circle = plt.Circle((0, 0), 0.7, color="white")
    p = plt.gcf()
    p.gca().add_artist(my_circle)

    plt.axis("equal")
    fig.patch.set_facecolor("xkcd:white")
    plt.tight_layout(pad=5)
    if len(savename) > 0:
        plt.savefig(savename + ".png")
        plt.savefig(savename + ".pdf")
    plt.show()


def get_ids_by_acc_threshold(acc, acc_threshold=0.9, above=True):
    thresholded_acc_idx = []
    for i, a in enumerate(acc):
        if above:
            if a < acc_threshold:
                thresholded_acc_idx.append(i)
        else:
            if a > acc_threshold:
                thresholded_acc_idx.append(i)
    return thresholded_acc_idx


def attribute_vs_time_in_video(attribute, times, videos, plottype="line"):
    unique_videos = np.unique(videos)
    for vid in unique_videos:
        idx = np.where(np.array(videos) == vid)[0]
        vid_times = [times[i] for i in range(len(attribute)) if i in idx]
        temp_res = np.mean(vid_times[0][1:] - vid_times[0][:-1])
        att = [attribute[i] for i in range(len(attribute)) if i in idx]

        max_len = int(np.round(np.max(np.hstack(vid_times)) / temp_res + 1))

        att_padded = np.zeros((len(att), int(max_len)))
        for i in range(len(idx)):
            t = [int(np.round(t / temp_res)) for t in vid_times[i]]
            if t[-1] < max_len:
                att_padded[i] = np.array(
                    [np.nan] * (t[0])
                    + list(att[i])
                    + [np.nan] * max(0, max_len - 1 - t[-1])
                )
            else:
                att_padded[i] = list(att[i])

        if plottype == "line":
            plt.figure(figsize=(20, 5))
            plt.title(vid)
            plt.errorbar(
                list(range(len(att_padded[0]) - 1)),
                np.nanmean(att_padded[:, 1:], axis=0),
                yerr=np.nanstd(att_padded[:, 1:], axis=0),
            )

            plt.figure(figsize=(20, 5))
            for a in att_padded:
                plt.plot(a)

        elif plottype == "stackplot":
            list_colsum = []
            difftypes = {0: "ND", 1: "DD", 2: "CD", 3: "SD"}
            unique_col_counts = np.zeros((len(difftypes), att_padded.shape[1]))
            for col in range(att_padded.shape[1]):
                col_unique, col_counts = np.unique(
                    att_padded[:, col], return_counts=True
                )
                colsum = sum(col_counts[~np.isnan(col_unique)])
                list_colsum.append(colsum)
                for key in difftypes.keys():
                    if key in col_unique:
                        unique_col_counts[key, col] = (
                            col_counts[list(col_unique).index(key)] / colsum
                        )
                    else:
                        unique_col_counts[key, col] = 0

            colors_dict = {
                "Normal": "blue",
                "Directed": "red",
                "Confined": "green",
                "Sub": "darkorange",
            }
            _, ax = plt.subplots(2, 1, figsize=(12, 6))
            plt.suptitle(
                "Density of diffusion type and number of tracks per frame\n" + vid,
                size=14,
            )

            ax1, ax2 = np.ravel(ax)
            ax1.stackplot(
                list(range(max_len)),
                unique_col_counts,
                colors=list(colors_dict.values()),
                labels=colors_dict.keys(),
                alpha=0.75,
            )
            ax1.set_ylabel("Density")

            ax2.plot(list_colsum)
            ax2.set_xlabel("Time [Frames]")
            ax2.set_ylabel("Track count")

            plt.tight_layout()


def format_value(value, decimals):
    """
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """

    if isinstance(value, (float, np.floating)):
        return f"{value:.{decimals}f}"
    elif isinstance(value, (int, np.integer)):
        return f"{value:d}"
    else:
        return f"{value}"


def values_to_string(values, decimals):
    """
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'.
    """

    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f"{tmp[0]} +/- {tmp[1]}")
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """Returns the length of the longest string in a list of strings"""
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.
    """

    names = d.keys()
    max_names = len_of_longest_string(names)

    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)

    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1
        string += "{name:s} {value:>{spacing}} \n".format(
            name=name, value=value, spacing=spacing
        )
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color="k"):
    """Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(
        x_coord,
        y_coord,
        string,
        family="monospace",
        fontsize=fontsize,
        transform=ax.transAxes,
        verticalalignment="top",
        color=color,
    )
    return None


# # need fig, ax = plt.subplots()
# d = {'ECE': ECE, 'Sharpness': Sharpness}
# text = nice_string_output(d, extra_spacing=2, decimals=3)
# add_text_to_ax(0.55, 0.2, text, ax, fontsize=14)


def start_end_type_analysis(results_dict):
    start_tmp = {}
    end_tmp = {}
    if "ensemble" in (list(results_dict.keys())):
        ens_pred = results_dict["ensemble"]
        start_type = [ens_pred[i][0] for i in range(len(ens_pred))]
        start_type_perc = np.unique(start_type, return_counts=True)[1] / len(start_type)

        end_type = [ens_pred[i][-1] for i in range(len(ens_pred))]
        end_type_perc = np.unique(end_type, return_counts=True)[1] / len(end_type)
    else:
        m1_pred = results_dict[list(results_dict.keys())[0]]["masked_pred"]
        start_type = [m1_pred[i][0] for i in range(len(m1_pred))]
        start_type_perc = np.unique(start_type, return_counts=True)[1] / len(start_type)

        end_type = [m1_pred[i][-1] for i in range(len(m1_pred))]
        end_type_perc = np.unique(end_type, return_counts=True)[1] / len(end_type)

    difftypes_converter = {0: "Normal", 1: "Directed", 2: "Confined", 3: "Subdiffusive"}
    UniqStart = np.unique(start_type)
    start_tmp["start_type"] = {
        difftypes_converter[UniqStart[i]]: start_type_perc[i]
        for i in range(len(UniqStart))
    }
    for key in difftypes_converter.values():
        if key not in start_tmp["start_type"].keys():
            start_tmp["start_type"][key] = 0

    UniqEnd = np.unique(end_type)
    end_tmp["end_type"] = {
        difftypes_converter[UniqEnd[i]]: end_type_perc[i] for i in range(len(UniqEnd))
    }
    for key in difftypes_converter.values():
        if key not in end_tmp["end_type"].keys():
            end_tmp["end_type"][key] = 0

    return start_tmp, end_tmp


def plot_diffusion_with_FP(
    trace_list, label_list, traces_in_segments, features, savename=""
):
    index_meanings = [
        "alpha",
        "D",
        "R_or_V",
        "pval",
        "logEfficiency",
        "Efficiency",
        "FractalDim",
        "Gaussianity",
        "Kurtosis",
        "MSDratio",
        "Trappedness",
        "t0",
        "t1",
        "t2",
        "t3",
        "lifetime",
        "length",
        "mean SL",
        "mean msd",
        "mean DP",
    ]

    color_dict = {
        "0": sns.color_palette("deep")[0],
        "1": sns.color_palette("dark")[3],
        "2": sns.color_palette("colorblind")[2],
        "3": sns.color_palette("bright")[1],
    }
    for i, track in enumerate(trace_list):
        plt.figure()
        x, y = track[:, 0], track[:, 1]
        c = [colors.to_rgba(color_dict[str(label)]) for label in label_list[i]]

        lines = [
            ((x0, y0), (x1, y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])
        ]

        colored_lines = LineCollection(lines, colors=c, linewidths=(2,))
        markers = [
            plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
            for color in color_dict.values()
        ]
        diff_types = ["Norm", "Dir", "Conf", "Sub"]
        # plot data
        fig, ax = plt.subplots()
        ax.add_collection(colored_lines)
        ax.autoscale_view()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(markers, diff_types, numpoints=1, bbox_to_anchor=(1.33, 1.04))

        trace_colors = []
        featnames = []
        vals = []
        for s in range(len(traces_in_segments["segment_trace"])):
            trace_colors.append(
                color_dict[str(np.unique(traces_in_segments["segment_pred"][s])[0])]
            )
            featname = []
            val = []
            if type(traces_in_segments["segment_fp"][s]) == type(None):
                featnames.append([])
                vals.append([])
                continue
            for feat in features:
                featname.append(index_meanings[feat] + ": ")
                val.append(np.round(traces_in_segments["segment_fp"][s][feat], 5))
            featnames.append(featname)
            vals.append(val)

        counter = 0
        for s in range(len(traces_in_segments["segment_trace"])):
            string = (
                diff_types[np.unique(traces_in_segments["segment_pred"][s])[0]]
                + " "
                + str(s)
            )
            if len(featnames[s]) == 0:
                continue
            for i in range(len(featnames[s])):
                pred = np.unique(traces_in_segments["segment_pred"][s])[0]
                if i == 0 and pred == 2:
                    continue
                string += "\n" + str(featnames[s][i]) + str(vals[s][i])
            ax.text(
                -0.15 + counter * 0.4,
                -0.2,
                string,
                family="monospace",
                fontsize=14,
                transform=ax.transAxes,
                verticalalignment="top",
                color=trace_colors[s],
            )
            counter += 1

        if len(savename) > 0:
            plt.savefig(savename + ".pdf", pad_inches=0.5, bbox_inches="tight")
            plt.savefig(savename + ".png", pad_inches=0.5, bbox_inches="tight")
        plt.show()


def acc_histogram(preds, labels, bins, model_used, datasets):
    acc = np.zeros([len(labels)])
    for i in range(len(labels)):
        acc[i] = np.mean(np.hstack(labels[i]) == np.hstack(preds[i]))
    print(np.mean(acc), np.median(acc), np.std(acc))
    plt.figure()
    plt.hist(acc, bins=bins)
    plt.xlabel("Accuracy")
    plt.ylabel("Density")

    plt.title(
        "N: {} bins: {} mean+/-std: {:.3f}+/-{:.3f} median: {:.3f}".format(
            len(acc), bins, np.mean(acc), np.std(acc), np.median(acc)
        )
    )
    plt.savefig(
        "Unet_results/figures/hist_acc_" + datasets[0] + "_" + model_used + ".pdf"
    )

    return plt.show()


def confusion_matrix_plotter(flat_test_pred, flat_test_true, model_used, datasets):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(flat_test_true, flat_test_pred, normalize="true"), annot=True
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    diffs = ["Normal", "Directed", "Confined", "Sub"]
    plt.xticks(np.linspace(0.5, 3.5, 4), diffs, rotation=45)
    plt.yticks(np.linspace(0.5, 3.5, 4), diffs, rotation=0)
    flat_acc = np.mean(np.hstack(flat_test_pred) == np.hstack(flat_test_true))
    plt.title("N: {}, Accuracy: {:.3f}".format(len(flat_test_true), flat_acc))
    plt.tight_layout()
    plt.savefig(
        "Unet_results/figures/confusion_matrix_"
        + datasets[0]
        + "_"
        + model_used
        + ".pdf"
    )
    plt.show()
    print(classification_report(flat_test_true, flat_test_pred, target_names=diffs))
    print("Accuracy:", np.mean(np.array(flat_test_pred) == np.array(flat_test_true)))

    return plt.show()


def plot_coloc_temporalcolor_w_full(
    POI, COL, POItracks_full, COLOCtracks_full, tracks_coloc_info, coloc_pred, pred_idx
):
    cmaps_names = ["winter_r", "autumn_r"]
    cmaps = [plt.get_cmap(name) for name in cmaps_names]
    cmaps_gradients = [cmap(np.linspace(0, 1, 200)) for cmap in cmaps]
    cmaps_dict = dict(zip(cmaps_names, cmaps_gradients))

    patches_cmaps_gradients = []
    for cmap_name, cmap_colors in cmaps_dict.items():
        cmap_gradient = [
            patches.Patch(facecolor=c, edgecolor=c, label=cmap_name)
            for c in cmap_colors
        ]
        patches_cmaps_gradients.append(cmap_gradient)

    MAP = "winter_r"
    fig = plt.figure()
    ax = fig.add_subplot()

    poi_full_idx = tracks_coloc_info[pred_idx, 0]
    col_full_idx = tracks_coloc_info[pred_idx, 1]
    plt.plot(POItracks_full[poi_full_idx][:, 0], POItracks_full[poi_full_idx][:, 1])
    plt.plot(COLOCtracks_full[col_full_idx][:, 0], COLOCtracks_full[col_full_idx][:, 1])

    cm = plt.get_cmap(MAP)
    ax.set_prop_cycle(
        "color", [cm(1.0 * i / (len(POI) - 1)) for i in range(len(POI) - 1)]
    )
    for i in range(len(POI) - 1):
        ax.plot(POI[i : i + 2, 0], POI[i : i + 2, 1], label="temporal evolution of POI")

    MAP = "autumn_r"
    cm = plt.get_cmap(MAP)
    ax.set_prop_cycle(
        "color", [cm(1.0 * i / (len(POI) - 1)) for i in range(len(POI) - 1)]
    )
    for i in range(len(POI) - 1):
        ax.plot(
            COL[i : i + 2, 0], COL[i : i + 2, 1], label="temporal evolution of Co-loc"
        )
    # Create custom legend (with a large fontsize to better illustrate the result)
    plt.legend(
        handles=patches_cmaps_gradients,
        labels=["Temporal evolution of POI", "Temporal evolution of Co-loc"],
        fontsize=14,
        handler_map={list: HandlerTuple(ndivide=None, pad=0)},
    )
    plt.title(str(coloc_pred[pred_idx][1]) + " " + str(len(POI)))


def plot_coloc_temporalcolor(POI, COL, coloc_pred, pred_idx):
    cmaps_names = ["winter_r", "autumn_r"]
    cmaps = [plt.get_cmap(name) for name in cmaps_names]
    cmaps_gradients = [cmap(np.linspace(0, 1, 200)) for cmap in cmaps]
    cmaps_dict = dict(zip(cmaps_names, cmaps_gradients))

    patches_cmaps_gradients = []
    for cmap_name, cmap_colors in cmaps_dict.items():
        cmap_gradient = [
            patches.Patch(facecolor=c, edgecolor=c, label=cmap_name)
            for c in cmap_colors
        ]
        patches_cmaps_gradients.append(cmap_gradient)

    MAP = "winter_r"
    fig = plt.figure()
    ax = fig.add_subplot()

    cm = plt.get_cmap(MAP)
    ax.set_prop_cycle(
        "color", [cm(1.0 * i / (len(POI) - 1)) for i in range(len(POI) - 1)]
    )
    for i in range(len(POI) - 1):
        ax.plot(POI[i : i + 2, 0], POI[i : i + 2, 1], label="temporal evolution of POI")

    MAP = "autumn_r"
    cm = plt.get_cmap(MAP)
    ax.set_prop_cycle(
        "color", [cm(1.0 * i / (len(POI) - 1)) for i in range(len(POI) - 1)]
    )
    for i in range(len(POI) - 1):
        ax.plot(
            COL[i : i + 2, 0], COL[i : i + 2, 1], label="temporal evolution of Co-loc"
        )
    # Create custom legend (with a large fontsize to better illustrate the result)
    plt.legend(
        handles=patches_cmaps_gradients,
        labels=["Temporal evolution of POI", "Temporal evolution of Co-loc"],
        fontsize=14,
        handler_map={list: HandlerTuple(ndivide=None, pad=0)},
    )
    plt.title(str(coloc_pred[pred_idx][1]) + " " + str(len(POI)))


def reliability_plot(ensemble_score, y_to_eval, number_quantiles=20, savename=""):
    flat_trues = np.hstack(y_to_eval)
    ND_diff_probs = np.hstack(
        [ensemble_score[i][0, :] for i in range(len(ensemble_score))]
    )
    DM_diff_probs = np.hstack(
        [ensemble_score[i][1, :] for i in range(len(ensemble_score))]
    )
    CD_diff_probs = np.hstack(
        [ensemble_score[i][2, :] for i in range(len(ensemble_score))]
    )
    SD_diff_probs = np.hstack(
        [ensemble_score[i][3, :] for i in range(len(ensemble_score))]
    )

    binedges = np.histogram(ND_diff_probs, bins=number_quantiles, range=(0, 1))[1]
    bin_place = np.digitize(ND_diff_probs, bins=binedges)
    ND_reliability = [
        np.mean(flat_trues[np.where(bin_place == i)] == 0)
        for i in range(1, number_quantiles + 1)
    ]

    binedges = np.histogram(DM_diff_probs, bins=number_quantiles, range=(0, 1))[1]
    bin_place = np.digitize(DM_diff_probs, bins=binedges)
    DM_reliability = [
        np.mean(flat_trues[np.where(bin_place == i)] == 1)
        for i in range(1, number_quantiles + 1)
    ]

    binedges = np.histogram(CD_diff_probs, bins=number_quantiles, range=(0, 1))[1]
    bin_place = np.digitize(CD_diff_probs, bins=binedges)
    CD_reliability = [
        np.mean(flat_trues[np.where(bin_place == i)] == 2)
        for i in range(1, number_quantiles + 1)
    ]

    binedges = np.histogram(SD_diff_probs, bins=number_quantiles, range=(0, 1))[1]
    bin_place = np.digitize(SD_diff_probs, bins=binedges)
    SD_reliability = [
        np.mean(flat_trues[np.where(bin_place == i)] == 3)
        for i in range(1, number_quantiles + 1)
    ][::-1]

    qs = (binedges[1:] + binedges[:-1]) / 2
    combined_reliability = np.nanmean(
        np.vstack([ND_reliability, DM_reliability, CD_reliability, SD_reliability]),
        axis=0,
    )
    expected_qs = np.linspace(0, 1, number_quantiles + 1)

    ECE = np.nanmean(np.abs(np.array(combined_reliability) - np.array(qs))).astype(
        float
    )
    print("ECE", ECE)

    NLL_benchmark = np.mean(
        [
            nn.NLLLoss()(
                torch.tensor(ensemble_score[i]).T,
                torch.tensor(y_to_eval[i])[torch.randperm(len(y_to_eval[i]))].long(),
            ).item()
            for i in range(len(y_to_eval))
        ]
    )
    NLL = np.mean(
        [
            nn.NLLLoss()(
                torch.tensor(ensemble_score[i]).T, torch.tensor(y_to_eval[i]).long()
            ).item()
            for i in range(len(y_to_eval))
        ]
    )
    NLL_improvement = (NLL - NLL_benchmark) / NLL_benchmark
    print("NLL improvement", NLL_improvement)

    Sharpness = 1 - np.nanmean(
        [
            np.mean(
                np.abs(
                    np.max(ensemble_score[i], axis=0)
                    - np.ones(len(np.max(ensemble_score[i], axis=0)))
                )
            )
            for i in range(len(y_to_eval))
        ]
    ).astype(float)
    print("Sharpness", Sharpness)

    fig, ax = plt.subplots(figsize=(9, 6))
    plt.plot(expected_qs, expected_qs, "--", c="k")
    plt.bar(qs, qs, color="red", alpha=0.25, width=1 / number_quantiles)
    plt.bar(
        qs,
        combined_reliability,
        width=1 / number_quantiles,
        color="lightgrey",
        edgecolor="dimgrey",
    )
    plt.plot(qs, combined_reliability, c="darkred")
    plt.scatter(qs, combined_reliability, c="darkred", zorder=2)
    plt.scatter(qs, qs, c="k", zorder=2)
    plt.xlim(0, 1)

    d = {"Sharpness": Sharpness, "ECE": ECE, "NLL": NLL_improvement}
    text = nice_string_output(d, extra_spacing=2, decimals=4)
    add_text_to_ax(0.01, 0.97, text, ax, fontsize=16)
    if len(savename) > 0:
        plt.savefig(savename + ".png")
        plt.savefig(savename + ".pdf")
    plt.show()


def plot_coverage(file, cov_savename, X_to_eval):
    plt.figure(figsize=(12, 12))
    x = np.hstack([pos[:, 0] for pos in X_to_eval])
    y = np.hstack([pos[:, 1] for pos in X_to_eval])

    from scipy.stats.kde import gaussian_kde

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=3)
    plt.title(file + "\nParticle movement coverage visualized\n")
    plt.xlabel("x [um]")
    plt.ylabel("y [um]")
    ax.set_aspect("equal")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    if len(cov_savename) > 1:
        plt.savefig(cov_savename)
    plt.show()


def difftype_piechart(means, stds, title="", savefig=""):
    difftypes = ["Normal", "Directed", "Confined", "Subdiffusive"]
    colors = [
        sns.color_palette("colorblind")[1],
        sns.color_palette("dark")[3],
        sns.color_palette("colorblind")[-1],
        sns.color_palette("dark")[0],
    ]
    labels = [
        difftypes[i]
        + " "
        + str(np.round(np.around(means[i], 3) * 100, 3))
        + "+-"
        + str(np.round(np.around(stds[i], 3) * 100, 3))
        + "%"
        for i in range(len(difftypes))
    ]
    fig = plt.figure(figsize=(4, 4))
    plt.pie(means, labels=labels, colors=colors, labeldistance=1.1)
    plt.axis("equal")
    fig.patch.set_facecolor("xkcd:white")
    my_circle = plt.Circle((0, 0), 0.7, color="white")
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.title(title, size=16)
    plt.rcParams["axes.titlepad"] = 24
    if len(savefig) != 0:
        plt.savefig(savefig + ".png", dpi=700, pad_inches=0.5, bbox_inches="tight")
        plt.savefig(savefig + ".pdf", dpi=700, pad_inches=0.5, bbox_inches="tight")
    plt.show()


def behavior_TDP_v2(
    lifetimes,
    std_lifetimes,
    norm_trans_dict,
    std_trans_dict,
    mass=10000,
    savename="",
    title="",
):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.scatter(
        [0.2, 0.2, 0.8, 0.8],
        [0.2, 0.8, 0.8, 0.2],
        s=[
            mass * lifetimes[0],
            mass * lifetimes[1],
            mass * lifetimes[2],
            mass * lifetimes[3],
        ],
        color=["blue", "red", "green", "darkorange"],
    )

    color_dict = {"0": "blue", "1": "red", "2": "green", "3": "darkorange"}
    markers = [
        plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
        for color in color_dict.values()
    ]
    diff_types = [
        "Normal: {:.2f}+-{:.1f}%".format(lifetimes[0] * 100, std_lifetimes[0] * 100),
        "Directed: {:.2f}+-{:.1f}%".format(lifetimes[1] * 100, std_lifetimes[1] * 100),
        "Confined: {:.2f}+-{:.1f}%".format(lifetimes[2] * 100, std_lifetimes[2] * 100),
        "Subdiffusive: {:.2f}+-{:.1f}%".format(
            lifetimes[3] * 100, std_lifetimes[3] * 100
        ),
    ]
    plt.legend(markers, diff_types, numpoints=1, bbox_to_anchor=(1.55, 1.01))

    dis = 0.02
    dx1, dy1 = 0, 0.2
    dx2, dy2 = 0.2, 0
    dx3, dy3 = 0.33, 0.33

    arrowx1, arrowy1 = 0.20 - dis, 0.40
    arrowx2, arrowy2 = 0.20 + dis, 0.60
    arrowx3, arrowy3 = 0.80 - dis, 0.4
    arrowx4, arrowy4 = 0.80 + dis, 0.6

    arrowx5, arrowy5 = 0.40, 0.20 - dis
    arrowx6, arrowy6 = 0.60, 0.20 + dis
    arrowx7, arrowy7 = 0.40, 0.80 - dis
    arrowx8, arrowy8 = 0.60, 0.80 + dis

    arrowx9, arrowy9 = 0.35 - dis * 1.5, 0.35
    arrowx10, arrowy10 = 0.65 + dis * 1.5, 0.65
    arrowx11, arrowy11 = 0.65, 0.35 - dis * 1.5
    arrowx12, arrowy12 = 0.35, 0.65 + dis * 1.5

    plt.arrow(
        arrowx1, arrowy1, dx1, dy1, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx2, arrowy2, -dx1, -dy1, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx3, arrowy3, dx1, dy1, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx4, arrowy4, -dx1, -dy1, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx5, arrowy5, dx2, dy2, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx6, arrowy6, -dx2, -dy2, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx7, arrowy7, dx2, dy2, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx8, arrowy8, -dx2, -dy2, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx9, arrowy9, dx3, dy3, head_width=0.03, color="k", length_includes_head=1
    )
    plt.arrow(
        arrowx10,
        arrowy10,
        -dx3,
        -dy3,
        head_width=0.03,
        color="k",
        length_includes_head=1,
    )
    plt.arrow(
        arrowx11,
        arrowy11,
        -dx3,
        dy3,
        head_width=0.03,
        color="k",
        length_includes_head=1,
    )
    plt.arrow(
        arrowx12,
        arrowy12,
        dx3,
        -dy3,
        head_width=0.03,
        color="k",
        length_includes_head=1,
    )

    possible_trans = [
        "01",
        "02",
        "03",
        "10",
        "12",
        "13",
        "20",
        "21",
        "23",
        "30",
        "31",
        "32",
    ]
    for trans in possible_trans:
        try:
            norm_trans_dict[trans]
        except:
            norm_trans_dict[trans] = 0

    plt.annotate(
        "{:.1f}+-\n{:.2f}".format(
            norm_trans_dict["01"] * 100, std_trans_dict["01"] * 100
        ),
        (0.1, 0.47),
    )
    plt.annotate(
        "{:.1f}+-\n{:.2f}".format(
            norm_trans_dict["10"] * 100, std_trans_dict["10"] * 100
        ),
        (0.23, 0.47),
    )
    plt.annotate(
        "{:.1f}+-\n{:.2f}".format(
            norm_trans_dict["32"] * 100, std_trans_dict["32"] * 100
        ),
        (0.68, 0.47),
    )
    plt.annotate(
        "{:.1f}+-\n{:.2f}".format(
            norm_trans_dict["23"] * 100, std_trans_dict["23"] * 100
        ),
        (0.83, 0.47),
    )

    plt.annotate(
        "{:.1f}+-{:.2f}".format(
            norm_trans_dict["21"] * 100, std_trans_dict["21"] * 100
        ),
        (0.43, 0.85),
    )
    plt.annotate(
        "{:.1f}+-{:.2f}".format(
            norm_trans_dict["12"] * 100, std_trans_dict["12"] * 100
        ),
        (0.43, 0.73),
    )
    plt.annotate(
        "{:.1f}+-{:.2f}".format(
            norm_trans_dict["30"] * 100, std_trans_dict["30"] * 100
        ),
        (0.43, 0.25),
    )
    plt.annotate(
        "{:.1f}+-{:.2f}".format(
            norm_trans_dict["03"] * 100, std_trans_dict["03"] * 100
        ),
        (0.43, 0.13),
    )

    plt.annotate(
        "{:.1f}+-\n{:.2f}".format(
            norm_trans_dict["13"] * 100, std_trans_dict["13"] * 100
        ),
        (0.30, 0.69),
    )
    plt.annotate(
        "{:.1f}+-\n{:.2f}".format(
            norm_trans_dict["20"] * 100, std_trans_dict["20"] * 100
        ),
        (0.69, 0.65),
    )
    plt.annotate(
        "{:.1f}+-\n{:.2f}".format(
            norm_trans_dict["02"] * 100, std_trans_dict["02"] * 100
        ),
        (0.25, 0.29),
    )
    plt.annotate(
        "{:.1f}+-\n{:.2f}".format(
            norm_trans_dict["31"] * 100, std_trans_dict["31"] * 100
        ),
        (0.65, 0.26),
    )

    plt.ylim(0.1, 0.9)
    plt.xlim(0.1, 0.9)
    ax.axis("off")
    ax.set_aspect("equal")
    plt.title(title)
    if len(savename) > 0:
        plt.savefig(savename + ".png", bbox_inches="tight", pad_inches=0.3)
        plt.savefig(savename + ".pdf", bbox_inches="tight", pad_inches=0.3)

    plt.show()


def FPsegment_hist_plotting(
    conds, biotypes, channels, bio_replis, savefig_bool=True, bins=50
):
    # fp
    index_meanings = [
        "alpha",
        "D",
        "R_or_v",
        "pval",
        "logEfficiency",
        "Efficiency",
        "FractalDim",
        "Gaussianity",
        "Kurtosis",
        "MSDratio",
        "Trappedness",
        "t0",
        "t1",
        "t2",
        "t3",
        "lifetime",
        "len(x)",
        "mean SL",
        "mean msd",
        "mean dp",
    ]
    # plot

    xmins = [
        -1,
        -0.5,
        -1,
        -1,
        -20,
        0,
        0,
        0,
        0,
        0,
        -3,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        0,
        0,
        0,
        0,
        -0.2,
    ]
    xmaxs = [
        2,
        1.5,
        2,
        2,
        0,
        1,
        5,
        5,
        10,
        2,
        2,
        0.1,
        0.1,
        0.1,
        0.1,
        200,
        200,
        0.75,
        2,
        0.2,
    ]

    for feat_num in range(len(index_meanings)):
        xmin, xmax = xmins[feat_num], xmaxs[feat_num]

        ND_fp = []
        DM_fp = []
        CD_fp = []
        SD_fp = []

        ys_ND = []
        ys_DM = []
        ys_CD = []
        ys_SD = []

        for n in range(len(conds)):
            datapath = "Insulin/" + conds[n] + "/" + biotypes[n] + "/"
            exp_name = (
                conds[n] + "_" + biotypes[n] + "_" + channels[n] + "_" + bio_replis[n]
            )
            traces_in_segments = pickle.load(
                open(datapath + "" + "_FP_trace_segments" + exp_name + ".pkl", "rb")
            )
            for key in traces_in_segments.keys():
                for i in range(len(traces_in_segments[key]["segment_pred"])):
                    pred = np.unique(traces_in_segments[key]["segment_pred"][i])[0]
                    fp = traces_in_segments[key]["segment_fp"][i]
                    if type(fp) == type(None):
                        continue
                    if pred == 0:
                        ys_ND.append(n)
                        ND_fp.append(fp)
                    if pred == 1:
                        ys_DM.append(n)
                        DM_fp.append(fp)
                    if pred == 2:
                        ys_CD.append(n)
                        CD_fp.append(fp)
                    if pred == 3:
                        ys_SD.append(n)
                        SD_fp.append(fp)
        ND_fp = np.vstack(ND_fp).astype(float)
        DM_fp = np.vstack(DM_fp).astype(float)
        CD_fp = np.vstack(CD_fp).astype(float)
        SD_fp = np.vstack(SD_fp).astype(float)
        ys_ND = np.array(ys_ND)
        ys_DM = np.array(ys_DM)
        ys_CD = np.array(ys_CD)
        ys_SD = np.array(ys_SD)

        print(index_meanings[feat_num])

        ND_fp_feat = ND_fp[:, feat_num][~np.isnan(ND_fp[:, feat_num])]
        DM_fp_feat = DM_fp[:, feat_num][~np.isnan(DM_fp[:, feat_num])]
        CD_fp_feat = CD_fp[:, feat_num][~np.isnan(CD_fp[:, feat_num])]
        SD_fp_feat = SD_fp[:, feat_num][~np.isnan(SD_fp[:, feat_num])]
        all = np.hstack((ND_fp_feat, DM_fp_feat, CD_fp_feat, SD_fp_feat))

        colors = ["darkred", "dimgrey", "royalblue", "k"]

        fig, ax = plt.subplots(4, 1, figsize=(4, 8))
        for a in range(len(ax)):
            if a == 0:
                for i in range(len(conds)):
                    ax[a].hist(
                        ND_fp[ys_ND == i][:, feat_num],
                        alpha=0.7,
                        bins=bins,
                        range=(xmin, xmax),
                        color=colors[i],
                        density=True,
                        label=conds[i] + " " + biotypes[i] + " " + channels[i],
                    )
                    ax[a].set_title(
                        "Normal N=" + str(len(ND_fp[ys_ND == i][:, feat_num]))
                    )
            if a == 1:
                for i in range(len(conds)):
                    ax[a].hist(
                        DM_fp[:, feat_num][ys_DM == i],
                        alpha=0.7,
                        bins=bins,
                        range=(xmin, xmax),
                        color=colors[i],
                        density=True,
                        label=conds[i] + " " + biotypes[i] + " " + channels[i],
                    )
                    ax[a].set_title(
                        "Directed N=" + str(len(DM_fp[ys_DM == i][:, feat_num]))
                    )
            if a == 2:
                if feat_num != 0:
                    for i in range(len(conds)):
                        ax[a].hist(
                            CD_fp[:, feat_num][ys_CD == i],
                            alpha=0.7,
                            bins=bins,
                            range=(xmin, xmax),
                            color=colors[i],
                            density=True,
                            label=conds[i] + " " + biotypes[i] + " " + channels[i],
                        )
                        ax[a].set_title(
                            "Confined N=" + str(len(CD_fp[ys_CD == i][:, feat_num]))
                        )
            if a == 3:
                for i in range(len(conds)):
                    ax[a].hist(
                        SD_fp[:, feat_num][ys_SD == i],
                        alpha=0.7,
                        bins=bins,
                        range=(xmin, xmax),
                        color=colors[i],
                        density=True,
                        label=conds[i] + " " + biotypes[i] + " " + channels[i],
                    )
                    ax[a].set_title(
                        "Subdiffusive N=" + str(len(SD_fp[ys_SD == i][:, feat_num]))
                    )

        plt.suptitle(index_meanings[feat_num])
        plt.tight_layout(h_pad=0.1, pad=1)
        plt.legend(bbox_to_anchor=(1, -0.2))

        figpath = "Insulin/figures/"
        string = ""
        for i in range(len(conds)):
            string += (
                index_meanings[feat_num]
                + "_"
                + conds[i]
                + "_"
                + biotypes[i]
                + "_"
                + channels[i]
                + "_"
            )
        savefig = figpath + string

        if savefig_bool:
            plt.savefig(savefig + ".png", dpi=700, bbox_inches="tight", pad_inches=0.3)
            plt.savefig(savefig + ".pdf", dpi=700, bbox_inches="tight", pad_inches=0.3)


def FP_trace_segments_LDA(
    tracks, predictions, fp_datapath, hmm_filename, dim, dt, savename, threshold=20
):

    fp_index = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]

    if os.path.exists(savename):
        traces_in_segments = pickle.load(open(savename, "rb"))
    else:
        difftypes = {0: "Normal", 1: "Directed", 2: "Confined", 3: "Subdiffusive"}
        traces_in_segments = []
        for i in tqdm(range(len(predictions))):
            ND_segment_fp = np.zeros(len(fp_index))
            DM_segment_fp = np.zeros(len(fp_index))
            CD_segment_fp = np.zeros(len(fp_index))
            SD_segment_fp = np.zeros(len(fp_index))
            ND_count = 0
            DM_count = 0
            CD_count = 0
            SD_count = 0
            _, pred_changepoints, pred_difftype = find_segments(predictions[i])
            for idx in range(len(pred_changepoints) - 1):
                segment = predictions[i][
                    pred_changepoints[idx] : pred_changepoints[idx + 1]
                ]
                segment_xyz = tracks[i][:, :2][
                    pred_changepoints[idx] : pred_changepoints[idx + 1]
                ]
                if len(segment_xyz) >= threshold:
                    difftype = difftypes[np.unique(segment)[0]]
                    out = create_fingerprint_track(
                        segment_xyz, fp_datapath, hmm_filename, dim, dt, difftype
                    )
                    out = out[fp_index].astype(float)
                    if pred_difftype[idx] == 0:
                        ND_segment_fp += out
                        ND_count += 1
                    if pred_difftype[idx] == 1:
                        DM_segment_fp += out
                        DM_count += 1
                    if pred_difftype[idx] == 2:
                        CD_segment_fp += out
                        CD_count += 1
                    if pred_difftype[idx] == 3:
                        SD_segment_fp += out
                        SD_count += 1
                else:
                    continue
            ND_segment_fp = ND_segment_fp / max(1, ND_count)
            DM_segment_fp = DM_segment_fp / max(1, DM_count)
            CD_segment_fp = CD_segment_fp / max(1, CD_count)
            SD_segment_fp = SD_segment_fp / max(1, SD_count)

            uniq_pred = np.unique(predictions[i], return_counts=True)
            tau = uniq_pred[1] / sum(uniq_pred[1])
            difftype = uniq_pred[0]
            lifetime = np.zeros(len(difftypes))
            for i in range(len(tau)):
                lifetime[difftype[i]] = tau[i]
            traces_in_segments.append(
                np.hstack(
                    (
                        ND_segment_fp,
                        DM_segment_fp,
                        CD_segment_fp,
                        SD_segment_fp,
                        lifetime,
                    )
                )
            )

        traces_in_segments = np.vstack(traces_in_segments)
        pickle.dump(traces_in_segments, open(savename, "wb"))
    return traces_in_segments


def jensen_shannon_distance(p, q):
    ref_means = torch.mean(torch.stack([p.loc, q.loc]), axis=0)
    ref_stds = torch.mean(torch.stack([p.scale**2, q.scale**2]), axis=0) ** 0.5
    ref_dist = Normal(ref_means, ref_stds)

    kl1 = 0.5 * torch.sum(torch.distributions.kl.kl_divergence(p, ref_dist))
    kl2 = 0.5 * torch.sum(torch.distributions.kl.kl_divergence(q, ref_dist))
    jsd = kl1 + kl2
    return np.sqrt(jsd.numpy())


def mahalanobis(u, v, cov):
    if type(cov) == np.ndarray:
        cov = torch.from_numpy(cov) ** 2 * torch.eye(len(cov))
    else:
        cov = cov**2 * torch.eye(len(cov))
    if type(u) == np.ndarray:
        u = torch.from_numpy(u)
    if type(v) == np.ndarray:
        v = torch.from_numpy(v)
    delta = u - v
    m = torch.dot(delta, torch.matmul(torch.inverse(cov + 10**-6), delta))
    return torch.sqrt(m).cpu().numpy()


def euclidean_dist(ref, q):
    if type(ref) == np.ndarray:
        ref = torch.from_numpy(ref)
    if type(q) == np.ndarray:
        q = torch.from_numpy(q)
    return np.linalg.norm(ref.cpu().numpy() - q.cpu().numpy())


def get_temporal_info_dict(
    globals,
    subpath,
    channel,
    min_seglen_for_FP=5,
    num_difftypes=4,
    D="Insulin/",
    bio_repli=0,
    the_data_is="2D",
    best_models_sorted=["best_models_sorted"],
    device="cpu",
    save_path="Insulin/predictions/",
    SEARCH_PATTERN="not_used",
    OUTPUT_NAME="not_used",
    fp_datapath="fp_datapath",
    hmm_filename="hmm_filename",
    dim=2,
    dt=0.036,
):

    difftypes = {"0": "Normal", "1": "Directed", "2": "Confined", "3": "Subdiffusive"}
    datapath = D + subpath
    file = channel + "/bg_corr_all_tracked" + str(bio_repli) + ".csv"
    exp_name = subpath.replace("/", "_") + file.split("/")[0] + "_" + str(bio_repli)
    savename = datapath + "" + "temporal_info_dict_" + exp_name + ".pkl"

    # print(datapath + 'temporal_info_dict_' + exp_name +'.pkl')
    if os.path.exists(datapath + "temporal_info_dict_" + exp_name + ".pkl"):
        savename = datapath + "temporal_info_dict_" + exp_name + ".pkl"
        temporal_info_dict = pickle.load(open(savename, "rb"))

    else:
        if the_data_is == "2D":
            save_dict_name = (
                datapath.replace("/", "_")
                + str(bio_repli)
                + "_"
                + str(channel)
                + "_results_dict_mldir"
                + best_models_sorted[0].split("/")[0]
                + ".pkl"
            )
            PROJECT_NAMES = [
                str(channel) + "/bg_corr_all_tracked" + str(bio_repli) + ".csv"
            ]
            xy_to_um = 1
            z_to_um = 1
            print(save_dict_name)
            results_dict, X_to_eval, _, _, _ = load_or_create_resultsdict_for_rawdata(
                PROJECT_NAMES,
                SEARCH_PATTERN,
                OUTPUT_NAME,
                globals,
                datapath,
                save_dict_name,
                save_path,
                best_models_sorted,
                the_data_is,
                xy_to_um=xy_to_um,
                z_to_um=z_to_um,
                device=device,
            )
            print(results_dict[file].keys())
            if "ensemble" in (list(results_dict[file].keys())):
                predictions = results_dict[file]["ensemble"]
            else:
                predictions = results_dict[file][list(results_dict[file].keys())[0]][
                    "masked_pred"
                ]

            tracks = X_to_eval

            assert len(tracks) == len(
                predictions
            ), "len(tracks) {} len(predictions){}".format(len(tracks), len(predictions))

            max_num_cp = np.max([len(find_segments(p)[1]) - 1 for p in predictions])
            temporal_tdp_occ = np.zeros((num_difftypes, max_num_cp))
            temporal_tdp_lifetime = np.zeros((num_difftypes, max_num_cp))
            temporal_tdp_D = np.zeros((num_difftypes, max_num_cp))
            temporal_tdp_a = np.zeros((num_difftypes, max_num_cp))
            temporal_trans = np.zeros((max_num_cp, num_difftypes, num_difftypes))

            for i in tqdm(range(len(predictions))):
                _, pred_changepoints, pred_diffs = find_segments(predictions[i])
                for idx in range(len(pred_changepoints) - 1):
                    segment = predictions[i][
                        pred_changepoints[idx] : pred_changepoints[idx + 1]
                    ]
                    segment_xyz = tracks[i][
                        pred_changepoints[idx] : pred_changepoints[idx + 1]
                    ]
                    behavior = np.unique(segment)[0]
                    lifetime = len(segment)
                    temporal_tdp_occ[behavior, idx] += 1
                    temporal_tdp_lifetime[behavior, idx] += lifetime

                    if len(segment_xyz) >= min_seglen_for_FP:
                        difftype = difftypes[str(np.unique(segment)[0])]
                        param = create_fingerprint_track(
                            segment_xyz, fp_datapath, hmm_filename, dim, dt, difftype
                        )
                        temporal_tdp_a[behavior, idx] = param[0]
                        temporal_tdp_D[behavior, idx] = param[1]
                for idx in range(len(pred_diffs)):
                    if len(pred_diffs) == 1:
                        temporal_trans[idx, pred_diffs[idx], pred_diffs[idx]] += 1
                    else:
                        if idx == len(pred_diffs) - 1:
                            continue
                        temporal_trans[idx, pred_diffs[idx], pred_diffs[idx + 1]] += 1

            temporal_info_dict = {
                "temporal_tdp_occ": temporal_tdp_occ,
                "temporal_tdp_lifetime": temporal_tdp_lifetime,
                "temporal_tdp_D": temporal_tdp_D,
                "temporal_tdp_a": temporal_tdp_a,
                "temporal_trans": temporal_trans,
            }
            pickle.dump(temporal_info_dict, open(savename, "wb"))
        elif the_data_is == "3D":
            print("tTDP 3D not implemented")

    return temporal_info_dict


def compute_tTDP_FP_lists(
    subpaths,
    channels=[],
    bio_repli=0,
    D="Insulin/",
    num_transistions=10,
    num_difftypes=4,
):
    temporal_tdp_occ_list = []
    temporal_trans_list = []
    temporal_tdp_lifetime_list = []
    temporal_tdp_D_list = []
    temporal_tdp_a_list = []
    for i in range(len(subpaths)):
        datapath = D + subpaths[i]
        file = channels[i] + "/bg_corr_all_tracked" + str(bio_repli) + ".csv"
        exp_name = (
            subpaths[i].replace("/", "_") + file.split("/")[0] + "_" + str(bio_repli)
        )
        savename = datapath + "" + "temporal_info_dict_" + exp_name + ".pkl"

        if os.path.exists(datapath + "temporal_info_dict_" + exp_name + ".pkl"):
            savename = datapath + "temporal_info_dict_" + exp_name + ".pkl"
            temporal_info_dict = pickle.load(open(savename, "rb"))

            temporal_tdp_occ = temporal_info_dict["temporal_tdp_occ"]
            temporal_trans = temporal_info_dict["temporal_trans"]

            temporal_tdp_lifetime = temporal_info_dict["temporal_tdp_lifetime"]
            temporal_tdp_D = temporal_info_dict["temporal_tdp_D"]
            temporal_tdp_a = temporal_info_dict["temporal_tdp_a"]

            temporal_tdp_occ_list.append(
                temporal_tdp_occ[:, :num_transistions] / np.sum(temporal_tdp_occ[:, 0])
            )
            temporal_trans_list.append(
                temporal_trans[:num_transistions]
                / np.sum(temporal_trans[:num_transistions], axis=2).reshape(
                    num_transistions, num_difftypes, 1
                )
            )
            temporal_tdp_lifetime_list.append(
                temporal_tdp_lifetime[:, :num_transistions]
                / temporal_tdp_occ[:, :num_transistions]
            )
            temporal_tdp_D_list.append(temporal_tdp_D[:, :num_transistions])
            temporal_tdp_a_list.append(temporal_tdp_a[:, :num_transistions])

    return (
        temporal_tdp_occ_list,
        temporal_trans_list,
        temporal_tdp_lifetime_list,
        temporal_tdp_D_list,
        temporal_tdp_a_list,
    )


def load_3D_data(
    file,
    SEARCH_PATTERN,
    OUTPUT_NAME,
    min_trace_length,
    min_brightness,
    max_brightness,
    dont_use=[],
):
    # Turn matlab df into useful numpy arrays
    df, seqOfEvents_list, original_idx = matlabloader(
        name=file, input=SEARCH_PATTERN, output=OUTPUT_NAME, dont_use=dont_use
    )
    df, keep_idx_len = remove_short_traces3D(df, threshold=min_trace_length)
    df, keep_idx_dim = remove_dim_traces3D(
        df, threshold=min_brightness
    )  # start value larger than this
    df, keep_idx_bri = remove_overlybright_traces3D(
        df, threshold=max_brightness
    )  # start value lower than this
    # pickle.dump(df, open(OUTPUT_NAME.format(file), 'wb'))
    return df, seqOfEvents_list, keep_idx_len, keep_idx_dim, keep_idx_bri, original_idx


def find_timepointchange(arr):
    cp = [0]
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            cp.append(i)
    cp = cp + [len(arr)]
    return cp


def handle_compound_tracks(
    compound_idx,
    seqOfEvents_list,
    tracks,
    timepoints,
    frames,
    track_ids,
    amplitudes,
    amplitudes_S,
    amplitudes_SM,
    amplitudes_sig,
    amplitudes_bg,
    min_len_threshold,
    min_bright,
):

    def transform_list(input_list):
        unique_values = sorted(set(input_list))
        value_map = {value: index for index, value in enumerate(unique_values)}

        transformed_list = [value_map[x] for x in input_list]

        return transformed_list, value_map

    fission0_concat1_fusion2_all = []
    changepoints_all = []
    for ci in compound_idx:
        seqOfEvents_list_ci = seqOfEvents_list[ci] - 1
        seqOfEvents_list_ci[2, :], value_map = transform_list(seqOfEvents_list_ci[2, :])

        non_nan_idx_list = np.array(range(len(seqOfEvents_list_ci[3, :])))[
            ~np.isnan(seqOfEvents_list_ci[3, :])
        ]
        for non_nan_idx in non_nan_idx_list:
            if seqOfEvents_list_ci[3, non_nan_idx] not in value_map.keys():
                seqOfEvents_list_ci[3, non_nan_idx] = np.nan
            else:
                seqOfEvents_list_ci[3, non_nan_idx] = int(
                    value_map[seqOfEvents_list_ci[3, non_nan_idx]]
                )

        cps = find_timepointchange(timepoints[ci])
        all_idx = np.array(range(len(timepoints[ci])))
        idx_list = []
        changepoints_all = []

        idx_list = []

        off = 0
        comp_idx = []
        frames_ci = []
        for i in range(len(cps) - 1):
            if i > 0:
                off = 1
            frames_ci.append(frames[ci][cps[i] + off : cps[i + 1]])
            comp_idx.append(np.array(all_idx[cps[i] + off : cps[i + 1]]))
        comp_idx = np.array(comp_idx, dtype=object)
        idx_list.append(comp_idx[0])
        for i, seqEv in enumerate(seqOfEvents_list_ci[3]):
            if seqEv >= 0:
                tracknum_doing_something = int(seqOfEvents_list_ci[2, i])
                tracknum_victim = int(seqOfEvents_list_ci[3, i])
                type_of_event = int(seqOfEvents_list_ci[1, i])  # 1: split 2: merge
                comp_idx_doing = comp_idx[tracknum_doing_something].astype(int)
                comp_idx_victim = comp_idx[tracknum_victim]

                try:
                    frame_of_event = list(frames_ci[tracknum_victim]).index(
                        int(seqOfEvents_list_ci[0, i])
                    )
                except:
                    print("frame not found CME error continue")
                    if (
                        len(tracks[ci][comp_idx_doing]) >= min_len_threshold
                        and amplitudes[ci][comp_idx_doing][0, 0] >= min_bright
                    ):
                        idx_list.append(comp_idx_doing)
                    continue

                if type_of_event == 0:
                    idx = np.hstack([comp_idx_victim[:frame_of_event], comp_idx_doing])
                if type_of_event == 1:
                    idx = np.hstack([comp_idx_doing, comp_idx_victim[frame_of_event:]])
                idx_list.append(idx)

        for i, idx in enumerate(idx_list):
            idx = idx.astype(int)
            if (
                len(tracks[ci][idx]) >= min_len_threshold
                and amplitudes[ci][idx][0, 0] >= min_bright
            ):
                if np.unique(frames[ci][idx], return_counts=True)[1].max() > 1:
                    print(tracks[ci])
                    print(frames[ci])
                    print(frames[ci][idx])
                    print(seqOfEvents_list_ci)
                    print(i, "frames not unique")
                    print(sdsdf)
                tracks.append(tracks[ci][idx])
                timepoints.append(timepoints[ci][idx])
                frames.append(frames[ci][idx])
                amplitudes.append(amplitudes[ci][idx])
                amplitudes_S.append(amplitudes_S[ci][idx])
                amplitudes_SM.append(amplitudes_SM[ci][idx])
                amplitudes_sig.append(amplitudes_sig[ci][idx])
                amplitudes_bg.append(amplitudes_bg[ci][idx])
                track_ids.append(track_ids[ci] + "_c" + str(i))

    for ci in sorted(compound_idx, reverse=True):
        tracks.pop(ci)
        timepoints.pop(ci)
        frames.pop(ci)
        amplitudes.pop(ci)
        amplitudes_S.pop(ci)
        amplitudes_SM.pop(ci)
        amplitudes_sig.pop(ci)
        amplitudes_bg.pop(ci)
        track_ids.pop(ci)

    return (
        tracks,
        timepoints,
        frames,
        amplitudes,
        amplitudes_S,
        amplitudes_SM,
        amplitudes_sig,
        amplitudes_bg,
        track_ids,
        fission0_concat1_fusion2_all,
        changepoints_all,
    )


def fuse_tracks(
    tracks,
    timepoints,
    frames,
    track_ids,
    amplitudes,
    amplitudes_S,
    amplitudes_SM,
    amplitudes_sig,
    amplitudes_bg,
    min_trace_length,
    min_brightness,
    blinking_forgiveness=1,
    factor=6,
):

    def within_ellipsoid(startcoord, endcoord, xyr=0.1, zr=0.25):
        startcoord, endcoord = np.array(startcoord), np.array(endcoord)
        x, y, z = startcoord - endcoord
        return 1 >= (x / xyr) ** 2 + (y / xyr) ** 2 + (z / zr) ** 2

    def squaredist(startcoord, endcoord):
        startcoord, endcoord = np.array(startcoord), np.array(endcoord)
        return np.sqrt(np.sum((startcoord - endcoord) ** 2))

    dt = timepoints[0][1] - timepoints[0][0]
    starttimes = [t[0] for t in timepoints]
    endtimes = [t[-1] for t in timepoints]
    startcoords = [t[0] for t in tracks]
    endcoords = [t[-1] for t in tracks]

    index_list = []
    dist_list = []
    for e, et in tqdm(enumerate(endtimes)):
        t = tracks[e]
        SL1 = np.sqrt(np.sum((t[1:] - t[:-1]) ** 2, axis=1))
        N1 = len(SL1)
        corr = gamma(N1) * np.sqrt(N1) / gamma(N1 + 0.5)
        shape_param1 = corr * np.sqrt((np.sum(SL1**2) / (2 * N1)))

        for s, st in enumerate(starttimes):
            if e == s:
                continue

            t = tracks[s]
            SL2 = np.sqrt(np.sum((t[1:] - t[:-1]) ** 2, axis=1))
            N2 = len(SL2)
            corr = gamma(N2) * np.sqrt(N2) / gamma(N2 + 0.5)
            shape_param2 = corr * np.sqrt((np.sum(SL2**2) / (2 * N2)))
            max_allowed1 = shape_param1 * factor + 1.253 * shape_param1
            max_allowed2 = shape_param2 * factor + 1.253 * shape_param2
            xyr = np.max([max_allowed1, max_allowed2])
            zr = xyr

            if np.round(st - et, 2) == blinking_forgiveness * np.round(dt, 2):
                if within_ellipsoid(startcoords[s], endcoords[e], xyr=xyr, zr=zr):
                    index_list.append([e, s])
                    dist_list.append(squaredist(startcoords[s], endcoords[e]))

    for e, et in tqdm(enumerate(endtimes)):
        if len(index_list) > 0:
            if e not in np.concatenate(index_list):
                index_list.append([e, -10])
        else:
            index_list.append([e, -10])

    if len(index_list) > 0:
        index_list = np.vstack(index_list)
        dist_list = np.array(dist_list)

        glued_tracks = []
        glued_timepoints = []
        glued_frames = []
        glued_track_ids = index_list
        glued_amplitudes = []
        glued_amplitudes_S = []
        glued_amplitudes_SM = []
        glued_amplitudes_sig = []
        glued_amplitudes_bg = []
        for idx in index_list:
            if idx[1] == -10:
                glued_tracks.append(tracks[idx[0]])
                glued_timepoints.append(timepoints[idx[0]])
                glued_frames.append(frames[idx[0]])
                glued_amplitudes.append(amplitudes[idx[0]])
                glued_amplitudes_S.append(amplitudes_S[idx[0]])
                glued_amplitudes_SM.append(amplitudes_SM[idx[0]])
                glued_amplitudes_sig.append(amplitudes_sig[idx[0]])
                glued_amplitudes_bg.append(amplitudes_bg[idx[0]])

            else:
                glued_tracks.append(np.vstack([tracks[idx[0]], tracks[idx[1]]]))
                glued_timepoints.append(
                    np.hstack([timepoints[idx[0]], timepoints[idx[1]]])
                )
                glued_frames.append(np.hstack([frames[idx[0]], frames[idx[1]]]))
                glued_amplitudes.append(
                    np.vstack([amplitudes[idx[0]], amplitudes[idx[1]]])
                )
                glued_amplitudes_S.append(
                    np.vstack([amplitudes_S[idx[0]], amplitudes_S[idx[1]]])
                )
                glued_amplitudes_SM.append(
                    np.vstack([amplitudes_SM[idx[0]], amplitudes_SM[idx[1]]])
                )
                glued_amplitudes_sig.append(
                    np.vstack([amplitudes_sig[idx[0]], amplitudes_sig[idx[1]]])
                )
                glued_amplitudes_bg.append(
                    np.vstack([amplitudes_bg[idx[0]], amplitudes_bg[idx[1]]])
                )

        glued_tracks = np.array(glued_tracks, dtype=object)
        glued_timepoints = np.array(glued_timepoints, dtype=object)
        glued_frames = np.array(glued_frames, dtype=object)
        glued_track_ids = np.array(glued_track_ids, dtype=object)
        glued_amplitudes = np.array(glued_amplitudes, dtype=object)
        glued_amplitudes_S = np.array(glued_amplitudes_S, dtype=object)
        glued_amplitudes_SM = np.array(glued_amplitudes_SM, dtype=object)
        glued_amplitudes_sig = np.array(glued_amplitudes_sig, dtype=object)
        glued_amplitudes_bg = np.array(glued_amplitudes_bg, dtype=object)

        for k, tp in enumerate(glued_timepoints):
            assert np.mean((tp[1:] - tp[:-1]) > 0) == 1, print(tp)

        lengths = np.array([len(t) for t in glued_tracks])
        len_filter = lengths > min_trace_length

        initial_bright = np.array(
            [glued_amplitudes[i][0, 0] for i in range(len(glued_tracks))]
        )
        brightness_filter = initial_bright > min_brightness

        FILTER = len_filter * brightness_filter

        index_list = index_list[FILTER]
        glued_tracks = glued_tracks[FILTER]
        glued_timepoints = glued_timepoints[FILTER]
        glued_frames = glued_frames[FILTER]
        glued_track_ids = glued_track_ids[FILTER]
        glued_amplitudes = glued_amplitudes[FILTER]
        glued_amplitudes_S = glued_amplitudes_S[FILTER]
        glued_amplitudes_SM = glued_amplitudes_SM[FILTER]
        glued_amplitudes_sig = glued_amplitudes_sig[FILTER]
        glued_amplitudes_bg = glued_amplitudes_bg[FILTER]

        sorted_idx = np.argsort(index_list[:, 0])
        return (
            glued_tracks[sorted_idx],
            glued_timepoints[sorted_idx],
            glued_frames[sorted_idx],
            glued_track_ids[sorted_idx],
            glued_amplitudes[sorted_idx],
            glued_amplitudes_S[sorted_idx],
            glued_amplitudes_SM[sorted_idx],
            glued_amplitudes_sig[sorted_idx],
            glued_amplitudes_bg[sorted_idx],
            index_list[sorted_idx],
        )
    else:
        glued_tracks = tracks.copy()
        glued_timepoints = timepoints.copy()
        glued_frames = frames.copy()
        glued_track_ids = track_ids.copy()
        glued_amplitudes = amplitudes.copy()
        glued_amplitudes_S = amplitudes_S.copy()
        glued_amplitudes_SM = amplitudes_SM.copy()
        glued_amplitudes_sig = amplitudes_sig.copy()
        glued_amplitudes_bg = amplitudes_bg.copy()

        return (
            glued_tracks[sorted_idx],
            glued_timepoints[sorted_idx],
            glued_frames[sorted_idx],
            glued_track_ids[sorted_idx],
            glued_amplitudes[sorted_idx],
            glued_amplitudes_S[sorted_idx],
            glued_amplitudes_SM[sorted_idx],
            glued_amplitudes_sig[sorted_idx],
            glued_amplitudes_bg[sorted_idx],
            index_list[sorted_idx],
        )


def read_3Dtracks(
    PN,
    SEARCH_PATTERN,
    OUTPUT_NAME,
    min_prefusetrace_length,
    min_prefusebrightness,
    max_brightness,
    min_trace_length,
    min_brightness,
    xy_to_um,
    z_to_um,
):

    df = load_3D_data(
        PN,
        SEARCH_PATTERN,
        OUTPUT_NAME,
        min_prefusetrace_length,
        min_prefusebrightness,
        max_brightness,
    )

    tracks, timepoints, track_ids, amplitudes, amplitudes_sig, amplitudes_bg, catIdx = (
        curate_3D_data_to_tracks(df, xy_to_um, z_to_um)
    )
    compound_idx = np.array(list(range(len(tracks))))[catIdx > 4]

    out = handle_compound_tracks(
        compound_idx,
        tracks,
        timepoints,
        track_ids,
        amplitudes,
        amplitudes_sig,
        amplitudes_bg,
        min_prefusetrace_length,
        min_prefusebrightness,
    )
    tracks, timepoints, track_ids, amplitudes, amplitudes_sig, amplitudes_bg = out

    out = fuse_tracks(
        tracks,
        timepoints,
        track_ids,
        amplitudes,
        amplitudes_sig,
        amplitudes_bg,
        min_trace_length,
        min_brightness,
        blinking_forgiveness=1,
    )
    tracks, timepoints, track_ids, amplitudes, amplitudes_sig, amplitudes_bg = out

    return tracks, timepoints


def handle_compound_tracks_slimversion(
    compound_idx, tracks, timepoints, track_ids, min_len_threshold=5
):
    for ci in compound_idx:
        cps = find_timepointchange(timepoints[ci])
        all_idx = np.array(range(len(timepoints[ci])))
        tp_list = []
        idx_list = []
        for i, cp in enumerate(cps[:-1]):
            if i == 0:
                tp_list.append(timepoints[ci][:cp])
                idx = all_idx[:cp]
                idx_list.append(idx)
                if (
                    len(tracks[ci][idx]) >= min_len_threshold
                    and amplitudes[ci][idx][0, 0] >= min_bright
                ):
                    tracks.append(tracks[ci][idx])
                    timepoints.append(timepoints[ci][idx])
                    track_ids.append(track_ids[ci] + "_c" + str(i))
            else:
                tp = timepoints[ci][cp : cps[i + 1]]
                idx = all_idx[cp : cps[i + 1]]

                together_idx = ~np.isin(tp_list[i - 1], tp)
                if together_idx[-1] == False:  # fission
                    tp = np.hstack([tp_list[i - 1][together_idx], tp])
                    idx = np.hstack([idx_list[i - 1][together_idx], idx])
                else:  # fusion
                    fuse_idx = (
                        idx_list[i - 1][~together_idx][-1] + 1
                    )  # because python counting and 0
                    tp = np.hstack([tp, tp_list[i - 1][fuse_idx]])
                    idx = np.hstack([idx, idx_list[i - 1][fuse_idx]])
                tp_list.append(tp)
                idx_list.append(idx)
                assert (
                    len(find_timepointchange(tp)) == 1
                ), f"{find_timepointchange(tp)} {tp}"

                if (
                    len(tracks[ci][idx]) >= min_len_threshold
                    and amplitudes[ci][idx][0, 0] >= min_bright
                ):
                    tracks.append(tracks[ci][idx])
                    timepoints.append(timepoints[ci][idx])
                    track_ids.append(track_ids[ci] + "_c" + str(i))

    for ci in sorted(compound_idx, reverse=True):
        tracks.pop(ci)
        timepoints.pop(ci)
        track_ids.pop(ci)

    return tracks, timepoints, track_ids


def curate_3D_data_to_tracks(df, xy_to_um, z_to_um):
    dfg = df.groupby("id", sort=False)
    (
        tracks,
        timepoints,
        frames,
        track_ids,
        amplitudes,
        amplitudes_S,
        amplitudes_SM,
        amplitudes_sig,
        amplitudes_bg,
        catIdx,
    ) = convert_matlabDF_to_arrays(dfg, xy_to_um, z_to_um)
    frames = [f - 1 for f in frames]  # because python counting
    return (
        tracks,
        timepoints,
        frames,
        track_ids,
        amplitudes,
        amplitudes_S,
        amplitudes_SM,
        amplitudes_sig,
        amplitudes_bg,
        catIdx,
    )


def tTDP_col_names_generator(max_change_points, add_FP):
    difftypes_abre = ["ND", "DM", "CD", "SD"]

    seq_difftype_names = []
    seg_lifetime_names = []
    seg_D_names = []
    seg_a_names = []
    seg_eff_names = []
    seg_FD_names = []
    seg_Gau_names = []
    seg_Kur_names = []
    seg_msdfrac_names = []
    seg_trap_names = []
    seg_SL_names = []
    seg_msd_names = []
    seg_dp_names = []
    seg_corr_dp_names = []
    seg_avg_sign_dp_names = []
    seg_sumSL_names = []
    seg_minSL_names = []
    seg_maxSL_names = []
    seg_maxminSL_names = []
    seg_speed_names = []
    seg_CV_names = []
    seg_arrest_names = []
    seg_fast_names = []
    seg_volume_names = []
    for i in range(max_change_points):
        for difftyp in difftypes_abre:
            seq_difftype_names.append("s" + str(i) + "_" + difftyp + "_difftype")
            seg_lifetime_names.append("s" + str(i) + "_" + difftyp + "_lifetime")
            seg_D_names.append("s" + str(i) + "_" + difftyp + "_D")
            seg_a_names.append("s" + str(i) + "_" + difftyp + "_a")
            seg_eff_names.append("s" + str(i) + "_" + difftyp + "_eff")
            seg_FD_names.append("s" + str(i) + "_" + difftyp + "_FD")
            seg_Gau_names.append("s" + str(i) + "_" + difftyp + "_Gau")
            seg_Kur_names.append("s" + str(i) + "_" + difftyp + "_Kur")
            seg_msdfrac_names.append("s" + str(i) + "_" + difftyp + "_msdfrac")
            seg_trap_names.append("s" + str(i) + "_" + difftyp + "_trap")
            seg_SL_names.append("s" + str(i) + "_" + difftyp + "_SL")
            seg_msd_names.append("s" + str(i) + "_" + difftyp + "_msd")
            seg_dp_names.append("s" + str(i) + "_" + difftyp + "_DP")
            seg_corr_dp_names.append("s" + str(i) + "_" + difftyp + "_corr_DP")
            seg_avg_sign_dp_names.append("s" + str(i) + "_" + difftyp + "_avg_sign_DP")
            seg_sumSL_names.append("s" + str(i) + "_" + difftyp + "_sumSL")
            seg_minSL_names.append("s" + str(i) + "_" + difftyp + "_minSL")
            seg_maxSL_names.append("s" + str(i) + "_" + difftyp + "_maxSL")
            seg_maxminSL_names.append("s" + str(i) + "_" + difftyp + "_SL_broadness")
            seg_speed_names.append("s" + str(i) + "_" + difftyp + "_speed")
            seg_CV_names.append("s" + str(i) + "_" + difftyp + "_CV")
            seg_arrest_names.append("s" + str(i) + "_" + difftyp + "_arrested_fraction")
            seg_fast_names.append("s" + str(i) + "_" + difftyp + "_fast_fraction")
            seg_volume_names.append("s" + str(i) + "_" + difftyp + "_volume")

    seq_trans_names = []
    for i in range(max_change_points):
        for s_difftyp in difftypes_abre:
            for e_difftyp in difftypes_abre:
                seq_trans_names.append("s" + str(i) + "_" + s_difftyp + "_" + e_difftyp)
    if add_FP:
        tTDP_col_names = (
            seq_difftype_names
            + seq_trans_names
            + seg_lifetime_names
            + seg_D_names
            + seg_a_names
            + seg_eff_names
            + seg_FD_names
            + seg_Gau_names
            + seg_Kur_names
            + seg_msdfrac_names
            + seg_trap_names
            + seg_SL_names
            + seg_msd_names
            + seg_dp_names
            + seg_corr_dp_names
            + seg_avg_sign_dp_names
            + seg_sumSL_names
            + seg_minSL_names
            + seg_maxSL_names
            + seg_maxminSL_names
            + seg_speed_names
            + seg_CV_names
            + seg_arrest_names
            + seg_fast_names
            + seg_volume_names
        )
    else:
        tTDP_col_names = seq_difftype_names + seq_trans_names + seg_lifetime_names

    return (
        tTDP_col_names,
        seq_difftype_names,
        seg_lifetime_names,
        seq_trans_names,
        seg_D_names,
        seg_a_names,
        seg_eff_names,
        seg_FD_names,
        seg_Gau_names,
        seg_Kur_names,
        seg_msdfrac_names,
        seg_trap_names,
        seg_SL_names,
        seg_msd_names,
        seg_dp_names,
        seg_corr_dp_names,
        seg_avg_sign_dp_names,
        seg_sumSL_names,
        seg_minSL_names,
        seg_maxSL_names,
        seg_maxminSL_names,
        seg_speed_names,
        seg_CV_names,
        seg_arrest_names,
        seg_fast_names,
        seg_volume_names,
    )


def tTDP_individuals_generator(
    ensemble_pred,
    tracks,
    max_change_points,
    num_difftypes,
    fp_datapath,
    hmm_filename,
    dim=3,
    dt=2,
    min_seglen_for_FP=5,
    add_FP=True,
):
    difftypes = {"0": "Normal", "1": "Directed", "2": "Confined", "3": "Subdiffusive"}

    tTDP_individuals = []
    for i in tqdm(range(len(ensemble_pred))):
        _, pred_changepoints, pred_diffs = find_segments(ensemble_pred[i])
        seg_diff_frac_ = np.zeros((num_difftypes, max_change_points))
        seg_lifetime_ = np.zeros((num_difftypes, max_change_points))
        seg_D = np.zeros((num_difftypes, max_change_points))
        seg_a = np.zeros((num_difftypes, max_change_points))

        seg_eff = np.zeros((num_difftypes, max_change_points))
        seg_FD = np.zeros((num_difftypes, max_change_points))
        seg_Gau = np.zeros((num_difftypes, max_change_points))
        seg_Kur = np.zeros((num_difftypes, max_change_points))
        seg_msdfrac = np.zeros((num_difftypes, max_change_points))
        seg_trap = np.zeros((num_difftypes, max_change_points))
        seg_SL = np.zeros((num_difftypes, max_change_points))
        seg_msd = np.zeros((num_difftypes, max_change_points))
        seg_dp = np.zeros((num_difftypes, max_change_points))

        seg_extra = np.zeros((num_difftypes, max_change_points))
        seg_pval = np.zeros((num_difftypes, max_change_points))

        seg_t0 = np.zeros((num_difftypes, max_change_points))
        seg_t1 = np.zeros((num_difftypes, max_change_points))
        seg_t2 = np.zeros((num_difftypes, max_change_points))
        seg_t3 = np.zeros((num_difftypes, max_change_points))

        seg_corr_dp = np.zeros((num_difftypes, max_change_points))
        seg_avg_sign_dp = np.zeros((num_difftypes, max_change_points))
        seg_sumSL = np.zeros((num_difftypes, max_change_points))
        seg_minSL = np.zeros((num_difftypes, max_change_points))
        seg_maxSL = np.zeros((num_difftypes, max_change_points))
        seg_maxminSL = np.zeros((num_difftypes, max_change_points))
        seg_speed = np.zeros((num_difftypes, max_change_points))
        seg_CV = np.zeros((num_difftypes, max_change_points))
        seg_arrest = np.zeros((num_difftypes, max_change_points))
        seg_fast = np.zeros((num_difftypes, max_change_points))
        seg_volume = np.zeros((num_difftypes, max_change_points))

        seg_trans = np.zeros((max_change_points, num_difftypes, num_difftypes))

        for idx in range(max_change_points):
            if idx >= len(pred_changepoints) - 1:
                continue
            segment = ensemble_pred[i][
                pred_changepoints[idx] : pred_changepoints[idx + 1]
            ]
            segment_xyz = tracks[i][pred_changepoints[idx] : pred_changepoints[idx + 1]]
            behavior = np.unique(segment)[0]
            lifetime = len(segment)
            seg_diff_frac_[behavior, idx] += 1
            seg_lifetime_[behavior, idx] += lifetime

            if len(segment_xyz) >= min_seglen_for_FP:
                difftype = difftypes[str(np.unique(segment)[0])]
                param = create_fingerprint_track(
                    segment_xyz, fp_datapath, hmm_filename, dim, dt, difftype
                )
                seg_a[behavior, idx] = param[0]
                seg_D[behavior, idx] = param[1]
                seg_extra[behavior, idx] = param[2]
                seg_pval[behavior, idx] = param[3]
                seg_eff[behavior, idx] = param[5]
                seg_FD[behavior, idx] = param[6]
                seg_Gau[behavior, idx] = param[7]
                seg_Kur[behavior, idx] = param[8]
                seg_msdfrac[behavior, idx] = param[9]
                seg_trap[behavior, idx] = param[10]

                seg_t0[behavior, idx] = param[11]
                seg_t1[behavior, idx] = param[12]
                seg_t2[behavior, idx] = param[13]
                seg_t3[behavior, idx] = param[14]

                seg_SL[behavior, idx] = param[17]
                seg_msd[behavior, idx] = param[18]
                seg_dp[behavior, idx] = param[19]
                seg_corr_dp[behavior, idx] = param[20]
                seg_avg_sign_dp[behavior, idx] = param[21]
                seg_sumSL[behavior, idx] = param[22]
                seg_minSL[behavior, idx] = param[23]
                seg_maxSL[behavior, idx] = param[24]
                seg_maxminSL[behavior, idx] = param[25]
                seg_speed[behavior, idx] = param[26]
                seg_CV[behavior, idx] = param[27]
                seg_arrest[behavior, idx] = param[28]
                seg_fast[behavior, idx] = param[29]
                seg_volume[behavior, idx] = param[30]

        for idx in range(max_change_points):
            if len(pred_diffs) == 1 and idx == 0:
                seg_trans[idx, pred_diffs[idx], pred_diffs[idx]] += 1
            if idx >= len(pred_diffs) - 1:
                continue
            else:
                seg_trans[idx, pred_diffs[idx], pred_diffs[idx + 1]] += 1

        tTDP_i = np.hstack(
            [
                np.hstack([seg_diff_frac_[:, col] for col in range(max_change_points)]),
                seg_trans.reshape(-1),
                np.hstack([seg_lifetime_[:, col] for col in range(max_change_points)]),
                np.hstack([seg_D[:, col] for col in range(max_change_points)]),
                np.hstack([seg_a[:, col] for col in range(max_change_points)]),
                np.hstack([seg_pval[:, col] for col in range(max_change_points)]),
                np.hstack([seg_extra[:, col] for col in range(max_change_points)]),
                np.hstack([seg_eff[:, col] for col in range(max_change_points)]),
                np.hstack([seg_FD[:, col] for col in range(max_change_points)]),
                np.hstack([seg_Gau[:, col] for col in range(max_change_points)]),
                np.hstack([seg_Kur[:, col] for col in range(max_change_points)]),
                np.hstack([seg_msdfrac[:, col] for col in range(max_change_points)]),
                np.hstack([seg_trap[:, col] for col in range(max_change_points)]),
                np.hstack([seg_t0[:, col] for col in range(max_change_points)]),
                np.hstack([seg_t1[:, col] for col in range(max_change_points)]),
                np.hstack([seg_t2[:, col] for col in range(max_change_points)]),
                np.hstack([seg_t3[:, col] for col in range(max_change_points)]),
                np.hstack([seg_SL[:, col] for col in range(max_change_points)]),
                np.hstack([seg_msd[:, col] for col in range(max_change_points)]),
                np.hstack([seg_dp[:, col] for col in range(max_change_points)]),
                np.hstack([seg_corr_dp[:, col] for col in range(max_change_points)]),
                np.hstack(
                    [seg_avg_sign_dp[:, col] for col in range(max_change_points)]
                ),
                np.hstack([seg_sumSL[:, col] for col in range(max_change_points)]),
                np.hstack([seg_minSL[:, col] for col in range(max_change_points)]),
                np.hstack([seg_maxSL[:, col] for col in range(max_change_points)]),
                np.hstack([seg_maxminSL[:, col] for col in range(max_change_points)]),
                np.hstack([seg_speed[:, col] for col in range(max_change_points)]),
                np.hstack([seg_CV[:, col] for col in range(max_change_points)]),
                np.hstack([seg_arrest[:, col] for col in range(max_change_points)]),
                np.hstack([seg_fast[:, col] for col in range(max_change_points)]),
                np.hstack([seg_volume[:, col] for col in range(max_change_points)]),
            ]
        )

        tTDP_individuals.append(tTDP_i)
    return tTDP_individuals


def postprocess_pred(ens_pred_raw, ens_score_raw, min_tracklet_len):
    ens_pred, ens_score = ens_pred_raw.copy(), ens_score_raw.copy()
    pred_seglens, pred_changepoints, _ = find_segments(ens_pred)

    offset = 0
    for idx in range(len(pred_seglens)):
        idx = idx + offset
        if idx >= len(pred_changepoints) - 1:
            continue
        start, end = pred_changepoints[idx], pred_changepoints[idx + 1]
        tracklet_score = np.mean(ens_score, axis=1)
        if pred_seglens[idx] < min_tracklet_len:
            right_idx = np.min([idx + 1, len(pred_seglens) - 1])
            start_right, end_right = (
                pred_changepoints[right_idx],
                pred_changepoints[right_idx + 1],
            )
            ens_pred_right = ens_pred[start_right:end_right]
            label_right = np.unique(ens_pred_right)

            left_idx = np.max([idx - 1, 0])
            start_left, end_left = (
                pred_changepoints[left_idx],
                pred_changepoints[left_idx + 1],
            )
            ens_pred_left = ens_pred[start_left:end_left]
            label_left = np.unique(ens_pred_left)

            if (
                len(ens_pred_left) >= min_tracklet_len
                and len(ens_pred_right) >= min_tracklet_len
            ):
                offset -= 1
                ens_pred[start:end] = (
                    label_right
                    if tracklet_score[label_right] > tracklet_score[label_left]
                    else label_left
                )
            elif len(ens_pred_left) >= min_tracklet_len:
                offset -= 1
                ens_pred[start:end] = label_left
            elif len(ens_pred_right) >= min_tracklet_len:
                offset -= 1
                ens_pred[start:end] = label_right
            else:
                offset -= 1
                for idx_nextright in range(idx + 2, len(pred_seglens)):
                    start_nextright, end_nextright = (
                        pred_changepoints[idx_nextright],
                        pred_changepoints[idx_nextright + 1],
                    )
                    ens_pred_nextright = ens_pred[start_nextright:end_nextright]
                    label_nextright = np.unique(ens_pred_nextright)
                    if len(ens_pred_nextright) >= min_tracklet_len:
                        ens_pred[start:start_nextright] = label_nextright
                        break
        pred_seglens, pred_changepoints, _ = find_segments(ens_pred)

    return ens_pred


def tTDP_individuals_generator_v2(
    ensemble_pred,
    tracks,
    max_change_points,
    num_difftypes,
    fp_datapath,
    hmm_filename,
    dim=3,
    dt=2,
    min_seglen_for_FP=5,
    add_FP=True,
):
    difftypes = {"0": "Normal", "1": "Directed", "2": "Confined", "3": "Subdiffusive"}

    tTDP_individuals = []
    for i in tqdm(range(len(ensemble_pred))):
        _, pred_changepoints, pred_diffs = find_segments(ensemble_pred[i])
        seg_diff_frac_ = np.ones(max_change_points) * 4
        seg_lifetime_ = np.zeros(max_change_points)
        seg_D = np.zeros(max_change_points)
        seg_a = np.zeros(max_change_points)

        seg_eff = np.zeros(max_change_points)
        seg_FD = np.zeros(max_change_points)
        seg_Gau = np.zeros(max_change_points)
        seg_Kur = np.zeros(max_change_points)
        seg_msdfrac = np.zeros(max_change_points)
        seg_trap = np.zeros(max_change_points)
        seg_SL = np.zeros(max_change_points)
        seg_msd = np.zeros(max_change_points)
        seg_dp = np.zeros(max_change_points)
        seg_t0 = np.zeros(max_change_points)
        seg_t1 = np.zeros(max_change_points)
        seg_t2 = np.zeros(max_change_points)
        seg_t3 = np.zeros(max_change_points)

        seg_trans = np.ones(max_change_points) * 4

        for idx in range(max_change_points):
            if idx >= len(pred_changepoints) - 1:
                continue
            segment = ensemble_pred[i][
                pred_changepoints[idx] : pred_changepoints[idx + 1]
            ]
            segment_xyz = tracks[i][pred_changepoints[idx] : pred_changepoints[idx + 1]]
            behavior = np.unique(segment)[0]
            lifetime = len(segment)
            seg_diff_frac_[idx] = behavior
            seg_lifetime_[idx] = lifetime

            if len(segment_xyz) >= min_seglen_for_FP:
                difftype = difftypes[str(np.unique(segment)[0])]
                param = create_fingerprint_track(
                    segment_xyz, fp_datapath, hmm_filename, dim, dt, difftype
                )
                seg_a[idx] = param[0]
                seg_D[idx] = param[1]
                seg_eff[idx] = param[4]
                seg_FD[idx] = param[6]
                seg_Gau[idx] = param[7]
                seg_Kur[idx] = param[8]
                seg_msdfrac[idx] = param[9]
                seg_trap[idx] = param[10]
                seg_t0[idx] = param[11]
                seg_t1[idx] = param[12]
                seg_t2[idx] = param[13]
                seg_t3[idx] = param[14]
                seg_SL[idx] = param[17]
                seg_msd[idx] = param[18]
                seg_dp[idx] = param[19]

        tTDP_i = np.hstack(
            [
                seg_diff_frac_,
                seg_lifetime_,
                seg_D,
                seg_a,
                seg_eff,
                seg_FD,
                seg_Gau,
                seg_Kur,
                seg_msdfrac,
                seg_trap,
                seg_SL,
                seg_msd,
                seg_dp,
                seg_t0,
                seg_t1,
                seg_t2,
                seg_t3,
            ]
        )
        tTDP_individuals.append(tTDP_i)
    return tTDP_individuals


def FP_xlabel_maker(feature):
    if "_lifetime" in feature:
        return "Track duration"
    if "_D" in feature and "P" not in feature:
        return "Diffusion coefficient (um^2/s)"
    if "_a" in feature:
        return "Alpha"
    if "_eff" in feature:
        return "Track Efficiency"
    if "_FD" in feature:
        return "Fractal Dimensions"
    if "_Gau" in feature:
        return "Track Gaussianity"
    if "_Kur" in feature:
        return "Track Kurtosis"
    if "_msdfrac" in feature:
        return "MSD ratio"
    if "_trap" in feature:
        return "Trappedness"
    if "_SL" in feature:
        return "Steplengths (um)"
    if "_msd" in feature:
        return "Mean squared displacement (um^2)"
    if "_DP" in feature:
        return "Dotproduct"
    if "_corr_DP" in feature:
        return "Fraction correlated sign of dotproduct"
    if "_avg_sign_DP" in feature:
        return "Fraction positive dotproduct"
    if "_sumSL" in feature:
        return "Cumulative Steplengths (um)"
    if "_minSL" in feature:
        return "Min Steplengths (um)"
    if "_maxSL" in feature:
        return "Max Steplengths (um)"
    if "_maxminSL" in feature:
        return "Steplength Broadness (um)"
    if "_speed" in feature:
        return "Track Speed (um/frame)"
    if "_CV" in feature:
        return "Coefficient of Variation"
    if "_arrest" in feature:
        return "Fraction Steps < 0.1 um"
    if "_fast" in feature:
        return "Fraction Steps > 0.4 um"
    if "_volume" in feature:
        return "Track Volume (um^3)"


""" MAHALANOBIS SIMILARIY"""


def mahalanobis(u, v, cov):
    if type(cov) == np.ndarray:
        cov = torch.from_numpy(cov) ** 2 * torch.eye(len(cov))
    else:
        cov = cov**2 * torch.eye(len(cov))
    if type(u) == np.ndarray:
        u = torch.from_numpy(u)
    if type(v) == np.ndarray:
        v = torch.from_numpy(v)
    delta = u - v
    m = torch.dot(delta, torch.matmul(torch.inverse(cov + 10**-6), delta))
    return torch.sqrt(m).cpu().numpy()


def euclidean_dist(ref, q):
    if type(ref) == np.ndarray:
        ref = torch.from_numpy(ref)
    if type(q) == np.ndarray:
        q = torch.from_numpy(q)
    return np.linalg.norm(ref.cpu().numpy() - q.cpu().numpy())


def load_or_create_TempPred(
    savepath,
    savename,
    best_models_sorted,
    temperature,
    datapath,
    filenames_X,
    filenames_y,
    globals,
    features=["XYZ", "SL", "DP"],
):

    if os.path.exists(savepath + savename):
        results_dict = pickle.load(open(savepath + savename, "rb"))
    else:
        # prep X and y
        X_to_eval, y_to_eval = load_Xy(
            datapath, filenames_X, filenames_y, features=features
        )

        results_dict = {}
        for modelname in best_models_sorted:
            model = load_UnetModels_directly(modelname, device=device, dim=dim)
            ensemble_score_temp = temperature_pred(
                X_to_eval,
                y_to_eval,
                model,
                temperature=temperature,
                X_padtoken=globals.X_padtoken,
                y_padtoken=globals.y_padtoken,
                device=globals.device,
            )
            results_dict[modelname] = ensemble_score_temp

        if len(best_models_sorted) > 1:
            ensemble_score = ensemble_scoring(
                results_dict[best_models_sorted[0]],
                results_dict[best_models_sorted[1]],
                results_dict[best_models_sorted[2]],
            )
            results_dict["ensemble_score"] = ensemble_score

            ensemble_pred = [
                np.argmax(ensemble_score[i], axis=0) for i in range(len(ensemble_score))
            ]
            results_dict["ensemble"] = ensemble_pred

        pickle.dump(results_dict, open(savepath + savename, "wb"))
    return results_dict


def compare_pred2sim_diffusion3D(trace_list, y_list, pred_list):
    diff_types = ["Normal", "Directed", "Confined", "Subdiffusive"]
    color_dict = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c", 3: "#ff7f0e"}
    for i in range(len(trace_list)):
        x = trace_list[i][:, 0]
        y = trace_list[i][:, 1]
        z = trace_list[i][:, 2]

        trace_true_labels = y_list[i]
        trace_pred_labels = pred_list[i]

        acc = np.mean(trace_true_labels == trace_pred_labels)
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Unlabelled Simulation",
                "Simulated length: " + str(len(trace_true_labels)),
                "Predicted Accuracy: {:.3f}".format(acc),
            ],
            specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
            column_widths=[4, 4, 4],
        )
        _, pos_true, seg_difftype_true = find_segments(trace_true_labels)
        _, pos_pred, seg_difftype_pred = find_segments(trace_pred_labels)

        names = []

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                showlegend=False,
                line=dict(color="dimgrey", width=2),
            ),
            row=1,
            col=1,
        )

        for idx in range(len(pos_true) - 1):
            fig.add_trace(
                go.Scatter3d(
                    x=x[pos_true[idx] : pos_true[idx + 1] + 1],
                    y=y[pos_true[idx] : pos_true[idx + 1] + 1],
                    z=z[pos_true[idx] : pos_true[idx + 1] + 1],
                    mode="lines",
                    name=diff_types[int(seg_difftype_true[idx])],
                    showlegend=(
                        True
                        if diff_types[int(seg_difftype_true[idx])] not in names
                        else False
                    ),
                    line=dict(color=color_dict[int(seg_difftype_true[idx])], width=2),
                ),
                row=1,
                col=2,
            )
            names.append(diff_types[int(seg_difftype_true[idx])])

        for idx in range(len(pos_pred) - 1):
            fig.add_trace(
                go.Scatter3d(
                    x=x[pos_pred[idx] : pos_pred[idx + 1] + 1],
                    y=y[pos_pred[idx] : pos_pred[idx + 1] + 1],
                    z=z[pos_pred[idx] : pos_pred[idx + 1] + 1],
                    mode="lines",
                    name=diff_types[int(seg_difftype_pred[idx])],
                    showlegend=(
                        True
                        if diff_types[int(seg_difftype_pred[idx])] not in names
                        else False
                    ),
                    line=dict(color=color_dict[int(seg_difftype_pred[idx])], width=2),
                ),
                row=1,
                col=3,
            )
            names.append(diff_types[int(seg_difftype_pred[idx])])
        fig.update_annotations(font_size=14)
        fig.update_layout(autosize=False, width=800, height=400)
        fig.show()


def compare_pred2sim_diffusion(x, y, y_list, pred_list, savename=""):
    color_dict = {"0": "blue", "1": "red", "2": "green", "3": "darkorange"}

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
    c_sim = [colors.to_rgba(color_dict[str(int(label))]) for label in y_list]
    c_pred = [colors.to_rgba(color_dict[str(label)]) for label in pred_list]

    lines = [
        ((x0, y0), (x1, y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])
    ]

    colored_lines_sim = LineCollection(lines, colors=c_sim, linewidths=(2,))
    colored_lines_pred = LineCollection(lines, colors=c_pred, linewidths=(2,))

    markers = [
        plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
        for color in color_dict.values()
    ]
    diff_types = ["Norm", "Dir", "Conf", "Sub"]

    acc = np.mean(np.hstack(y_list) == np.hstack(pred_list))

    # plot data
    ax1.add_collection(colored_lines_sim)
    ax1.autoscale_view()
    ax2.add_collection(colored_lines_pred)
    ax2.autoscale_view()

    ax1.set_title("Simulated trace length: {}".format(len(y_list)))
    ax2.set_title("Predicted Accuracy: {:.3f}".format(acc))
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    plt.tight_layout()

    if len(savename) > 0:
        plt.savefig(savename + str(i) + ".pdf")
    # plt.legend(markers, diff_types, numpoints=1, bbox_to_anchor=(1.36, 1.03))

    plt.show()


def gen_temporal_features(ensemble_pred):
    new_feature1 = []
    new_feature2 = []
    new_feature3 = []
    new_feature4 = []
    new_feature5 = []
    new_feature6 = []
    for ep in ensemble_pred:
        segl, cps, vals = find_segments(ep)
        sim_value = []
        sequence_value = []
        for i, val in enumerate(vals):
            if val == 0:
                sequence_value.append(1 * i)
                sim_value.append(0)
            elif val == 1:
                sequence_value.append(2 * i)
                sim_value.append(1)
            elif val == 2:
                sequence_value.append(3 * i)
                sim_value.append(4)
            elif val == 3:
                sequence_value.append(4 * i)
                sim_value.append(6)

        new_feature1.append(np.mean(sequence_value))
        new_feature2.append(np.median(sequence_value))
        new_feature3.append(np.max(sequence_value))
        new_feature4.append(np.min(sequence_value))
        new_feature5.append(np.std(sequence_value))
        new_feature6.append(np.median(np.sqrt(np.diff(sim_value) ** 2)))
    new_feature1 = np.array(new_feature1)
    new_feature2 = np.array(new_feature2)
    new_feature3 = np.array(new_feature3)
    new_feature4 = np.array(new_feature4)
    new_feature5 = np.array(new_feature5)
    new_feature6 = np.array(new_feature6)
    return (
        new_feature1,
        new_feature2,
        new_feature3,
        new_feature4,
        new_feature5,
        new_feature6,
    )


def get_perc_per_diff(preds):
    perc_0 = []
    perc_1 = []
    perc_2 = []
    perc_3 = []
    num_change_points_after = []
    for ep_i, ep in enumerate(preds):
        num_change_points_after.append(len(find_segments(ep)[-1]))
        perc_0.append(np.mean(ep == 0))
        perc_1.append(np.mean(ep == 1))
        perc_2.append(np.mean(ep == 2))
        perc_3.append(np.mean(ep == 3))
    perc_0 = np.array(perc_0)
    perc_1 = np.array(perc_1)
    perc_2 = np.array(perc_2)
    perc_3 = np.array(perc_3)
    num_change_points_after = np.array(num_change_points_after)
    return perc_0, perc_1, perc_2, perc_3, num_change_points_after


def SquareDist(x0, x1, y0, y1, z0, z1):
    """Computes the squared distance between the two points (x0,y0) and (y1,y1)

    Returns
    -------
    float
        squared distance between the two input points

    """
    return (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2


def get_inst_msd(tracks, dim, dt):
    inst_msds_all = []
    for i, t in enumerate(tracks):
        if dim == 3:
            x, y, z = t[:, 0], t[:, 1], t[:, 2]
        elif dim == 2:
            x, y, z = t[:, 0], t[:, 1], np.zeros(len(t))
        lag = 1
        inst_msd = np.mean(
            [
                SquareDist(x[j], x[j + lag], y[j], y[j + lag], z[j], z[j + lag])
                for j in range(len(x) - lag)
            ]
        ) / (2 * dim * dt)
        inst_msds_all.append(inst_msd)
    inst_msds_all = np.array(inst_msds_all)
    return inst_msds_all


def create_temporalfingerprint(
    tracks,
    ensemble_pred,
    fp_datapath,
    hmm_filename,
    dim,
    dt,
    selected_features=list(range(43)),
):
    """
    Create temporal fingerprint for a single track
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    FP = []
    for track in tqdm(tracks):
        FP.append(
            create_fingerprint_track(
                track, fp_datapath, hmm_filename, dim, dt, "Normal"
            )
        )
    FP = np.vstack(FP)

    (
        new_feature1,
        new_feature2,
        new_feature3,
        new_feature4,
        new_feature5,
        new_feature6,
    ) = gen_temporal_features(ensemble_pred)
    inst_msds_D_all = get_inst_msd(tracks, dim, dt)
    perc_ND, perc_DM, perc_CD, perc_SD, num_cp = get_perc_per_diff(ensemble_pred)

    FP_all = np.column_stack(
        [
            FP,
            perc_ND.reshape(-1, 1),
            perc_DM.reshape(-1, 1),
            perc_CD.reshape(-1, 1),
            perc_SD.reshape(-1, 1),
            num_cp.reshape(-1, 1),
            inst_msds_D_all.reshape(-1, 1),
            new_feature1.reshape(-1, 1),
            new_feature2.reshape(-1, 1),
            new_feature3.reshape(-1, 1),
            new_feature4.reshape(-1, 1),
            new_feature5.reshape(-1, 1),
            new_feature6.reshape(-1, 1),
        ]
    )
    FP_all[np.isnan(FP_all)] = 0

    return FP_all[:, selected_features]


def run_temporalsegmentation(
    best_models_sorted,
    X_to_eval,
    y_to_eval,
    use_mlflow=False,
    modelpath="",
    dir_name="",
    device="cpu",
    dim=2,
    min_max_len=601,
    X_padtoken=0,
    y_padtoken=10,
    batch_size=32,
    seed=42,
    rerun_segmentaion=True,
    savename_score="ensemble_score.pkl",
    savename_pred="ensemble_pred.pkl",
    use_temperature=False,
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    "returns ensemble_score, ensemble_pred"
    save_path_score = os.path.join(dir_name, savename_score)
    save_path_pred = os.path.join(dir_name, savename_pred)
    # one can change the if statement to True to recompute the ensemble score
    if rerun_segmentaion:
        run_temporalsegmentation = True
    elif not os.path.exists(save_path_score):
        run_temporalsegmentation = True
    else:
        run_temporalsegmentation = False

    if run_temporalsegmentation:
        print("Running temporal segmentation, maybe this takes a while")
        files_dict = {}
        for modelname in best_models_sorted:
            if use_mlflow:
                model = load_UnetModels(
                    modelname, dir=modelpath, device=device, dim=dim
                )
            else:
                model = load_UnetModels_directly(modelname, device=device, dim=dim)

            if not use_temperature:
                tmp_dict = make_preds(
                    model,
                    X_to_eval,
                    y_to_eval,
                    min_max_len=min_max_len,
                    X_padtoken=X_padtoken,
                    y_padtoken=y_padtoken,
                    batch_size=batch_size,
                    device=device,
                )
            elif use_temperature:
                tmp_dict = make_temperature_pred(
                    model,
                    X_to_eval,
                    y_to_eval,
                    min_max_len=min_max_len,
                    X_padtoken=X_padtoken,
                    y_padtoken=y_padtoken,
                    batch_size=batch_size,
                    device=device,
                    temperature=3.8537957365297553,
                )

            files_dict[modelname] = tmp_dict

        if len(list(files_dict.keys())) > 1:
            files_dict["ensemble_score"] = ensemble_scoring(
                files_dict[list(files_dict.keys())[0]]["masked_score"],
                files_dict[list(files_dict.keys())[1]]["masked_score"],
                files_dict[list(files_dict.keys())[2]]["masked_score"],
            )
            ensemble_score = files_dict["ensemble_score"]
            ensemble_pred = [
                np.argmax(files_dict["ensemble_score"][i], axis=0)
                for i in range(len(files_dict["ensemble_score"]))
            ]
        else:
            ensemble_score = files_dict[list(files_dict.keys())[0]]["masked_score"]
            ensemble_pred = [
                np.argmax(
                    files_dict[list(files_dict.keys())[0]]["masked_score"][i], axis=0
                )
                for i in range(
                    len(files_dict[list(files_dict.keys())[0]]["masked_score"])
                )
            ]

        pickle.dump(ensemble_score, open(save_path_score, "wb"))
        pickle.dump(ensemble_pred, open(save_path_pred, "wb"))
    else:
        ensemble_score = pickle.load(open(save_path_score, "rb"))
        ensemble_pred = pickle.load(open(save_path_pred, "rb"))

    ensemble_pred = np.array(ensemble_pred, dtype=object)
    return ensemble_score, ensemble_pred


def run_temporalsegmentation_ANDI(
    best_models_sorted,
    X_to_eval,
    y_to_eval,
    modelpath="",
    dir_name="",
    device="cpu",
    dim=2,
    min_max_len=200,
    X_padtoken=0,
    y_padtoken=10,
    batch_size=32,
    seed=42,
    rerun_segmentaion=True,
    savename_score="ensemble_score.pkl",
    savename_pred="ensemble_pred.pkl",
    use_temperature=True,
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    "returns ensemble_score, ensemble_pred"

    # one can change the if statement to True to recompute the ensemble score
    if rerun_segmentaion:
        run_temporalsegmentation = True
    elif not os.path.exists(dir_name + savename_score):
        run_temporalsegmentation = True
    else:
        run_temporalsegmentation = False

    if run_temporalsegmentation:
        print("Running temporal segmentation, maybe this takes a while")
        files_dict = {}
        for modelname in best_models_sorted:

            model = load_UnetModels_directlyANDI(modelname, device=device, dim=dim)

            tmp_dict = make_preds(
                model,
                X_to_eval,
                y_to_eval,
                min_max_len=min_max_len,
                X_padtoken=X_padtoken,
                y_padtoken=y_padtoken,
                batch_size=batch_size,
                device=device,
            )
            files_dict[modelname] = tmp_dict

        if len(list(files_dict.keys())) > 1:
            files_dict["ensemble_score"] = ensemble_scoring(
                files_dict[list(files_dict.keys())[0]]["masked_score"],
                files_dict[list(files_dict.keys())[1]]["masked_score"],
                files_dict[list(files_dict.keys())[2]]["masked_score"],
            )
            ensemble_score = files_dict["ensemble_score"]
            ensemble_pred = [
                np.argmax(files_dict["ensemble_score"][i], axis=0)
                for i in range(len(files_dict["ensemble_score"]))
            ]
        else:
            ensemble_score = files_dict[list(files_dict.keys())[0]]["masked_score"]
            ensemble_pred = [
                np.argmax(
                    files_dict[list(files_dict.keys())[0]]["masked_score"][i], axis=0
                )
                for i in range(
                    len(files_dict[list(files_dict.keys())[0]]["masked_score"])
                )
            ]
        pickle.dump(ensemble_score, open(dir_name + savename_score, "wb"))
        pickle.dump(ensemble_pred, open(dir_name + savename_pred, "wb"))
    else:
        ensemble_score = pickle.load(open(dir_name + savename_score, "rb"))
        ensemble_pred = pickle.load(open(dir_name + savename_pred, "rb"))

    ensemble_pred = np.array(ensemble_pred, dtype=object)
    return ensemble_score, ensemble_pred


def make_tracks_into_FP_timeseries(
    track,
    pred_track,
    window_size=40,
    selected_features=[],
    fp_datapath="",
    hmm_filename="",
    dim=3,
    dt=2,
):
    import warnings

    warnings.filterwarnings("ignore")
    timeseries = np.ones((len(track), len(track), len(selected_features))) * -10
    for center in range(0, len(track)):
        min_value = np.max([0, center - window_size // 2])
        max_value = np.min([len(track), center + window_size // 2 + 1])
        corr_min = center + window_size // 2 + 1 - max_value
        corr_max = min_value + window_size // 2 - center
        t = track[
            np.max([0, min_value - corr_min]) : np.min(
                [max_value + corr_max, len(track)]
            ),
            :,
        ]
        p = pred_track[
            np.max([0, min_value - corr_min]) : np.min(
                [max_value + corr_max, len(track)]
            )
        ]
        FP_vanilla_segment = create_fingerprint_track(
            t, fp_datapath, hmm_filename, dim, dt, "Normal"
        ).reshape(1, -1)

        (
            new_feature1,
            new_feature2,
            new_feature3,
            new_feature4,
            new_feature5,
            new_feature6,
        ) = gen_temporal_features([p])
        inst_msds_D_all = get_inst_msd([t], dim, dt)
        perc_ND, perc_DM, perc_CD, perc_SD, num_cp = get_perc_per_diff([p])
        FP_segment = np.column_stack(
            [
                FP_vanilla_segment,
                perc_ND.reshape(-1, 1),
                perc_DM.reshape(-1, 1),
                perc_CD.reshape(-1, 1),
                perc_SD.reshape(-1, 1),
                num_cp.reshape(-1, 1),
                inst_msds_D_all.reshape(-1, 1),
                new_feature1.reshape(-1, 1),
                new_feature2.reshape(-1, 1),
                new_feature3.reshape(-1, 1),
                new_feature4.reshape(-1, 1),
                new_feature5.reshape(-1, 1),
                new_feature6.reshape(-1, 1),
            ]
        )
        FP_segment[np.isnan(FP_segment)] = 0
        FP_segment = FP_segment[:, selected_features].reshape(
            -1, 1, len(selected_features)
        )
        FP_segment_repeat = np.repeat(FP_segment, len(t), axis=1)
        timeseries[
            center,
            np.max([0, min_value - corr_min]) : np.min(
                [max_value + corr_max, len(track)]
            ),
            :,
        ] = FP_segment_repeat
    timeseries_clean = np.zeros((1, len(track), len(selected_features)))
    for col in range(len(timeseries)):
        a = np.mean(
            timeseries[:, col, :][timeseries[:, col, :] != -10].reshape(
                -1, len(selected_features)
            ),
            axis=0,
        )
        timeseries_clean[0, col, :] = a

    return timeseries_clean


def distance_to_plane(params, points):
    a, b, c, d = params
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    distances = np.abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
    return distances


def mahalanobis_distance(points, mean, cov):
    return np.sqrt(
        np.sum(
            np.dot(np.linalg.inv(cov), (points - mean).T).T * (points - mean), axis=1
        )
    )


def sum_squared_distances(params, points):
    a, b, c, d = params
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    distances = np.abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
    return np.sum(distances**2)


def rotate_around_y(points, angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)],
        ]
    )
    return np.dot(points, rotation_matrix)


# flatten list of lists
def flatten_list(l):
    return [item for sublist in l for item in sublist]


def sci_notation(number, sig_fig=2):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    # remove leading "+" and strip leading zeros
    b = int(b)
    print(a, b, number)
    return str(np.round(float(a), 5)) + " \u22C5 10" + get_super(str(b))


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans("".join(normal), "".join(super_s))
    return x.translate(res)
