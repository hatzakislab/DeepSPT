from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import *
from sklearn.linear_model import *
import matplotlib.pyplot as plt  
from sklearn.ensemble import *
from sklearn.metrics import *
import itertools
import numpy as np
import pandas as pd
import torch
import datetime
import pickle
import os
import math
from matplotlib.collections import LineCollection
from matplotlib import colors 
import matplotlib
from .statbib import Chi2Fit
from .Fingerprint_functions import *
from scipy import stats
from iminuit import Minuit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerTuple
