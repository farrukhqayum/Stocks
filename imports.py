# Importing necessary libraries
import os
from datetime import datetime, timedelta
from time import sleep

from pathlib import Path
import appdirs as ad
CACHE_DIR = ".cache"
ad.user_cache_dir = lambda *arg: CACHE_DIR
Path(CACHE_DIR).mkdir(exist_ok=True)

import pandas as pd
import numpy as np
import openpyxl

import yfinance as yf
from curl_cffi import requests
import ta_functions as ta
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.table import Table

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from tabulate import tabulate
import emoji

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')
