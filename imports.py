# Importing necessary libraries
import os
from datetime import datetime, timedelta
from time import sleep

import pandas as pd
import numpy as np

import yfinance as yf
import pandas_datareader.data as web
import ta_functions as ta
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText
from matplotlib.collections import LineCollection

import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from tabulate import tabulate

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')
