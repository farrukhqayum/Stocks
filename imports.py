# Standard Library
import os
import re
import time
from datetime import datetime, timedelta
from time import sleep
from pathlib import Path
import warnings
import alpha_vantage_loader as load

# Third‑Party Libraries
import pandas as pd
import numpy as np
import openpyxl
import yfinance as yf
from curl_cffi import requests
import ta_functions as ta
from scipy.stats import norm

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText
from matplotlib.collections import LineCollection
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from matplotlib.patches import Rectangle
from matplotlib.table import Table
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import altair as alt

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Utilities
from tabulate import tabulate
import emoji

# Settings
warnings.filterwarnings('ignore')
import time, random
time.sleep(random.uniform(1, 3))
