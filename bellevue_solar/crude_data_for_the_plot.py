"""Auto-split from legacy monolithic script."""

from scipy.optimize import linprog
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import torch
import torch.nn as nn

def crude_data_for_the_plot() -> None:
    crudes = ['A', 'B', 'C']

    api = [34, 40, 30]

    sulfur = [1.2, 0.5, 2.0]

    cost = [70, 80, 65]

    available = [5000, 3000, 4000]

    blend = [3000, 2000, 1000]

    fig, ax1 = plt.subplots(figsize=(8, 4))

    width = 0.3

    x = np.arange(len(crudes))

    ax1.bar(x - width, api, width, label='API Gravity')

    ax1.bar(x, sulfur, width, label='Sulfur (%)')

    ax1.set_xticks(x)

    ax1.set_xticklabels(crudes)

    ax1.set_ylabel('Value')

    ax1.set_title('Crude Property Comparison')

    ax1.legend()

    plt.tight_layout()

    plt.savefig('crude_properties.png')

    plt.show()

    fig, ax2 = plt.subplots(figsize=(8, 4))

    ax2.bar(x - width, cost, width, label='Cost ($/bbl)')

    ax2.bar(x, available, width, label='Available (bbl)', color='gray')

    ax2.set_xticks(x)

    ax2.set_xticklabels(crudes)

    ax2.set_ylabel('Value')

    ax2.set_title('Cost and Available Volume')

    ax2.legend()

    plt.tight_layout()

    plt.savefig('crude_cost_volume.png')

    plt.show()

    fig, ax3 = plt.subplots(figsize=(8, 4))

    ax3.bar(crudes, blend, color='green')

    ax3.set_ylabel('Volume (bbl)')

    ax3.set_title('Optimal Blend Volumes by Crude')

    plt.tight_layout()

    plt.savefig('blend_volumes.png')

    plt.show()

