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

def build_a_directed_network_graph() -> None:
    pipes = pd.read_csv('eia_crude_pipelines.csv')

    G = nx.DiGraph()

    for _, r in pipes.iterrows():
        G.add_edge(r['origin'], r['destination'], capacity=r['capacity_bpd'])

    crude_specs = pd.DataFrame({'supplierA': {'cost': 70, 'api': 34, 'sulfur': 1.2, 'vol': 5000}, 'supplierB': {'cost': 80, 'api': 40, 'sulfur': 0.5, 'vol': 3000}, 'tankA': {'cost': 0, 'api': 36, 'sulfur': 0.8, 'vol': 2000}}).T

    target_vol = 6000

    api_min = 35

    sulfur_max = 1.0

    crudes = crude_specs.index.tolist()

    c = crude_specs['cost'].values

    A_ub, b_ub = ([], [])

    A_ub.append(crude_specs['sulfur'].values)

    b_ub.append(sulfur_max * target_vol)

    A_ub.append(-crude_specs['api'].values)

    b_ub.append(-api_min * target_vol)

    A_ub += np.eye(len(crudes)).tolist()

    b_ub += crude_specs['vol'].tolist()

    A_eq = [[1] * len(crudes)]

    b_eq = [target_vol]

    res = linprog(c, A_ub=np.vstack(A_ub), b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * len(crudes), method='highs')

    if not res.success:
        raise RuntimeError(res.message)

    blend = dict(zip(crudes, res.x))

    print('Optimal blend:', blend)

    flow = nx.min_cost_flow(G, demand={...}, capacity='capacity')

