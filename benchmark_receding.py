from collections import Counter
from config import *
from copy import deepcopy
from pathlib import Path
from process_data import get_data
from structure import init_problem, update_problem, cleanup_solution
import dill
import json
import numpy as np
import os
import pandas as pd
import pickle
import sys
import time

##### INITIAL CONFIGURATIONS
days = 6
T = 48
H = days * 48
SLICE = T

solar = True
if solar:
    data_original = pd.read_csv(DATA, index_col='date', parse_dates=True)
    data_forecast = pd.read_csv(DATA_FORCAST, index_col='date', parse_dates=True)
else:
    data_original= pd.read_csv(DATA_SOLAR, index_col='date', parse_dates=True)
    data_forecast = pd.read_csv(DATA_SOLAR_FORCAST, index_col='date', parse_dates=True)

##### END INITIAL CONFIGURATIONS

def init_data(T, L):

    #prices = np.zeros((L, 7))
    prices = np.zeros((L, 3))
    for i in range(L): 
        if ((i % 48) < 14) or ((i % 48) >= 46):
            pb = 12.3
        else:
            pb = 15.8
        prices[i, 1] = pb
        prices[i, 0] = 10.0

    loads = np.zeros(L)



    data = {
        'T':          T,
        'num_slopes': 2,
        'efc':        0.95,
        'efd':        0.95,
        'bmax':       13,
        'bmin':       0,
        'charge':     0,
        'dmax':       2.5,
        'dmin':       2.5,
        'price': np.zeros((T, 4)),
        'load':  np.zeros(T),
        'history_bat': np.zeros(H),
        'history_cost': np.zeros(H),
        'history_post_net': np.zeros(H),
    }

    return data, prices, loads



def get_consumptions(id_, solar, start, length):


        # Initialize data

    DFS = [data_original, data_forecast]
    load_ = get_data(id_, start, length, DFS[0])
    forecast_ = get_data(id_, start, length, DFS[1])

    return load_, forecast_


def onerun(data, loads, forecasts, errors=True):

    status = []
    times = []
    for i in range(0, H):
        start = time.perf_counter()
        price = P[i:, :]
    #    load = L[i : i + T]
        if errors:
            load = forecasts[i:].copy()
            load[0] = loads[i]
        else:
            load = loads[i:].copy()

        data['price'] = price
        data['load'] = load
        data['T'] = H - i
        mo, c_, v_ = init_problem(data)
        mo = update_problem(mo, c_, v_, data)
        _ = mo.solve()
        sol = cleanup_solution(mo, c_, v_, data)
        bat = sol['var'][H - i] - sol['var'][2 * (H - i)]
        net = sol['net'][0]
        data['history_bat'][i] = bat
        data['history_cost'][i] = sol['var'][0]
        data['history_post_net'][i] = sol['net'][0]

        data['charge'] += bat
        end = time.perf_counter() - start
        times.append(end)
            
        if i % 50 == 0:
            print('ITER', i)

    return data

costs_selling = []
costs_keeping = []

if __name__ == '__main__':

    # seed = int(sys.argv[1])
    ID = int(sys.argv[1])
    start = int(sys.argv[2])
    filename = 'results/stats_sensitivity_receding_{}_{}'.format(ID, start)

    data, P, L = init_data(H, H)
    loads, forecasts = get_consumptions(ID, True, start, days)
    data_true = onerun(deepcopy(data), loads, forecasts, False)
    data_errors = onerun(deepcopy(data), loads, forecasts, True)

    costs = [X['history_cost'].sum() - 10.0 * X['charge'] for X in [data_true, data_errors]]
    costs_selling.append(costs)

    costs_ = [X['history_cost'].sum() for X in [data_true, data_errors]]
    costs_keeping.append(costs_)
    
    with open(filename, 'wb') as fh:
        dill.dump([costs_selling, costs_keeping], fh)
