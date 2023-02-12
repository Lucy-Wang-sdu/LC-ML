
from cProfile import label
from cmath import log, sqrt
import random
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import bernoulli
import itertools as it
import matplotlib.pyplot as plt
import csv

def channel_test():
    R1 = random.random()
    if R1 < 0.3:
        result = 0
    else:
        result = 1
    return result


M = 1
N = 10
restart = True
while restart:
    if 2**M < N < 2** (M + 1):
        restart = False
    M += 1
loss_num = 2 ** M - N
Result = []
for time in [1, 2, 3]:
    s = 0
    for rand in [1, 2]:
        T = 10000
        alpha = 0.99
        Z = 128
        delta = 0.1
        Bit = 100
        iteration = 3
        epochs = 10
        epsilon = 1
        L = 2
        NI = 1
        I = 7 * epsilon ** (2) / (48 * L * (1 + 2) ** (2))
        t = 0

        users = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        Threshold = {
            'a': {'TH1': 0, 'TH2_0': 0, 'TH2_1': 0, 'TH3_00': 0, 'TH3_01': 0, 'TH3_10': 0, 'TH3_11': 0, 'TH4_111': 0,
                  'TH4_110': 0, 'TH4_101': 0, 'TH4_100': 0, 'TH4_011': 0, 'TH4_010': 0, 'TH4_001': 0, 'TH4_000': 0, 'ome': 1,
                  'lam': 1},
            'b': {'TH1': 0, 'TH2_0': 0, 'TH2_1': 0, 'TH3_00': 0, 'TH3_01': 0, 'TH3_10': 0, 'TH3_11': 0, 'TH4_111': 0,
                  'TH4_110': 0, 'TH4_101': 0, 'TH4_100': 0, 'TH4_011': 0, 'TH4_010': 0, 'TH4_001': 0, 'TH4_000': 0, 'ome': 1,
                  'lam': 1},
            'c': {'TH1': 0, 'TH2_0': 0, 'TH2_1': 0, 'TH3_00': 0, 'TH3_01': 0, 'TH3_10': 0, 'TH3_11': 0, 'TH4_111': 0,
                  'TH4_110': 0, 'TH4_101': 0, 'TH4_100': 0, 'TH4_011': 0, 'TH4_010': 0, 'TH4_001': 0, 'TH4_000': 0, 'ome': 1,
                  'lam': 1},
            'd': {'TH1': 0, 'TH2_0': 0, 'TH2_1': 0, 'TH3_00': 0, 'TH3_01': 0, 'TH3_10': 0, 'TH3_11': 0, 'TH4_111': 0,
                  'TH4_110': 0, 'TH4_101': 0, 'TH4_100': 0, 'TH4_011': 0, 'TH4_010': 0, 'TH4_001': 0, 'TH4_000': 0, 'ome': 1,
                  'lam': 1},
            'e': {'TH1': 0, 'TH2_0': 0, 'TH2_1': 0, 'TH3_00': 0, 'TH3_01': 0, 'TH3_10': 0, 'TH3_11': 0, 'TH4_111': 0,
                  'TH4_110': 0, 'TH4_101': 0, 'TH4_100': 0, 'TH4_011': 0, 'TH4_010': 0, 'TH4_001': 0, 'TH4_000': 0, 'ome': 1,
                  'lam': 1},
            'f': {'TH1': 0, 'TH2_0': 0, 'TH2_1': 0, 'TH3_00': 0, 'TH3_01': 0, 'TH3_10': 0, 'TH3_11': 0, 'TH4_111': 0,
                  'TH4_110': 0, 'TH4_101': 0, 'TH4_100': 0, 'TH4_011': 0, 'TH4_010': 0, 'TH4_001': 0, 'TH4_000': 0, 'ome': 1,
                  'lam': 1},
            'g': {'TH1': 0, 'TH2_0': 0, 'TH2_1': 0, 'TH3_00': 0, 'TH3_01': 0, 'TH3_10': 0, 'TH3_11': 0, 'TH4_111': 0,
                  'TH4_110': 0, 'TH4_101': 0, 'TH4_100': 0, 'TH4_011': 0, 'TH4_010': 0, 'TH4_001': 0, 'TH4_000': 0, 'ome': 1,
                  'lam': 1},
            'h': {'TH1': 0, 'TH2_0': 0, 'TH2_1': 0, 'TH3_00': 0, 'TH3_01': 0, 'TH3_10': 0, 'TH3_11': 0, 'TH4_111': 0,
                  'TH4_110': 0, 'TH4_101': 0, 'TH4_100': 0, 'TH4_011': 0, 'TH4_010': 0, 'TH4_001': 0, 'TH4_000': 0, 'ome': 1,
                  'lam': 1},
        }



        # Initial source nodes setting
        init_num = 8
        init_num = np.clip(init_num, 1, len(users))

        # The bernoulli relay distribution settings for different source nodes
        priv_prob = [
            [[0.9, 0, 0], [0.8, 0, 0], [0.7, 0, 0], [0.6, 0, 0], [0.5, 0, 0], [0.4, 0, 0], [0.3, 0, 0], [0.2, 0, 0],
             [0.9, 0, 0], [0.8, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0.7, 0, 0], [0.6, 0, 0], [0.5, 0, 0], [0.4, 0, 0], [0.3, 0, 0], [0.2, 0, 0], [0.9, 0, 0], [0.8, 0, 0],
             [0.9, 0, 0], [0.8, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0.2, 0, 0], [0.5, 0, 0], [0.2, 0, 0], [0.9, 0, 0], [0.1, 0, 0], [0.8, 0, 0], [0.6, 0, 0], [0.8, 0, 0],
             [0.3, 0, 0], [0.1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0.8, 0, 0], [0.9, 0, 0], [0.6, 0, 0], [0.5, 0, 0], [0.4, 0, 0], [0.3, 0, 0], [0.4, 0, 0], [0.7, 0, 0],
             [0.1, 0, 0], [0.9, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0.4, 0, 0], [0.3, 0, 0], [0.8, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.8, 0, 0], [0.9, 0, 0], [0.1, 0, 0],
             [0.5, 0, 0], [0.2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0.2, 0, 0], [0.9, 0, 0], [0.4, 0, 0], [0.8, 0, 0], [0.6, 0, 0], [0.5, 0, 0], [0.4, 0, 0], [0.1, 0, 0],
             [0.3, 0, 0], [0.2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0.7, 0, 0], [0.6, 0, 0], [0.2, 0, 0], [0.8, 0, 0], [0.9, 0, 0], [0.4, 0, 0], [0.3, 0, 0], [0.2, 0, 0],
             [0.1, 0, 0], [0.7, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0.6, 0, 0], [0.5, 0, 0], [0.4, 0, 0], [0.3, 0, 0], [0.2, 0, 0], [0.1, 0, 0], [0.8, 0, 0], [0.9, 0, 0],
             [0.7, 0, 0], [0.6, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]

        priv_prob_change = {
            'a':[0.7, 0.6, 0.3, 0.4, 0.8, 0.2, 0.9, 0.1, 0.5, 0.4],
            'b':[0.5, 0.2, 0.4, 0.7, 0.9, 0.6, 0.8, 0.3, 0.1, 0.1],
            'c':[0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6],
            'd':[0.6, 0.8, 0.5, 0.1, 0.2, 0.4, 0.3, 0.7, 0.9, 0.1],
            'e':[0.1, 0.8, 0.7, 0.3, 0.4, 0.6, 0.2, 0.9, 0.5, 0.8],
            'f':[0.1, 0.5, 0.3, 0.7, 0.8, 0.2, 0.6, 0.9, 0.4, 0.5],
            'g':[0.8, 0.3, 0.2, 0.6, 0.9, 0.7, 0.5, 0.4, 0.1, 0.2],
            'h':[0.8, 0.6, 0.1, 0.3, 0.9, 0.4, 0.5, 0.2, 0.7, 0.9]
        }



        ## 初始化信道
        trans_rvs = list(it.product(range(2), repeat=M))
        signalpaths = []
        for trans_rv in trans_rvs:
            res = []
            for rv in trans_rv:
                res.append('%d' % rv)
            trans = ''.join(res)
            signalpaths.append(trans)
        init_path = signalpaths[0:N]
        signalpaths_DSSL = signalpaths[0:N]

        # loss_paths = signalpaths[-loss_num:]


        # Initialize the return, preferences and relay distributions for each SN.
        priv = {}
        # priv_fuzz = {}
        ave_earn = [0] * len(signalpaths)
        signal_path = {}
        index = 0
        for user in users:
            priv[user] = {'path': '', 'pref': []}
            # priv_fuzz[user] = {'path': '', 'pref': []}
            priv[user]['ave_earn'] = dict(zip(signalpaths, ave_earn))  # return
            # priv_fuzz[user]['ave_earn'] = dict(zip(signalpaths, ave_earn))
            signal_path[user] = dict(zip(signalpaths, priv_prob[index]))  # relay distribution
            index = index + 1

        for user in users:

            p = [1 / len(init_path)] * len(init_path)
            p_accu = list(np.divide(range(0, len(init_path) + 1), len(init_path)))

            for m in range(0, iteration):
                R = random.random()
                for i in range(0, len(init_path)):
                    if (p_accu[i] < R < p_accu[i + 1]):

                        channel = init_path[i]
                        result = channel_test()
                        if (result == 1):
                            p = [0] * len(init_path)
                            p[i] = 1
                        else:
                            p = [(x * 0.9 + 0.1 / (len(init_path) - 1)) for x in p]
                            p[i] = p[i] - 0.1 / (len(init_path) - 1)
                        accu = 0
                        for k in range(0, len(init_path)):
                            accu += p[k]
                            p_accu[k + 1] = accu

                        break
                    else:
                        continue


            priv[user]['path'] = channel
        total_earns = []
        change_times = [0]
        # LCB-PL Algorithm
        for time in range(0, T):
            if(time >= 3000):
                for user in users:
                    index = 0
                    for path in signalpaths[0:N]:
                        signal_path[user][path][0] = priv_prob_change[user][index]
                        index += 1
            total_earn_epoch = 0
            change_time_epoch = 0

            for epoch in range(0, epochs):
                change_time = 0
                for user in users:
                    s1 = float(rand_number[s])
                    s += 1
                    s2 = float(rand_number[s])
                    s += 1
                    s3 = float(rand_number[s])
                    s += 1
                    s4 = float(rand_number[s])
                    s += 1
                    if s1 >= Threshold[user]['TH1']:
                        j1 = '1'
                        if s2 >= Threshold[user]['TH2_1']:
                            j2 = '1'
                            if s3 >= Threshold[user]['TH3_11']:
                                j3 = '1'
                                if s4 >= Threshold[user]['TH4_111']:
                                    j4 = '1'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    if (earn > 0):
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['lam']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] - Threshold[user]['lam']
                                        Threshold[user]['TH3_11'] = alpha * Threshold[user]['TH3_11'] - Threshold[user]['lam']
                                        Threshold[user]['TH4_111'] = alpha * Threshold[user]['TH4_111'] - Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['ome']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] + Threshold[user]['ome']
                                        Threshold[user]['TH3_11'] = alpha * Threshold[user]['TH3_11'] + Threshold[user]['ome']
                                        Threshold[user]['TH4_111'] = alpha * Threshold[user]['TH4_111'] + Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_1'] = np.clip(Threshold[user]['TH2_1'], -Z, Z)
                                    Threshold[user]['TH3_11'] = np.clip(Threshold[user]['TH3_11'], -Z, Z)
                                    Threshold[user]['TH4_111'] = np.clip(Threshold[user]['TH4_111'], -Z, Z)
                                else:
                                    j4 = '0'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if earn > 0:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['lam']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] - Threshold[user]['lam']
                                        Threshold[user]['TH3_11'] = alpha * Threshold[user]['TH3_11'] - Threshold[user]['lam']
                                        Threshold[user]['TH4_111'] = alpha * Threshold[user]['TH4_111'] + Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['ome']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] + Threshold[user]['ome']
                                        Threshold[user]['TH3_11'] = alpha * Threshold[user]['TH3_11'] + Threshold[user]['ome']
                                        Threshold[user]['TH4_111'] = alpha * Threshold[user]['TH4_111'] - Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_1'] = np.clip(Threshold[user]['TH2_1'], -Z, Z)
                                    Threshold[user]['TH3_11'] = np.clip(Threshold[user]['TH3_11'], -Z, Z)
                                    Threshold[user]['TH4_111'] = np.clip(Threshold[user]['TH4_111'], -Z, Z)
                            else:
                                j3 = '0'
                                if s4 >= Threshold[user]['TH4_110']:
                                    j4 = '1'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if (earn > 0):
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['lam']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] - Threshold[user]['lam']
                                        Threshold[user]['TH3_11'] = alpha * Threshold[user]['TH3_11'] + Threshold[user]['lam']
                                        Threshold[user]['TH4_110'] = alpha * Threshold[user]['TH4_110'] - Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['ome']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] + Threshold[user]['ome']
                                        Threshold[user]['TH3_11'] = alpha * Threshold[user]['TH3_11'] - Threshold[user]['ome']
                                        Threshold[user]['TH4_110'] = alpha * Threshold[user]['TH4_110'] + Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_1'] = np.clip(Threshold[user]['TH2_1'], -Z, Z)
                                    Threshold[user]['TH3_11'] = np.clip(Threshold[user]['TH3_11'], -Z, Z)
                                    Threshold[user]['TH4_110'] = np.clip(Threshold[user]['TH4_110'], -Z, Z)
                                else:
                                    j4 = '0'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if earn > 0:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['lam']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] - Threshold[user]['lam']
                                        Threshold[user]['TH3_11'] = alpha * Threshold[user]['TH3_11'] + Threshold[user]['lam']
                                        Threshold[user]['TH4_110'] = alpha * Threshold[user]['TH4_110'] + Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['ome']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] + Threshold[user]['ome']
                                        Threshold[user]['TH3_11'] = alpha * Threshold[user]['TH3_11'] - Threshold[user]['ome']
                                        Threshold[user]['TH4_110'] = alpha * Threshold[user]['TH4_110'] - Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_1'] = np.clip(Threshold[user]['TH2_1'], -Z, Z)
                                    Threshold[user]['TH3_11'] = np.clip(Threshold[user]['TH3_11'], -Z, Z)
                                    Threshold[user]['TH4_110'] = np.clip(Threshold[user]['TH4_110'], -Z, Z)
                        else:
                            j2 = '0'
                            if s3 >= Threshold[user]['TH3_10']:
                                j3 = '1'
                                if s4 >= Threshold[user]['TH4_101']:
                                    j4 = '1'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if earn > 0:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['lam']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] + Threshold[user]['lam']
                                        Threshold[user]['TH3_10'] = alpha * Threshold[user]['TH3_10'] - Threshold[user]['lam']
                                        Threshold[user]['TH4_101'] = alpha * Threshold[user]['TH4_101'] - Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['ome']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] - Threshold[user]['ome']
                                        Threshold[user]['TH3_10'] = alpha * Threshold[user]['TH3_10'] + Threshold[user]['ome']
                                        Threshold[user]['TH4_101'] = alpha * Threshold[user]['TH4_101'] + Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_1'] = np.clip(Threshold[user]['TH2_1'], -Z, Z)
                                    Threshold[user]['TH3_10'] = np.clip(Threshold[user]['TH3_10'], -Z, Z)
                                    Threshold[user]['TH4_101'] = np.clip(Threshold[user]['TH4_101'], -Z, Z)
                                else:
                                    j4 = '0'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if earn > 0:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['lam']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] + Threshold[user]['lam']
                                        Threshold[user]['TH3_10'] = alpha * Threshold[user]['TH3_10'] - Threshold[user]['lam']
                                        Threshold[user]['TH4_101'] = alpha * Threshold[user]['TH4_101'] + Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['ome']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] - Threshold[user]['ome']
                                        Threshold[user]['TH3_10'] = alpha * Threshold[user]['TH3_10'] + Threshold[user]['ome']
                                        Threshold[user]['TH4_101'] = alpha * Threshold[user]['TH4_101'] - Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_1'] = np.clip(Threshold[user]['TH2_1'], -Z, Z)
                                    Threshold[user]['TH3_10'] = np.clip(Threshold[user]['TH3_10'], -Z, Z)
                                    Threshold[user]['TH4_101'] = np.clip(Threshold[user]['TH4_101'], -Z, Z)
                            else:
                                j3 = '0'
                                if s4 >= Threshold[user]['TH4_100']:
                                    j4 = '1'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if earn > 0:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['lam']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] + Threshold[user]['lam']
                                        Threshold[user]['TH3_10'] = alpha * Threshold[user]['TH3_10'] + Threshold[user]['lam']
                                        Threshold[user]['TH4_100'] = alpha * Threshold[user]['TH4_100'] - Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['ome']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] - Threshold[user]['ome']
                                        Threshold[user]['TH3_10'] = alpha * Threshold[user]['TH3_10'] - Threshold[user]['ome']
                                        Threshold[user]['TH4_100'] = alpha * Threshold[user]['TH4_100'] + Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_1'] = np.clip(Threshold[user]['TH2_1'], -Z, Z)
                                    Threshold[user]['TH3_10'] = np.clip(Threshold[user]['TH3_10'], -Z, Z)
                                    Threshold[user]['TH4_100'] = np.clip(Threshold[user]['TH4_100'], -Z, Z)
                                else:
                                    j4 = '0'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if earn > 0:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['lam']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] + Threshold[user]['lam']
                                        Threshold[user]['TH3_10'] = alpha * Threshold[user]['TH3_10'] + Threshold[user]['lam']
                                        Threshold[user]['TH4_100'] = alpha * Threshold[user]['TH4_100'] + Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['ome']
                                        Threshold[user]['TH2_1'] = alpha * Threshold[user]['TH2_1'] - Threshold[user]['ome']
                                        Threshold[user]['TH3_10'] = alpha * Threshold[user]['TH3_10'] - Threshold[user]['ome']
                                        Threshold[user]['TH4_100'] = alpha * Threshold[user]['TH4_100'] - Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_1'] = np.clip(Threshold[user]['TH2_1'], -Z, Z)
                                    Threshold[user]['TH3_10'] = np.clip(Threshold[user]['TH3_10'], -Z, Z)
                                    Threshold[user]['TH4_100'] = np.clip(Threshold[user]['TH4_100'], -Z, Z)
                    else:
                        j1 = '0'
                        if s2 >= Threshold[user]['TH2_0']:
                            j2 = '1'
                            if s3 >= Threshold[user]['TH3_01']:
                                j3 = '1'
                                if s4 >= Threshold[user]['TH4_011']:
                                    j4 = '1'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if (earn > 0):
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['lam']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] - Threshold[user]['lam']
                                        Threshold[user]['TH3_01'] = alpha * Threshold[user]['TH3_01'] - Threshold[user]['lam']
                                        Threshold[user]['TH4_011'] = alpha * Threshold[user]['TH4_011'] - Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['ome']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] + Threshold[user]['ome']
                                        Threshold[user]['TH3_01'] = alpha * Threshold[user]['TH3_01'] + Threshold[user]['ome']
                                        Threshold[user]['TH4_011'] = alpha * Threshold[user]['TH4_011'] + Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_0'] = np.clip(Threshold[user]['TH2_0'], -Z, Z)
                                    Threshold[user]['TH3_01'] = np.clip(Threshold[user]['TH3_01'], -Z, Z)
                                    Threshold[user]['TH4_011'] = np.clip(Threshold[user]['TH4_011'], -Z, Z)
                                else:
                                    j4 = '0'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if (earn > 0):
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['lam']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] - Threshold[user]['lam']
                                        Threshold[user]['TH3_01'] = alpha * Threshold[user]['TH3_01'] - Threshold[user]['lam']
                                        Threshold[user]['TH4_011'] = alpha * Threshold[user]['TH4_011'] + Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['ome']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] + Threshold[user]['ome']
                                        Threshold[user]['TH3_01'] = alpha * Threshold[user]['TH3_01'] + Threshold[user]['ome']
                                        Threshold[user]['TH4_011'] = alpha * Threshold[user]['TH4_011'] - Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_0'] = np.clip(Threshold[user]['TH2_0'], -Z, Z)
                                    Threshold[user]['TH3_01'] = np.clip(Threshold[user]['TH3_01'], -Z, Z)
                                    Threshold[user]['TH4_011'] = np.clip(Threshold[user]['TH4_011'], -Z, Z)
                            else:
                                j3 = '0'
                                if s4 >= Threshold[user]['TH4_010']:
                                    j4 = '1'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if (earn > 0):
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['lam']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] - Threshold[user]['lam']
                                        Threshold[user]['TH3_01'] = alpha * Threshold[user]['TH3_01'] + Threshold[user]['lam']
                                        Threshold[user]['TH4_010'] = alpha * Threshold[user]['TH4_010'] - Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['ome']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] + Threshold[user]['ome']
                                        Threshold[user]['TH3_01'] = alpha * Threshold[user]['TH3_01'] - Threshold[user]['ome']
                                        Threshold[user]['TH4_010'] = alpha * Threshold[user]['TH4_010'] + Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_0'] = np.clip(Threshold[user]['TH2_0'], -Z, Z)
                                    Threshold[user]['TH3_01'] = np.clip(Threshold[user]['TH3_01'], -Z, Z)
                                    Threshold[user]['TH4_010'] = np.clip(Threshold[user]['TH4_010'], -Z, Z)
                                else:
                                    j4 = '0'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if (earn > 0):
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['lam']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] - Threshold[user]['lam']
                                        Threshold[user]['TH3_01'] = alpha * Threshold[user]['TH3_01'] + Threshold[user]['lam']
                                        Threshold[user]['TH4_010'] = alpha * Threshold[user]['TH4_010'] + Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['ome']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] + Threshold[user]['ome']
                                        Threshold[user]['TH3_01'] = alpha * Threshold[user]['TH3_01'] - Threshold[user]['ome']
                                        Threshold[user]['TH4_010'] = alpha * Threshold[user]['TH4_010'] - Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_0'] = np.clip(Threshold[user]['TH2_0'], -Z, Z)
                                    Threshold[user]['TH3_01'] = np.clip(Threshold[user]['TH3_01'], -Z, Z)
                                    Threshold[user]['TH4_010'] = np.clip(Threshold[user]['TH4_010'], -Z, Z)
                        else:
                            j2 = '0'
                            if s3 >= Threshold[user]['TH3_00']:
                                j3 = '1'
                                if s4 >= Threshold[user]['TH4_001']:
                                    j4 = '1'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if (earn > 0):
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['lam']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] + Threshold[user]['lam']
                                        Threshold[user]['TH3_00'] = alpha * Threshold[user]['TH3_00'] - Threshold[user]['lam']
                                        Threshold[user]['TH4_001'] = alpha * Threshold[user]['TH4_001'] - Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['ome']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] - Threshold[user]['ome']
                                        Threshold[user]['TH3_00'] = alpha * Threshold[user]['TH3_00'] + Threshold[user]['ome']
                                        Threshold[user]['TH4_001'] = alpha * Threshold[user]['TH4_001'] + Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_0'] = np.clip(Threshold[user]['TH2_0'], -Z, Z)
                                    Threshold[user]['TH3_00'] = np.clip(Threshold[user]['TH3_00'], -Z, Z)
                                    Threshold[user]['TH4_001'] = np.clip(Threshold[user]['TH4_001'], -Z, Z)
                                else:
                                    j4 = '0'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if earn > 0:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['lam']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] + Threshold[user]['lam']
                                        Threshold[user]['TH3_00'] = alpha * Threshold[user]['TH3_00'] - Threshold[user]['lam']
                                        Threshold[user]['TH4_001'] = alpha * Threshold[user]['TH4_001'] + Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['ome']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] - Threshold[user]['ome']
                                        Threshold[user]['TH3_00'] = alpha * Threshold[user]['TH3_00'] + Threshold[user]['ome']
                                        Threshold[user]['TH4_001'] = alpha * Threshold[user]['TH4_001'] - Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_0'] = np.clip(Threshold[user]['TH2_0'], -Z, Z)
                                    Threshold[user]['TH3_00'] = np.clip(Threshold[user]['TH3_00'], -Z, Z)
                                    Threshold[user]['TH4_001'] = np.clip(Threshold[user]['TH4_001'], -Z, Z)
                            else:
                                j3 = '0'
                                if s4 >= Threshold[user]['TH4_000']:
                                    j4 = '1'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if earn > 0:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['lam']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] + Threshold[user]['lam']
                                        Threshold[user]['TH3_00'] = alpha * Threshold[user]['TH3_00'] + Threshold[user]['lam']
                                        Threshold[user]['TH4_000'] = alpha * Threshold[user]['TH4_000'] - Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['ome']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] - Threshold[user]['ome']
                                        Threshold[user]['TH3_00'] = alpha * Threshold[user]['TH3_00'] - Threshold[user]['ome']
                                        Threshold[user]['TH4_000'] = alpha * Threshold[user]['TH4_000'] + Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_0'] = np.clip(Threshold[user]['TH2_0'], -Z, Z)
                                    Threshold[user]['TH3_00'] = np.clip(Threshold[user]['TH3_00'], -Z, Z)
                                    Threshold[user]['TH4_000'] = np.clip(Threshold[user]['TH4_000'], -Z, Z)
                                else:
                                    j4 = '0'
                                    j = j1 + j2 + j3 + j4
                                    earn = bernoulli.rvs(signal_path[user][j][0])
                                    signal_path[user][j][1] = signal_path[user][j][1] + 1
                                    signal_path[user][j][2] = signal_path[user][j][2] + earn
                                    # priv[user]['path'] = j
                                    # priv_fuzz[user]['ave_earn'][j] = signal_path[user][j][2]/signal_path[user][j][1]
                                    priv[user]['ave_earn'][j] = signal_path[user][j][2] / signal_path[user][j][1]
                                    if earn > 0:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] + Threshold[user]['lam']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] + Threshold[user]['lam']
                                        Threshold[user]['TH3_00'] = alpha * Threshold[user]['TH3_00'] + Threshold[user]['lam']
                                        Threshold[user]['TH4_000'] = alpha * Threshold[user]['TH4_000'] + Threshold[user]['lam']
                                    else:
                                        Threshold[user]['TH1'] = alpha * Threshold[user]['TH1'] - Threshold[user]['ome']
                                        Threshold[user]['TH2_0'] = alpha * Threshold[user]['TH2_0'] - Threshold[user]['ome']
                                        Threshold[user]['TH3_00'] = alpha * Threshold[user]['TH3_00'] - Threshold[user]['ome']
                                        Threshold[user]['TH4_000'] = alpha * Threshold[user]['TH4_000'] - Threshold[user]['ome']
                                    Threshold[user]['TH1'] = np.clip(Threshold[user]['TH1'], -Z, Z)
                                    Threshold[user]['TH2_0'] = np.clip(Threshold[user]['TH2_0'], -Z, Z)
                                    Threshold[user]['TH3_00'] = np.clip(Threshold[user]['TH3_00'], -Z, Z)
                                    Threshold[user]['TH4_000'] = np.clip(Threshold[user]['TH4_000'], -Z, Z)

                    seq = []
                    prefs = sorted(priv[user]['ave_earn'].items(), key=lambda x: x[1], reverse=True)
                    for pref in prefs:
                        seq.append(pref[0])
                    priv[user]['pref'] = seq
                initiators = random.sample(users, init_num)

                for initiator in initiators:
                    priv[initiator]['flag'] = 1
                    occ_paths = []
                    # occ_paths_fuzz = []
                    other_users = list(set(users) - set(initiator))
                    for other_user in other_users:
                        occ_paths.append(priv[other_user]['path'])
                        # occ_paths_fuzz.append(priv_fuzz[other_user]['path'])
                    occ_paths = list(set(occ_paths))
                    # occ_paths_fuzz = list(set(occ_paths_fuzz))

                    for path in priv[initiator]['pref']:
                        if (path in occ_paths):
                            for other_user in other_users:
                                if (priv[other_user]['path'] == path):
                                    the_user = other_user
                            condition_1 = (priv[initiator]['ave_earn'][path] >= priv[the_user]['ave_earn'][path])
                            condition_3 = (abs(priv[initiator]['ave_earn'][path] - priv[the_user]['ave_earn'][path]) <= delta)
                            condition_4 = (abs(priv[the_user]['ave_earn'][priv[initiator]['path']] - priv[the_user]['ave_earn'][
                                path]) <= delta)
                            if  rand == 1 and condition_1:
                                change = priv[initiator]['path']
                                priv[initiator]['path'] = path
                                priv[the_user]['path'] = change
                                change_time += 1
                            elif condition_3 and condition_4:
                                change = priv[initiator]['path']
                                priv[initiator]['path'] = path
                                priv[the_user]['path'] = change
                                change_time += 1
                                break
                            else:
                                continue
                        else:

                            priv[initiator]['path'] = path
                            priv[initiator]['flag'] = 0
                            break

                change_time_epoch += change_time

                for user in users:
                    total_earn_epoch += Bit * signal_path[user][priv[user]['path']][0]
            total_earns.append(total_earn_epoch / epochs)
            change_times.append(change_times[time] + change_time_epoch / epochs)
        Result.append(total_earns)
