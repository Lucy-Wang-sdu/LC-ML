import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt


class DSMU_MAB(object):

    def __init__(self, M, K, T):
        '''
            M: Initial Source Node Number
            K: Initial Relay Number
            T: Circulation time
        '''
        self.M = M
        self.K = K
        self.T = T

    def DSMU(self):
        Bits = 100
        users = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        users_num = [user for user in range(self.M)]
        Channels = [channel for channel in range(self.K)]
        # Channel distribution
        channel_distribution = [
            [[0.9, 0, 0], [0.8, 0, 0], [0.7, 0, 0], [0.6, 0, 0], [0.4, 0, 0], [0.4, 0, 0], [0.3, 0, 0], [0.7, 0, 0],
             [0.9, 0, 0], [0.6, 0, 0]],
            [[0.7, 0, 0], [0.6, 0, 0], [0.5, 0, 0], [0.4, 0, 0], [0.2, 0, 0], [0.2, 0, 0], [0.9, 0, 0], [0.8, 0, 0],
             [0.9, 0, 0], [0.8, 0, 0]],
            [[0.2, 0, 0], [0.5, 0, 0], [0.2, 0, 0], [0.9, 0, 0], [0.8, 0, 0], [0.8, 0, 0], [0.6, 0, 0], [0.5, 0, 0],
             [0.3, 0, 0], [0.2, 0, 0]],
            [[0.8, 0, 0], [0.9, 0, 0], [0.6, 0, 0], [0.5, 0, 0], [0.1, 0, 0], [0.3, 0, 0], [0.4, 0, 0], [0.7, 0, 0],
             [0.1, 0, 0], [0.2, 0, 0]],
            [[0.4, 0, 0], [0.3, 0, 0], [0.8, 0, 0], [0.1, 0, 0], [0.3, 0, 0], [0.8, 0, 0], [0.9, 0, 0], [0.1, 0, 0],
             [0.5, 0, 0], [0.4, 0, 0]],
            [[0.2, 0, 0], [0.9, 0, 0], [0.4, 0, 0], [0.8, 0, 0], [0.5, 0, 0], [0.5, 0, 0], [0.4, 0, 0], [0.1, 0, 0],
             [0.3, 0, 0], [0.3, 0, 0]],
            [[0.7, 0, 0], [0.6, 0, 0], [0.2, 0, 0], [0.8, 0, 0], [0.9, 0, 0], [0.4, 0, 0], [0.3, 0, 0], [0.2, 0, 0],
             [0.1, 0, 0], [0.2, 0, 0]],
            [[0.6, 0, 0], [0.5, 0, 0], [0.4, 0, 0], [0.3, 0, 0], [0.2, 0, 0], [0.1, 0, 0], [0.8, 0, 0], [0.9, 0, 0],
             [0.7, 0, 0], [0.1, 0, 0]],
        ]
        # Channel information
        channel_info = {}
        index = 0
        for user in users:
            channel_info[user] = channel_distribution[index]
            index += 1
        user_info = {}
        for user in users:
            user_info[user] = {'channel': '', 'UCB': [0] * self.K, 'return': [0] * self.K}

        # Initialization
        for t in range(self.K):
            for user in users:
                earn = bernoulli.rvs(channel_info[user][(users.index(user) + 1 + t) % self.K][0])
                channel_info[user][(users.index(user) + 1 + t) % self.K][1] += earn
                channel_info[user][(users.index(user) + 1 + t) % self.K][2] += 1
                user_info[user]['return'][(users.index(user) + 1 + t) % self.K] = \
                channel_info[user][(users.index(user) + 1 + t) % self.K][1] / \
                channel_info[user][(users.index(user) + 1 + t) % self.K][2]

        # Main circulation
        total_earns_DSMU = []
        B_UCB = np.mat(np.zeros((self.M, self.K)))
        for t in range(self.K + 1, self.T):
            index = 0
            for user in users:
                for channel in Channels:
                    user_info[user]['UCB'][channel] = channel_info[user][channel][1] + np.sqrt(
                        2 * np.log(t) / channel_info[user][channel][2])
                B_UCB[index:] = user_info[user]['UCB']

            for iter in range(self.M):
                # Choose the biggest position
                max_row = np.unravel_index(np.argmax(B_UCB), B_UCB.shape)[0]
                max_column = np.unravel_index(np.argmax(B_UCB), B_UCB.shape)[1]
                for column in Channels:
                    B_UCB[max_row, column] = 0
                for row in users_num:
                    B_UCB[row, max_column] = 0
                user_info[users[max_row]]['channel'] = Channels[max_column]
            # Observe the channel an update
            for user in users:
                for channel in Channels:
                    earn = bernoulli.rvs(channel_info[user][channel][0])
                    channel_info[user][channel][1] += earn
                    channel_info[user][channel][2] += 1
                    user_info[user]['return'][channel] = channel_info[user][channel][1] / channel_info[user][channel][2]
            earns = 0
            for user in users:
                earns += Bits * channel_info[user][user_info[user]['channel']][0]
            total_earns_DSMU.append(earns)
        return total_earns_DSMU
