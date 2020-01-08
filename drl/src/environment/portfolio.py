"""
Modified from https://github.com/wassname/rl-portfolio-management/blob/master/src/environments/portfolio.py
"""
from __future__ import print_function

from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
import gym.spaces

from utils.data import date_to_index, index_to_date

eps = 1e-8


def random_shift(x, fraction):
    """ Apply a random shift to a pandas series. """
    min_x, max_x = np.min(x), np.max(x)
    m = np.random.uniform(-fraction, fraction, size=x.shape) + 1
    return np.clip(x * m, min_x, max_x)


def scale_to_start(x):
    """ Scale pandas series so that it starts at one. """
    x = (x + eps) / (x[0] + eps)
    return x


def sharpe(returns, freq=30, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)


def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / (peak + eps)

class DataGenerator(object):
    """Acts as data provider for each new episode."""
    
    step = 0
    
    # def __init__(self, history, abbreviation, steps=730, window_length=50, start_idx=0, start_date=None):
    def __init__(self, parameters, steps=730, window_length=50, start_idx=0):
        """
        New Args:
            parameters: dictionary of mgarch parameters from R
            steps: the total number of steps to simulate, default is 2 years
            window_length: observation window, must be less than 50

        Old Args:
            history: (num_stocks, timestamp, 5) open, high, low, close, volume
            abbreviation: a list of length num_stocks with assets name
            steps: the total number of steps to simulate, default is 2 years
            window_length: observation window, must be less than 50
            start_date: the date to start. Default is None and random pick one.
                        It should be a string e.g. '2012-08-13'
        """
        # assert history.shape[0] == len(abbreviation), 'Number of stock is not consistent'
        import copy

        self.parameters = parameters
        self.steps = steps + 1
        self.window_length = window_length
        self.idx = 0
        # self.start_idx = start_idx
        # self.start_date = start_date

        # make immutable class
        # self._data = history.copy()  # all data
        # self.asset_names = copy.copy(abbreviation)

        # NEW
        self._data = self.generate_data()
        
        print('Data generated')
        # get data for this episode, each episode might be different. you can change start date for each episode
        # if self.start_date is None:
        #     self.idx = np.random.randint(low=self.window_length, high=self._data.shape[1] - self.steps)
        # else:
        #     # compute index corresponding to start_date for repeatable sequence
        #     self.idx = date_to_index(self.start_date) - self.start_idx
        #     assert self.idx >= self.window_length and self.idx <= self._data.shape[1] - self.steps, \
        #         'Invalid start date, must be window_length day after start date and simulation steps day before end date'
        # print('Start date: {}'.format(index_to_date(self.idx)))

    def _step(self):
        # get observation matrix from history, exclude volume, maybe volume is useful as it
        # indicates how market total investment changes. Normalize could be critical here
        self.step += 1
        
        # OLD
        # data in shape [num_assets, num_days, 3]
        # last dim = [open, close, condition_num]
        obs = self._data[:, self.step:self.step + self.window_length, :].copy()
        
        # normalize obs with open price
        
        # OLD
        # used for compute optimal action and sanity check
        ground_truth_obs = self._data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()
        
        done = self.step >= self.steps
        return obs, done, ground_truth_obs

    def reset(self):
        self.step = 0
        
        # get data for this episode, each episode might be different.
        # OLD
        '''
        if self.start_date is None:
            self.idx = np.random.randint(
                low=self.window_length, high=self._data.shape[1] - self.steps)
        else:
            # compute index corresponding to start_date for repeatable sequence
            self.idx = date_to_index(self.start_date) - self.start_idx
            assert self.idx >= self.window_length and self.idx <= self._data.shape[1] - self.steps, \
                'Invalid start date, must be window_length day after start date and simulation steps day before end date'
        # print('Start date: {}'.format(index_to_date(self.idx)))
        '''
        
        data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1, :]
        # apply augmentation?
        self.data = data
        return self.data[:, self.step:self.step + self.window_length, :].copy(), \
               self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()
        
       
    def generate_data(self):

        T = self.steps
        Q_bar = self.parameters['Q_bar']
        Q = self.parameters['Q']
        num_assets = self.parameters['num_assets']
        small_scalar = self.parameters['small_scalar']

        condNum_list = []
        a_list = []

        # initialize
        a0 = np.linalg.cholesky(parameters['H_init'])@np.random.multivariate_normal(np.zeros(parameters['num_assets']),1*np.identity(parameters['num_assets']))

        # needed to ensure H psd
        h0 = np.ones(parameters['num_assets'])*parameters['small_scalar']

        Q = parameters['Q']

        for t in range(T):
            # compute R_t
            Q_star_inv = np.linalg.inv(np.diag(Q.diagonal()))

            R = Q_star_inv@parameters['Q_bar']@Q_star_inv

            # compute D_t
            h1 = parameters['omega'] + parameters['alpha']*a0**2 + parameters['beta']*h0
            D = np.power(np.diag(h1),1/2)

            # compute H
            H = D@R@D

            # draw z
            z = np.random.multivariate_normal(np.zeros(num_assets),1*np.identity(num_assets))

            # compute a: a = 𝐻^1/2 @ z
            a1 = np.linalg.cholesky(H)@z

            # draw e
            e = np.random.multivariate_normal(np.zeros(num_assets),R)

            # step Q
            Q = (1-a-b)*parameters['Q_bar'] + a*np.outer(e,e) + b*Q

            # step a,h
            h0 = h1
            a0 = a1.squeeze()

            # covariance matrices - keep track of our generated data
            condNum_list.append(np.linalg.cond(H))
            H_list.append(H)
            # returns
            a_list.append(a0)
        
        # [assets,num_days]
        returns_close = np.vstack(a_list).T
        
        # roll 1 day forward to get open,close - close becomes new open
        returns_open = np.roll(returns_close,1,axis=1)
        # start at 0 
        returns_open[:,0] = np.zeros(num_assets)
        
        returns_stack = np.stack([returns_open,returns_close],axis=2)
        
        return np.concatenate((returns_stack,np.tile(np.asarray(condNum_list),(num_assets,1))[:,:,None]),axis=2)


class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """
    
    p0 = 1.0
    infos = []

    def __init__(self, asset_names=list(), steps=730, epsilon=0.01, time_cost=0.0):
        self.asset_names = asset_names
        self.epsilon = epsilon
        self.time_cost = time_cost
        self.steps = steps

    def _step(self, w1, y1, c1):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9,0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        p0 = self.p0

        dw1 = (y1 * w1) / (np.dot(y1, w1) + eps)  # (eq7) weights evolve into
        
        ### UPDATE
        mu1 = self.epsilon * c1 * (np.square(dw1 - w1)).sum()  # (eq16) cost to change portfolio with condition number cost and quadratic penalty - MODIFIED

        # mu1 = self.cost * (np.abs(dw1 - w1)).sum() # (eq16) cost to change portfolio - ORIGINAL

        # assert mu1 < 1.0, 'Cost is larger than current holding'

        # p1 = p0 * (1 - mu1) * np.dot(y1, w1)  # (eq11) final portfolio value - ORIGINAL

        p1 = p0 * np.dot(y1, w1) - mu1 # (eq11) final portfolio value - MODIFIED

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + eps) / (p0 + eps))  # log rate of return
        reward = r1 / self.steps * 1000.  # (22) average logarithmic accumulated return
        # remember for next step
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        done = p1 == 0

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": mu1,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.p0 = 1.0

#### environment ####
'''
environments are rendered at each step
step moves the environment forward
step returns 4 values:
    -   observation (object): an environment-specific object representing your observation of the environment. 
        For example, pixel data from a camera, joint angles and joint velocities of a robot, or the board state in a board game.
    -   reward (float): amount of reward achieved by the previous action. 
        The scale varies between environments, but the goal is always to increase your total reward.
    -   done (boolean): whether it’s time to reset the environment again. Most (but not all) tasks are divided up into well-defined episodes, 
        and done being True indicates the episode has terminated. (For example, perhaps the pole tipped too far, or you lost your last life.)
    -   info (dict): diagnostic information useful for debugging. It can sometimes be useful for learning (for example, 
        it might contain the raw probabilities behind the environment’s last state change). 
        However, official evaluations of your agent are not allowed to use this for learning.

Every environment comes with an action_space and an observation_space. These attributes are of type Space, 
and they describe the format of valid actions and observations:
'''

class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['human', 'ansi']}
    
    infos = []

    def __init__(self,
                 history,
                 abbreviation,
                 steps=730,  # 2 years
                 epsilon=0.01,
                 time_cost=0.00,
                 window_length=50,
                 start_idx=0,
                 sample_start_date=None
                 ):
        """
        An environment for financial portfolio management.
        Params:
            steps - steps in episode
            scale - scale data and each episode (except return)
            augment - fraction to randomly shift data by
            epsilon - see formula
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
            start_idx - The number of days from '2012-08-13' of the dataset
            sample_start_date - The start date sampling from the history
        """
        self.window_length = window_length
        self.num_stocks = history.shape[0]
        self.start_idx = start_idx


        # self.src = DataGenerator(history, abbreviation, steps=steps, window_length=window_length, start_idx=start_idx,
        #                          start_date=sample_start_date)

        self.src = DataGenerator(parameters, steps=steps, window_length=window_length, start_idx=start_idx,
                                  start_date=sample_start_date)

        self.sim = PortfolioSim(
            asset_names=abbreviation,
            epsilon=epsilon,
            time_cost=time_cost,
            steps=steps)

        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=(len(self.src.asset_names) + 1,))  # include cash

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(abbreviation), window_length,
                                                                                 history.shape[-1]))
        
    def step(self, action):
        return self._step(action)

    def _step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        np.testing.assert_almost_equal(
            action.shape,
            (len(self.sim.asset_names) + 1,)
        )

        # normalise just in case
        action = np.clip(action, 0, 1)

        weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)  # so if weights are all zeros we normalise to [1,0...]

        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        # step in source
        observation, done1, ground_truth_obs = self.src._step()

        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)


        ### THESE AREN'T PRICES ANYMORE -- NEED TO FIX THIS??
        # relative price vector of last observation day (close/open)
        close_price_vector = observation[:, -1, 1]
        open_price_vector = observation[:, -1, 0]
        

        # condition number
        ### UPDATE
        c1 = observation[0, -1, 2]
        y1 = close_price_vector / open_price_vector
        reward, info, done2 = self.sim._step(weights, y1, c1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step)
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs

        self.infos.append(info)

        return observation, reward, done1 or done2, info
    
    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        observation, ground_truth_obs = self.src.reset()
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        info = {}
        info['next_obs'] = ground_truth_obs
        return observation, info

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'human':
            self.plot()
            
    def render(self, mode='human', close=False):
        return self._render(mode='human', close=False)

    def plot(self):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos)
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        mdd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe_ratio)
        df_info[["portfolio_value", "market_value"]].plot(title=title, fig=plt.gcf(), rot=30)


class MultiActionPortfolioEnv(PortfolioEnv):
    def __init__(self,
                 history,
                 abbreviation,
                 model_names,
                 steps=730,  # 2 years
                 epsilon=0.01,
                 time_cost=0.00,
                 window_length=50,
                 start_idx=0,
                 sample_start_date=None,
                 ):
        super(MultiActionPortfolioEnv, self).__init__(history, abbreviation, steps, epsilon, time_cost, window_length,
                              start_idx, sample_start_date)
        self.model_names = model_names
        # need to create each simulator for each model
        self.sim = [PortfolioSim(
            asset_names=abbreviation,
            epsilon=epsilon,
            time_cost=time_cost,
            steps=steps) for _ in range(len(self.model_names))]

    def _step(self, action):
        """ Step the environment by a vector of actions

        Args:
            action: (num_models, num_stocks + 1)

        Returns:

        """
        assert action.ndim == 2, 'Action must be a two dimensional array with shape (num_models, num_stocks + 1)'
        assert action.shape[1] == len(self.sim[0].asset_names) + 1
        assert action.shape[0] == len(self.model_names)
        # normalise just in case
        action = np.clip(action, 0, 1)
        weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]
        weights /= (np.sum(weights, axis=1, keepdims=True) + eps)
        # so if weights are all zeros we normalise to [1,0...]
        weights[:, 0] += np.clip(1 - np.sum(weights, axis=1), 0, 1)
        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(np.sum(weights, axis=1), np.ones(shape=(weights.shape[0])), 3,
                                       err_msg='weights should sum to 1. action="%s"' % weights)
        observation, done1, ground_truth_obs = self.src._step()

        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        # relative price vector of last observation day (close/open)
        close_price_vector = observation[:, -1, 3]
        open_price_vector = observation[:, -1, 0]
        # condition number
        ### UPDATE
        c1 = observation[0, -1, 1]
        y1 = close_price_vector / open_price_vector

        rewards = np.empty(shape=(weights.shape[0]))
        info = {}
        dones = np.empty(shape=(weights.shape[0]), dtype=bool)
        for i in range(weights.shape[0]):
            reward, current_info, done2 = self.sim[i]._step(weights[i], y1)
            rewards[i] = reward
            info[self.model_names[i]] = current_info['portfolio_value']
            info['return'] = current_info['return']
            dones[i] = done2

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step)
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs

        self.infos.append(info)

        return observation, rewards, np.all(dones) or done1, info

    def _reset(self):
        self.infos = []
        for sim in self.sim:
            sim.reset()
        observation, ground_truth_obs = self.src.reset()
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        info = {}
        info['next_obs'] = ground_truth_obs
        return observation, info

    def plot(self):
        df_info = pd.DataFrame(self.infos)
        fig=plt.gcf()
        title = 'Trading Performance of Various Models'
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        df_info[self.model_names + ['market_value']].plot(title=title, fig=fig, rot=30)
