import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from collections import namedtuple
import time
import matplotlib.pyplot as plt
import pandas as pd
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import pickle
from hpbandster.optimizers import BOHB as BOHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import logging
from utils import tt, soft_update, hard_update, cos_ann_w_restarts
from utils import exponential_decay_w_restarts, angle_normalize
from utils import plot_episode_stats, continous_plot, acton_discrete
from utils import smooth_reward
from continuous_cartpole import ContinuousCartPoleEnv
from replay_buffer import ReplayBuffer

logging.basicConfig(level=logging.DEBUG)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards",
                                   "episode_loss", "episode_epsilon"])

class Q(nn.Module):
    def __init__(self, state_dim, action_dim, non_linearity=F.relu,
                 hidden_dim=30, dropout_rate=0.0):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self.dropout(self._non_linearity(self.fc2(x)))
        x = self.dropout(self._non_linearity(self.fc3(x)))
        return self.fc4(x)

class DQN:
    def __init__(self, state_dim, action_dim, gamma,
                 conf={'lr':0.001, 'bs':64, 'loss':nn.MSELoss, 'hidden_dim':64,
                       'activation':'relu',
                       'mem_size':1e6, 'epsilon':1., 'eps_scheduler':'exp',
                       'n_episodes':1000, 'n_cycles':1, 'subtract':0.,
                      }):
        if conf['activation'] == 'relu':

            activation = torch.relu
        elif conf['activation'] == 'tanh':
            activation = torch.tanh

        self._q = Q(state_dim, action_dim,
                    non_linearity=activation, hidden_dim=conf['hidden_dim'],
                    dropout_rate=conf['dropout_rate']).to(device)
        self._q_target = Q(state_dim, action_dim,
                    non_linearity=activation, hidden_dim=conf['hidden_dim'],
                    dropout_rate=0.0).to(device)

        self._gamma = gamma
        ############################
        # exploration exploitation tradeoff
        self.epsilon = conf['epsilon']
        self.n_episodes = conf['n_episodes']
        self.n_cycles = conf['n_cycles']
        self.eps_scheduler = conf['eps_scheduler']
        ############################
        # Network
        self.bs = conf['bs']
        self._loss_function = nn.MSELoss()  # conf['loss']
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=conf['lr'])
        self._action_dim = action_dim
        self._replay_buffer = ReplayBuffer(conf['mem_size'])
        ############################
        # actions
        self.action = acton_discrete(action_dim)

    def get_action(self, x, epsilon):
        u = np.argmax(self._q(tt(x)).cpu().detach().numpy())
        r = np.random.uniform()
        if r < epsilon:
            return np.random.randint(self._action_dim)
        return u

    def train(self, episodes, time_steps, env, conf):
        # Statistics for each episode
        # start_time = time.time()
        stats = EpisodeStats(episode_lengths=np.zeros(episodes),
                             episode_rewards=np.zeros(episodes),
                             episode_loss=np.zeros(episodes),
                             episode_epsilon=np.zeros(episodes))
        ############################
        # opt: each 20e continuos ploting (only works without bohb)
        # cont_plot = continous_plot()
        ############################
        # Loop over episodes
        eps = self.epsilon
        for e in range(episodes):
            # reduce epsilon by decay rate
            if self.eps_scheduler == 'cos':
                eps = cos_ann_w_restarts(e, self.n_episodes,
                             self.n_cycles, self.epsilon)
            elif self.eps_scheduler == 'exp':
                eps = exponential_decay_w_restarts(e, self.n_episodes,
                                                   self.n_cycles,
                                                   1, 0.03, conf['decay_rate'])
            stats.episode_epsilon[e] = eps
            ############################
            # opt: each 20e continuos ploting (only works without bohb)
            #if e % 20 == 0:
            #    cont_plot.plot_stats(stats)
            ############################
            stats.episode_lengths[e] = 0
            s = env.reset()
            for t in range(time_steps):
                ############################
                # opt: render env every 5 episodes
                #if e % 5 == 0:
                #    env.render()
                ############################
                # act and get results
                a = self.get_action(s, eps)
                ns, r, d, _ = env.step(self.action.act(a))
                ns[2] = angle_normalize(ns[2])
                stats.episode_rewards[e] += r
                self._replay_buffer.add_transition(s, a, ns, r, d)
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = self._replay_buffer.random_next_batch(self.bs)  # NOQA
                # get actions of Target network
                target = (batch_rewards
                          + (1 - batch_terminal_flags)
                             * self._gamma
                             * self._q_target(batch_next_states)[
                                    torch.arange(conf['bs']).long(),
                                    torch.argmax(self._q(batch_next_states),
                                    dim=1)])
                # get actions of value network
                current_prediction = self._q(batch_states)[
                                        torch.arange(self.bs).long(),
                                        batch_actions.long()]
                ############################
                # Update acting network
                loss = self._loss_function(current_prediction, target.detach())
                stats.episode_loss[e] += loss.cpu().detach()
                self._q_optimizer.zero_grad()
                loss.backward()
                self._q_optimizer.step()
                # Update target network
                soft_update(self._q_target, self._q, 0.01)
                ############################
                # stop episode if carte leaves boundaries
                if d:
                    stats.episode_lengths[e] = t
                    break
                s = ns
            ############################
            # if episode didn't failed, time is maximal time
            if stats.episode_lengths[e] == 0:
                stats.episode_lengths[e] = time_steps
        return stats

class poleWorker(Worker):
    def __init__(self, **kwargs):
            super().__init__(**kwargs)

    def compute(self, config, budget, working_directory, *args, **kwargs):
            """
            Simple example for a compute function using a feed forward network.
            It is trained on the MNIST dataset.
            The input parameter "config" (dictionary) contains the sampled
            configurations passed by the bohb optimizer
            """
            env = ContinuousCartPoleEnv(reward_function=smooth_reward)
            state_dim = env.observation_space.shape[0]
            # Try to ensure determinism
            ############################
            torch.manual_seed(config['seed'])
            env.seed(config['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            ############################
            # conf dictionary to controll training
            conf = {'lr':config['lr'], 'bs':64, 'loss':nn.MSELoss(),
                    'hidden_dim':config['hidden_dim'],
                    'mem_size':1e6, 'activation':config['activation'],
                    'epsilon':config['epsilon'],
                    'eps_scheduler':'exp', 'n_episodes':budget,
                    'dropout_rate': 0.0, 'n_cycles': 1,
                    'decay_rate': config['decay_rate']
                  }
            ############################
            # create dqn object and train it
            dqn = DQN(state_dim, config['action_dim'],
                      gamma=config['gamma'], conf=conf)
            time_steps = 1000
            stats = dqn.train(int(budget), time_steps, env, conf)
            # plot_episode_stats(stats, noshow=True)
            env.close()
#           ###########################
            return ({
                     # remember: HpBandSter always minimizes!
                    'loss': -max(stats.episode_rewards),
                    'info': {'max_len': max(stats.episode_lengths),
                             'max_reward': max(stats.episode_rewards) }
            })


    @staticmethod
    def get_configspace():
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of
            hyperparameters. Beside float-hyperparameters on a log scale, it
            is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            cs = CS.ConfigurationSpace()
            if True:
                lr = CSH.CategoricalHyperparameter('lr',[0.001])
                hidden_dim = CSH.CategoricalHyperparameter('hidden_dim',
                        [16, 32, 64, 128, 256])
                activation = CSH.CategoricalHyperparameter('activation',
                        ['tanh', 'relu'])
                epsilon =  CSH.UniformFloatHyperparameter('epsilon',
                        lower=0.01, upper=1., default_value=1., log=True)
                decay_rate =  CSH.UniformFloatHyperparameter('decay_rate',
                        lower=0.001, upper=.1, default_value=.01, log=True)
                gamma = CSH.CategoricalHyperparameter('gamma',[0.99])
                action_dim = CSH.UniformIntegerHyperparameter('action_dim',
                        lower=3, upper=20)
                seed = CSH.UniformIntegerHyperparameter('seed',
                        lower=0, upper=1000, default_value=42)
                dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate',
                        lower=0.0, upper=0.2, default_value=0.0)
            else:
                lr = CSH.UniformFloatHyperparameter('lr',
                        lower=0.001, upper=0.0012, default_value=0.001,
                        log=True)
                hidden_dim = CSH.CategoricalHyperparameter('hidden_dim',
                        [16, 32, 64, 128, 256])
                activation = CSH.CategoricalHyperparameter('activation',
                        ['tanh', 'relu'])
                epsilon =  CSH.UniformFloatHyperparameter('epsilon',
                        lower=0.01, upper=1., default_value=1., log=True)
                decay_rate =  CSH.UniformFloatHyperparameter('decay_rate',
                        lower=0.001, upper=.1, default_value=.01)
                gamma = CSH.UniformFloatHyperparameter('gamma',
                        lower=0.5, upper=0.999, default_value=0.99, log=True)
                action_dim = CSH.UniformIntegerHyperparameter('action_dim',
                        lower=3, upper=100)
                seed = CSH.UniformIntegerHyperparameter('seed',
                        lower=0, upper=1000, default_value=42)
                dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate',
                        lower=0.0, upper=0.3, default_value=0.0)
            cs.add_hyperparameters([lr, hidden_dim, activation, epsilon,
                                    gamma, action_dim, seed, dropout_rate,
                                    decay_rate,])

            return cs

############################
# parsing
parser = argparse.ArgumentParser(
    description='Rl-Project DQN Approach')
parser.add_argument('--min_budget',   type=float,
        help='Minimum budget used during the optimization.',    default=20)
parser.add_argument('--max_budget',   type=float,
        help='Maximum budget used during the optimization.',    default=200)
parser.add_argument('--n_iterations', type=int,
        help='Number of iterations performed by the optimizer', default=5)
parser.add_argument('--n_workers', type=int,
        help='Number of workers to run in paralell', default=1)
args=parser.parse_args()
############################
# nameserver variables
run_id = 'rl_project_ftw'
nameserver_name = '127.0.0.1'
############################
# False: run Bohb, True run only on sampled config (for debugging)
if False:
    w = poleWorker( nameserver=nameserver_name,run_id=run_id, id=0)
    cs = w.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = w.compute(config=config, budget=20, working_directory='.')
    print(res)
else:
    if not os.path.exists('./bohb_res'):
        os.makedirs('./bohb_res')
    result_logger = hpres.json_result_logger(directory='./bohb_res',
                                             overwrite=True)
    NS = hpns.NameServer(run_id=run_id, host=nameserver_name, port=None)
    NS.start()
    ############################
    # create workers for threaded paralell runs
    workers=[]
    for i in range(args.n_workers):
        w = poleWorker( nameserver=nameserver_name,run_id=run_id, id=i)
        w.run(background=True)
        workers.append(w)
    ############################
    # create logged bohb object and run it
    bohb = BOHB(  configspace = w.get_configspace(),
                  run_id = run_id,
                  min_budget=args.min_budget, max_budget=args.max_budget,
                  result_logger=result_logger,
               )
    res = bohb.run(n_iterations=args.n_iterations,
                   min_n_workers=args.n_workers)
    ############################
    # kill bohb object with workers
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    ############################
    # show the results concisely for plotting run the plot_bohb.py
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    all_runs = res.get_all_runs()
    # save the res, for visualization
    with open('./bohb_res/res.pkl', 'wb') as f:
        pickle.dump(res, f)
    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(
          id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(
          sum([r.budget for r in all_runs])/args.max_budget))
    print('Total budget corresponds to %.1f full function evaluations.'%(
          sum([r.budget for r in all_runs])/args.max_budget))
    print('The run took  %.1f seconds to complete.'%(
         (all_runs[-1].time_stamps['finished']
          - all_runs[0].time_stamps['started'])))
