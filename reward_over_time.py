import matplotlib.pyplot as plt
import numpy as np

def plot(states1, states2):
    ''' Plots the mean, and variances of several different runs'''
    rewards = states1[:,1]

    mean = np.mean(rewards,axis=0)
    std = np.std(rewards,axis=0)

    #plt.rc('text', usetex=True)
    plt.rc('font', size=15.0, family='serif')
    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    plt.plot(np.arange(400), np.ones(400) * 1000, label='Max Possible Reward', c='black')
    #plt.plot(np.arange(400), mean, label='Sparse Rewards ($\sigma=$'+str(round(std.mean(),2))+')',c='blue')
    #plt.fill_between(np.arange(400), mean + std, mean - std, color="blue", alpha=0.2)

    rewards = states2[:, 1]

    mean = np.mean(rewards, axis=0)
    std = np.std(rewards, axis=0)
    plt.plot(np.arange(400), mean, label='Smoothed Rewards ($\sigma=$'+str(round(std.mean(),2))+')', c='red')
    plt.fill_between(np.arange(400), mean + std, mean - std, color="red", alpha=0.2)

    #plt.legend(loc='best')
    plt.legend(loc=4)
    plt.xlabel('Epochs')
    plt.ylabel('Reward average over 6 runs')

    plt.show()


if __name__ == '__main__':
    sparse = np.load('sparse_rewards.npy')
    own = []
    for name in ['smooth_rewards1run.npy','smooth_rewards2run.npy','smooth_rewards3run.npy',
                 'smooth_rewards4run.npy','smooth_rewards5run.npy','smooth_rewards6run.npy']:
        own.append(np.load(name))

    own = np.array(own).reshape((6,4,400))
    plot(sparse,own)
