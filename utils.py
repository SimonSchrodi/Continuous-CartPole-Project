import torch
from torch.autograd import Variable
import numpy as np

#cuda = False
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def tt(ndarray):
  return Variable(torch.from_numpy(ndarray).float().to(device), requires_grad=False)

def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
  soft_update(target, source, 1.0)

def reward(cart_pole):
  x_threshold= 2.4
  if cart_pole.state[0] < -x_threshold or cart_pole.state[0] > x_threshold:
    return -500
  from continuous_cartpole import angle_normalize
  normalized_angle = angle_normalize(cart_pole.state[2])
  special_sauce = 2 if -0.1 <= angle_normalize(cart_pole.state[2]) <= 0.1 else 1
  #return special_sauce*(1-np.abs(normalized_angle/np.pi)) + 0.01 - 0.2*np.abs(cart_pole.state[0]/x_threshold)
  #return 5*(1-np.abs(normalized_angle/np.pi)) + 0.2
  return 1

def smooth_reward(cart_pole):
  x_threshold= 2.4
  if cart_pole.state[0] < -x_threshold or cart_pole.state[0] > x_threshold:
    return -10
  normalized_angle = angle_normalize(cart_pole.state[2])
  return 1 if -0.1 <= angle_normalize(cart_pole.state[2]) <= 0.1 else 0.001

def angle_normalize(x):
  return (((x+np.pi) % (2*np.pi)) - np.pi)

def cos_ann_w_restarts(epoch, n_epochs, n_cycles, lr_max, lr_min=0.01, subtract=0.):
  epochs_per_cycle = np.floor(n_epochs / n_cycles)
  cos_inner = (np.pi * ((epoch - 1) % epochs_per_cycle)) / epochs_per_cycle
  return max(lr_min, (lr_max / 2) * (np.cos(cos_inner) + 1) - subtract)

def exponential_decay_w_restarts(epoch, n_epochs, n_cycles, lr_max, lr_min, decay_rate):
  epochs_per_cycle = np.floor(n_epochs / n_cycles)
  return max(lr_min, lr_max * np.power(np.e,- decay_rate * epoch))

def plot_episode_stats(stats, smoothing_window=10, noshow=False, count=0):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    fig1.savefig('episode_lengths_{}.png'.format(count))
    if noshow:
            plt.close(fig1)
    else:
            plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    fig2.savefig('reward_{}.png'.format(count))
    if noshow:
            plt.close(fig2)
    else:
            plt.show(fig2)

class acton_discrete():
    ''' Class to normalize normal numbers (actions) to [-1,1] '''
    def __init__(self, actions):
        self.n_actions = actions - 1

    def act(self, action):
        return np.array([2 * action / (self.n_actions) - 1])

class continous_plot:
    ''' Class to plot episode lengths, rewards and losses continuosly averaged over
        20 episodes'''
    def __init__(self):
        # continuos result plotting
        self.fig, self.ax1 = plt.subplots(figsize=(10,6))  # episode lengths
        self.ax2 = self.ax1.twinx()  # rewards
        self.ax3 = self.ax1.twinx() # network losses
        self.ax4 = self.ax1.twinx() # eps
        self.fig.show()
        self.fig.canvas.draw()

    def plot_stats(self, s):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        # episode lengts
        color = 'r'
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Episode Length', color=color)
        lengths_smoothed = pd.Series(s.episode_lengths).rolling(20, min_periods=20).mean()
        self.ax1.plot(lengths_smoothed, color=color)
        self.ax1.tick_params(axis='y', labelcolor=color)
        # rewards
        color = 'b'
        self.ax2.set_ylabel('Episode Reward', color=color)  # we already handled the x-label with ax1
        rewards_smoothed = pd.Series(s.episode_rewards).rolling(20, min_periods=20).mean()
        self.ax2.plot(rewards_smoothed, color=color)
        self.ax2.tick_params(axis='y', labelcolor=color)
        self.ax2.spines['right'].set_position(('outward', 0))
        # losses
        color = 'g'
        self.ax3.set_ylabel('Episode Loss', color=color)  # we already handled the x-label with ax1
        loss_avg = np.divide(s.episode_loss, s.episode_lengths)
        loss_avg = pd.Series(loss_avg).rolling(20, min_periods=20).mean()
        self.ax3.plot(loss_avg, color=color)
        self.ax3.tick_params(axis='y', labelcolor=color)
        self.ax3.spines['right'].set_position(('outward', 60))
        # epsilon
        color = 'black'
        self.ax4.set_ylabel('Episode Epsilon', color=color)  # we already handled the x-label with ax1
        self.ax4.plot(s.episode_epsilon, color=color)
        self.ax4.tick_params(axis='y', labelcolor=color)
        self.ax4.spines['right'].set_position(('outward', 120))
        self.fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid()
        self.fig.show()
        self.fig.canvas.draw()
