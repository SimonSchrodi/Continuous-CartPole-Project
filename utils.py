import torch
from torch.autograd import Variable
import numpy as np

cuda = False

def tt(ndarray):
  if cuda:
    return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
  return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)

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