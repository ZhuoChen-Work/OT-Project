from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

class Gaussian_distribution():
  def __init__(self,mean,std,dim):
    
    self.dim = dim
    self.mean = mean
    self.std = std
    assert len(self.mean)==self.dim
    
  def generate_gaussian(self,n_sample=1000):   
    cov = np.diag(np.ones(self.dim))*self.std
    return  np.random.multivariate_normal(self.mean, cov, n_sample)


class TwoMoon():
  def __init__(self,n_samples=1000,noise=0.05,ratio=0.2,random_state=0):
    X,y = datasets.make_moons(n_samples=1000,noise=0.05,random_state=random_state)
    X = StandardScaler().fit_transform(X)

    self.source = X[y==0]
    self.target = X[y==1]

    np.random.seed(random_state)
    np.random.shuffle(self.source)
    np.random.seed(random_state)
    np.random.shuffle(self.target)

    self.source_train = self.source[:int(0.4*len(self.source))]
    self.target_train = self.target[:int(0.4*len(self.target))]

    self.source_test = self.source[int(0.4*len(self.source)):]
    self.target_test = self.target[int(0.4*len(self.target)):]

def data_noise(x,std=0.15,seed=0,device='cuda:0'):
  if type(x) != torch.Tensor:
    x = torch.Tensor(x) 

  torch.manual_seed(seed)
  gaussian = torch.normal(0,std,x.shape).to(device)
  temp = x.to(device)+gaussian 
  return temp
