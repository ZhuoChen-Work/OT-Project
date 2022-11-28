""" Author Zhuo Chen"""

import torch
import torch.nn as nn
import ot


device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'


class ICNN(nn.Module):
  def __init__(self,input_dim, dim_hidden=[100,100,100], device=device):    
    super(ICNN,self).__init__()
    """ 
    Input convex neural network:  
    w_zs and w_xs are the weights in ICNN model, where w_zs should be positive during the training
    the number of units in each layer of ICNN is (input_dim, 1 ,dim_hidden[0],dim_hidden[1],...,dim_hidden[-1], 1 );  

    the shapes of w_zs are (1,dim_hidden[0]),(dim_hidden[0],dim_hidden[1]),...,(dim_hidden[-2],dim_hidden[-1]),(dim_hidden[-1],1)   
    bias = False
    the initial weight of w_zs is 1/dim_hidden[i]
    len(w_zs) = dim_hidden+1;    

    the shapes of w_xs are (input_dim,input_dim),(input,dim_hidden[0]),...,(input_dim, dim_hidden[-1]),(inpit_dim,1)
    bias = True
    Initializing weight,bias with Gaussian  
    len(w_xs) = dim_hidden+2.   
    """
    self.input_dim = input_dim
    self.num_hidden = len(dim_hidden)
    self.dim_hidden = dim_hidden

    self.w_zs = nn.ModuleList()
    for i in range(self.num_hidden):
      if i==0:
        self.w_zs.append( nn.Linear(1,self.dim_hidden[i],bias = False) )
        self.w_zs[i].weight.data.fill_(1.)
      else:
        self.w_zs.append( nn.Linear( self.dim_hidden[i-1],self.dim_hidden[i], bias=False,) )
        self.w_zs[i].weight.data.fill_(1./self.dim_hidden[i-1])
    
    self.w_zs.append(nn.Linear( self.dim_hidden[-1] , 1 ,bias = False ))
    self.w_zs[-1].weight.data.fill_(1./self.dim_hidden[-1])

    self.w_xs = nn.ModuleList()
    self.w_xs.append( nn.Linear(input_dim,input_dim,bias= True))
    for i in range(self.num_hidden):
      self.w_xs.append(nn.Linear(input_dim,dim_hidden[i],bias=True))

    self.w_xs.append(nn.Linear(input_dim,1,bias=True))

    for l in range(len(self.w_xs)):
      nn.init.normal_(self.w_xs[l].weight,std=0.1)
      nn.init.normal_(self.w_xs[l].bias,std=0.1)



  def forward(self,x):
    """
    the output is a convex function of x.
    """
    assert x.shape[1] == self.input_dim
    
    # initializinng z with form (x-b)^T@ (AA^T) @(x-b)  
    z = 0.5* torch.sum(((x-self.w_xs[0].bias)@self.w_xs[0].weight)**2,dim=-1).reshape(-1,1)

    for i in range(self.num_hidden):
      z = self.w_zs[i](z)+self.w_xs[i+1](x)
      z = torch.relu(z)
    
    z = self.w_zs[-1](z)+self.w_xs[-1](x)
    return z



class LSE(nn.Module):
  def __init__(self,input_dim,hid_dim=200,device=device,alpha=0.5):
    """
    Log Sum Exp 
    """
    super(LSE, self).__init__()
    self.W = nn.Linear(input_dim,hid_dim,bias=True)
    self.alpha = alpha
    
  def forward(self,x):
    """
    The forward function consist of two parts: 1) the standard LSE function; 2) alpha*||x||^2
    """
  # try to make it quadratic
    out = torch.logsumexp(self.W(x),dim=-1)+torch.sum(self.alpha*(x**2),dim=-1)
    return out

  def transport(self,x):
    """
    According to Brenier theorem, the optimal transport mapping is the gradiant of the convex forward function
    """
    temp = torch.clone(x)
    temp.requires_grad_()
    loss=self.forward(temp)
    loss.backward(torch.ones_like(loss))
    return temp.grad

class c_transform(nn.Module):
  def __init__(self,train_target,convex_func):
    """
    C-transform 
    https://arxiv.org/pdf/1910.03875.pdf,  formula(20),(21)
    """
    super(c_transform, self).__init__()
    self.target = train_target
    self.init_target = train_target
    self.convex_func = convex_func
    
  def phi_Brenier(self,x):  
    """
    phi_Brenier is a convex function
    """
    # size n 
    return self.convex_func(x).reshape(-1,1)
    
  def phi_Kantorovich(self,x):
    """
    phi_kantorovich = 0.5||x||^2 - phi_Brenier
    """
    # size n x 1
    return 0.5*torch.sum(x**2,dim=-1).reshape(-1,1) - self.phi_Brenier(x) 

  def phi_c(self,x):
    """
    C-transform function of x
    """
    DD = 0.5*torch.cdist(x,self.target)**2  # size n x m
    return torch.amin(DD - self.phi_Kantorovich(x).reshape(-1,1),dim=0) # size m
  
  def transport(self,x): 
    """
    According to Brenier theorem, the optimal transport mapping is the gradiant of the convex forward function
    size = nxd
    """
    temp = torch.clone(x)
    temp.requires_grad_()
    loss=self.phi_Brenier(temp)
    loss.backward(torch.ones_like(loss))   
    return temp.grad

  def dual(self,x):
    """
    the form of kantorovich dual, which needs to be maximised
    """
    return self.phi_Kantorovich(x).mean()+self.phi_c(x).mean()


class MLP(nn.Module):
  def __init__(self,input_dim,dim_hidden=[100,100,100]):
    """
    Standard  MLP model
    """
    super(MLP,self).__init__()
    self.input_dim = input_dim
    self.num_hidden = len(dim_hidden)
    self.dim_hidden = dim_hidden

    self.w = nn.ModuleList()
    self.w.append(nn.Linear(self.input_dim,self.dim_hidden[0]))
    for i in range(self.num_hidden-1):
      self.w.append( nn.Linear(dim_hidden[i],self.dim_hidden[i+1],bias= True))

    self.w.append(nn.Linear(self.dim_hidden[-1],self.input_dim,bias= True))


    for l in range(len(self.w)):
      nn.init.normal_(self.w[l].weight,std=0.1)
      nn.init.normal_(self.w[l].bias,std=0.1)

  def forward(self,x):
    z = self.w[0](x)
    for l in range(1,len(self.w)):
      z = torch.relu(z)
      z = self.w[l](z)

    return z


def LSE_train(lse_model,device, train_loader,optimizer, epoch ,print_bool=True):
    """
    Training lse C-Transform model, loss is the wasserstein dual form
    """
    model = lse_model.to(device)
    for ep in range(epoch):
      for data in train_loader:
          data = data.float().to(device)

          optimizer.zero_grad()
          loss = -model.dual(data)
          loss.backward()
          optimizer.step() 

      if ep%500==0 and print_bool:
        print(ep,model.dual(data).data)


def ICNN_train(model_icnn,device, train_loader,optimizer, epoch,print_bool=True):
    """
    Training ICNN C-Transform model, loss is the wasserstein dual form
    """
    model = model_icnn.to(device)
    for ep in range(epoch):
      for data in train_loader:
          data = data.float().to(device)

          optimizer.zero_grad()
          loss = -model.dual(data)
          loss.backward()
          optimizer.step()

          for l in range(len(model.convex_func.w_zs)):
            model.convex_func.w_zs[l].weight.data.copy_(torch.relu(model.convex_func.w_zs[l].weight.data))

      if ep%100==0 and print_bool:
        print(ep,model.dual(data).data)


def MLP_train(model_mlp,device, train_loader, target, optimizer, epoch,print_bool=False):
    """
    Training MLP moldel, loss is the sum of squared error.
    """
    model = model_mlp.to(device)

    for data in train_loader:
          temp_data = data.float().to(device)
          temp_cp = get_coupling(temp_data,target).to(device)

    for ep in range(epoch):
      for data in train_loader:
          temp_data = data.float().to(device)

          optimizer.zero_grad()

          fake_target = model(temp_data)

          # barycenter method
          loss= (temp_cp* torch.cdist(fake_target,target)**2).sum()
          
          loss.backward()
          optimizer.step()

      if ep%100==0 and print_bool:
        print(ep,loss.data)


def get_coupling(source,target,device=device):
  """
  compute the groud truth optimal transport mapping with package OT
  """
  M = 0.5*ot.dist(source, target, 'sqeuclidean').to(device)
  A = ot.emd((torch.ones(len(source))/len(source)).to(device), 
           (torch.ones(len(target))/len(target)).to(device), M, numItermax=1e6)
  return A

def get_wdist(source,target,device=device):
  """
  compute the ground truth Wasserstein Distance
  """
  M = 0.5*ot.dist(source, target, 'sqeuclidean').to(device)
  wdist = ot.emd2((torch.ones(len(source))/len(source)).to(device), 
           (torch.ones(len(target))/len(target)).to(device), M, numItermax=1e6)
  return wdist


def mix_train( model_icnn,device, train_loader, optimizer, epoch,alpha=0,print_bool=True):
  """
  hybrid model of ICNN C-transoform and MLP
  when alpha=0, it is standard ICNN C-transform
  """
  temp_cp = None
  # C_Transform ICNN
  model = model_icnn.to(device)    
  for ep in range(epoch):
    for data in train_loader:
        temp_data = data.float().to(device) # Training source
        optimizer.zero_grad()

        fake_target = model.transport(temp_data)  # by grad of Phi_B(from icnn) 
        if temp_cp == None:      
          temp_cp = get_coupling(temp_data,model.target).to(device)


        loss1= (temp_cp* torch.cdist(fake_target,model.target)**2).sum()

        optimizer.zero_grad()
        loss2 = -model.dual(temp_data)
        loss = alpha*loss1+(1-alpha)*loss2

        loss.backward()
        optimizer.step()

        for l in range(len(model.convex_func.w_zs)):
           model.convex_func.w_zs[l].weight.data.copy_(torch.relu(model.convex_func.w_zs[l].weight.data))

    if ep%100==0 and print_bool:
      print(ep,loss1.data,-loss2.data)
