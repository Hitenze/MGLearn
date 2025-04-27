# this file contains the pytorch module for the PINNs module, techniqually just a simple fully connected neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_theme()
from tqdm.auto import tqdm
from IPython import get_ipython
import gc

def using_notebook():
    try:
        if 'IPKernelApp' not in get_ipython().config:
            return False
        return True
    except Exception:
        return False

class Relu3(torch.nn.Module):
    def __init__(self):
        super(Relu3, self).__init__()

    def forward(self, x):
        return torch.maximum(torch.tensor(0.0, device=x.device, dtype=x.dtype), 1/6 * x**3)

def reset_seed(seed):
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)

class GPINNs_Model(nn.Module):
   '''
   The nn.Module for multiscale green's pinns.
   The activation is in the form $eps^\alpha * activation(x/(eps^\beta))$
   '''
   def __init__(self, 
                  dim,
                  nunknown,
                  nhidden,
                  nlayers,
                  eps = 1.0,
                  alpha = 1.0,
                  beta = 1.0,
                  use_diff = False,
                  activation = 'Default',
                  dtype = torch.float64,
                  device = torch.device('cpu'),
                  init_method = 'xavier',
                  init_seed = 815,
                  print_info = False
                  ):
      '''
      The init function for GPINNs_Model.
      Inputs:
         dim: Integer. 
            Dimension of the PDE.
         nunknown: Integer. 
            Number of unknowns.
         nhidden: Integer. 
            Number of neurals at each hidden layers.
         nlayers: Integer. 
            Number of hidden layers.
         eps: Float.
            The eps in the activation function.
         alpha: Float.
            The alpha in the activation function.
         beta: Float.
            The beta in the activation function.
         use_diff: Bollean.
            Do we also include the difference between x and y in the network.
         activation: String.
            Activation function.
         dtype: Torch dtype.
            Double or single.
         device: String.
            Device for pytorch to use.
         init_method: String.
            Initialization method for the network.
         init_seed: Integer.
            Random seed for model initialization.
         print_info: Boolean.
            Do we print model initialization info.
      '''
      super().__init__()
      # save params
      self._dim = dim
      self._nunknown = nunknown
      self._nhidden = nhidden
      self._nlayers = nlayers
      self._eps = eps
      self._alpha = alpha
      self._beta = beta
      self._alpha_eps = eps ** alpha
      self._beta_eps = 1.0 / (eps ** beta)
      self._use_diff = use_diff
      self._dtype = dtype
      self._device = device
      self._init_method = init_method
      self._init_seed = init_seed
      self._print_info = print_info
      # define the newtork structure
      self._nin = dim * 3 if self._use_diff else dim * 2
      self._nout = nunknown
      self._net = nn.ModuleList()
      self._net.append(nn.Linear(self._nin, self._nhidden, dtype = self._dtype))
      for _ in range(self._nlayers-1):
         self._net.append(nn.Linear(self._nhidden, self._nhidden, dtype = self._dtype))
      self._net.append(nn.Linear(self._nhidden, self._nout, dtype = self._dtype))
      self._nparams = self.get_num_params()

      if activation in dir(nn):
         self._activation = eval(f'nn.{activation}()')
      else:
         self._activation = nn.Tanh()

      reset_seed(self._init_seed)

      # xavier_init
      if self._init_method == 'xavier':
         # xavier initialization
         for i in range(nlayers):
            torch.nn.init.xavier_normal_(self._net[i].weight)
            torch.nn.init.zeros_(self._net[i].bias)
      elif self._init_method == 'random':
         # random initialization
         for i in range(nlayers):
            torch.nn.init.normal_(self._net[i].weight, 0, 1)
            torch.nn.init.zeros_(self._net[i].bias)
      elif self._init_method == 'zero':
         # zero initialization
         for i in range(nlayers):
            torch.nn.init.zeros_(self._net[i].weight)
            torch.nn.init.zeros_(self._net[i].bias)
      else:
         # no specific initialization
         pass
      self.to(self._device)

      if self._print_info:
         self.print_info()

   def update_eps(self, eps):
      '''
      Update the eps value for the model.
      '''
      self._eps = eps
      self._alpha_eps = eps ** self._alpha
      self._beta_eps = 1.0 / (eps ** self._beta)

   def print_info(self):
      '''
      Print the model information to putput
      '''
      print(f'Printing model information:')
      print(f'Number of parameters: {self.get_num_params()}')
      print(f'Number of layers: {self._nlayers}')
      print(f'Number of hidden nodes: {self._nhidden}')
      print(f'Eps value: {self._eps}')
      print(f'Alpha value: {self._alpha}')
      print(f'Beta value: {self._beta}')
      print(f'Use difference: {self._use_diff}')
      print(f'Activation function: {self._activation}')
      print(f'Initialization: {self._init_method}')
      print(f'Device: {self._device}')
      print(f'Dtype: {self._dtype}')

   def get_num_params(self):
      '''
      Return the number of parameters in the network.
      Output:
         num_params: Integer.
            Number of parameters in the network.
      '''
      if hasattr(self, '_nparams'):
         return self._nparams
      else:
         model_parameters = filter(lambda p: p.requires_grad, self.parameters())
         num_params = 0
         for p in model_parameters:
            num_params += np.prod(p.size() + (2,) if p.is_complex() else p.size())
         return num_params

   def save(self, path):
      torch.save(self.state_dict(), path)

   def load(self, path, model_name=""):
      '''
      Return whether the model is loaded successfully or not
      '''
      if os.path.exists(path):
         print("Model %s loaded from file: %s" % (model_name, path))
         self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
         return True
      else:
         print("Model %s create new model" % model_name)
         return False

   def forward(self, x):
      if self._use_diff:
         # we add x[0:self._dim,:]-x[self._dim:,:] as another input to the network
         x = torch.cat((x, x[:,0:self._dim]-x[:,self._dim:]), dim=1)
      # only apply beta to the first layer and alpha to the last layer
      x = self._activation(self._net[0](x)*self._beta_eps)
      for i in range(1,self._nlayers):
         x = self._activation(self._net[i](x))
      x = self._net[-1](x*self._alpha_eps)
      return x


class MSGPINNs:
   def __init__(self,
                  pde,
                  nunknown,
                  gdomain,
                  eps = 1e-01,
                  r_near = 2.0,
                  nsamples = (0, 1, 100, 200, 500),
                  use_diff = False,
                  model_params = [(50,5)],
                  model_activation = ['Default'],
                  model_init_method = ['xavier'],
                  model_print_info = [True],
                  model_name_base = ['single'],
                  model_state_file_base = [None],
                  sol = None,
                  bc = None,
                  init_seed = [815],
                  alpha = [1.0],
                  beta = [1.0],
                  dtype=torch.float32,
                  device=torch.device('cpu'),
                  on_the_fly = False):
      '''
      pde: function.
         The PDE operator.
      nunknown: Integer.
         The number of unknown functions.
      gdomain: geometry class.
         The computation domain.
      eps: Float. Optional, default 1e-01.
         The eps for approximation of the delta function.
      r_near: Float. Optional, default 2.
         The ratio for radius for sampling near each y value (r_near * eps).
      nsamples: Tuple of integers of length 5. Optional, default (0, 1, 100, 200, 500).
         In the order (y_onquad, y_random, x_bc, x_near, x_far) for number of y on quad points, number of random y, bc for each y, near points for each y, and far points for each y.
      use_diff: Booleans. Default False.
         Set to True to include x-y as the input for GPINNs model.
      model_params: Tuple of tuple of size 2. Default [(50,5)].
         Each entry is the parameter option for a model as (num_neurons, num_hidden_layers).
      model_activation: Tuple of Strings. Optional. Default ['Tanh'].
         The activation of each model.
      model_init_method: Tuple of strings. Optional. Default ['xavier'].
         The initialization method for each model.
      model_print_info: Tuple of booleans. Default [True].
         Set to True to print model info to output.
      model_name_base: Tuple of strings. Default ['single'].
         The name base of the model. For DD the list of models are in the format name_base + number.
      model_state_file_base: Tuple of strings or None. Default [None].
         If not None, the file path base for saving the model.
      sol: Function. Default None.
         If not None, the true Greens function for error calculation.
      bc: Function. Default None.
         The boundary condition. If None, zero boundary condition is used.
      init_seed: Tuple of integers. Default [815].
         Random seed for model parameter initilization.
      alpha: Tuple of float. Default [1.0].
         Alpha value for each model.
      beta: Tuple of float. Default [1.0].
         Beta value for each model.
      dtype: Torch datatype. Default torch.float32
         Datatype for models.
      device: Torch device. Default torch.device('cpu').
         Default device for models.
      on_the_fly: Boolean. Default False.
         If True, the model is loaded only when used.
      '''

      # 1. Problem info
      self._pde = pde
      self._gdomain = gdomain
      self._ndomains = self._gdomain._ndomains
      self._dim = self._gdomain.get_dim()
      self._nunknown = nunknown
      self._sol = sol
      self._bc = bc
      self._eps = eps
      self._r_near = r_near
      self._nsamples = nsamples
      self._nsample_y_onquad = nsamples[0]
      self._nsample_y_random = nsamples[1]
      self._nsample_x_bc = nsamples[2]
      self._nsample_x_near = nsamples[3]
      self._nsample_x_far = nsamples[4]
      self._nsamples_interior_total = (self._nsample_y_onquad + self._nsample_y_random) * (self._nsample_x_near + self._nsample_x_far)
      self._use_diff = use_diff

      # 2. Model info
      self._nmodels = len(model_params)
      self._init_seed = init_seed
      self._models_params = model_params
      self._models_lock = [True] * self._ndomains
      self._models_inuse = [False] * self._ndomains
      self._models_activation = model_activation
      self._models_init_method = model_init_method
      self._models_print_info = model_print_info
      self._models_name_base = model_name_base
      self._models_alpha = alpha
      self._models_beta = beta
      
      # 3. Other info
      self._dtype = dtype
      self._device = device
      self._on_the_fly = on_the_fly

      # 4. Create Model
      self._using_notebook = using_notebook()

      self._models = []
      self._models_state_file_base = []
      for i in range(self._nmodels):
         if model_state_file_base[i] is None:
            if self._using_notebook: 
               self._models_state_file_base.append(os.path.join(os.path.abspath(''),'model' + str(i)))
            else:
               self._models_state_file_base.append(os.path.join(os.path.dirname(__file__),'model' + str(i)))
         else:
            self._models_state_file_base.append(model_state_file_base[i])

      self._models = []
      self._models_names = []
      self._models_state_files = []
      if self._on_the_fly:
         # only create the model when use, so we do not create the model here
         for j in range(self._nmodels):
            models_j = []
            models_names_j = []
            models_state_file_j = []
            for i in range(self._ndomains):
               models_names_j.append(self._models_name_base[j] + '_' + str(i))
               models_state_file_j.append(self._models_state_file_base[j] + '_' + str(i) + '.pth')
               models_j.append(None)
            self._models.append(models_j)
            self._models_names.append(models_names_j)
            self._models_state_files.append(models_state_file_j)
         
      else:
         for j in range(self._nmodels):
            models_j = []
            models_names_j = []
            models_state_files_j = []
            for i in range(self._ndomains):
               model_j_i = GPINNs_Model(self._dim,
                                          self._nunknown,
                                          self._models_params[j][0],
                                          self._models_params[j][1],
                                          eps = self._eps,
                                          alpha = self._models_alpha[j],
                                          beta = self._models_beta[j],
                                          use_diff = self._use_diff,
                                          activation = self._models_activation[j],
                                          dtype = self._dtype,
                                          device = self._device,
                                          init_method = self._models_init_method[j],
                                          init_seed = self._init_seed[j],
                                          print_info = self._models_print_info[j] if i == 0 else False
                                          )
               state_file_j_i = self._models_state_file_base[j] + '_' + str(i) + '.pth'
               model_name_j_i = self._models_name_base[j] + '_' + str(i)
               isloaded_j_i = model_j_i.load(state_file_j_i, model_name=model_name_j_i)
               if isloaded_j_i:
                  print("Model %s loaded from file: %s", (model_name_j_i, state_file_j_i))
               models_j.append(model_j_i)
               models_names_j.append(model_name_j_i)
               models_state_files_j.append(state_file_j_i)
         
            self._models.append(models_j)
            self._models_names.append(models_names_j)
            self._models_state_files.append(models_state_files_j)

   @staticmethod
   def get_grad(u, x, dim = 1):
      if dim == 1:
         u_grad = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
         u_x = u_grad[:,0]
         u_x_grad = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
         u_xx = u_x_grad[:,0]
         return (u, u_x, u_xx)
      elif dim == 2:
         u_grad = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
         u_x = u_grad[:,0]
         u_y = u_grad[:,1]
         u_x_grad = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
         u_y_grad = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
         u_xx = u_x_grad[:,0]
         u_yy = u_y_grad[:,1]
         u_xy = u_x_grad[:,1]
         return (u, u_x, u_y, u_xx, u_xy, u_yy)
      else:
         print("Dimension not supported")
         return None
   
   def get_sym(self,
               x,
               domi):
      x1 = torch.hstack((x[..., self._dim:], x[..., :self._dim]))
      diff = None
      for modeli in range(self._nmodels):
         if not self._models_inuse[modeli]:
            continue
         if diff is None:
            diff = self._models[modeli][domi](x) - self._models[modeli][domi](x1)
         else:
            diff = diff + self._models[modeli][domi](x) - self._models[modeli][domi](x1)
      return diff
   
   @staticmethod
   def get_delta(x, eps = 1.0, dim = 1):
      if dim == 1:
         return (1/np.sqrt(np.pi)/eps) * torch.exp(-((x[...,0]-x[...,1])**2)/eps**2)
      elif dim == 2:
         return (1/np.pi/eps**2) * torch.exp(-((x[...,0]-x[...,2])**2 + (x[...,1]-x[...,3])**2)/eps**2)
      else:
         tqdm.write("Dimension not supported")
         return None

   def update_device(self, device):
      '''
      Change the defual device of models and domain.
      '''
      self._gdomain.update_device(device)
      self._device = device

   def update_eps(self, eps):
      '''
      Update the eps value for the model.
      '''
      print("Updating eps from %f to %f" % (self._eps, eps))
      self._eps = eps
      for modeli in range(self._nmodels):
         for domi in range(self._ndomains):
            self._models[modeli][domi].update_eps(eps)

   def update_data(self, domi):
      '''
      Update the dataset for training.
      '''
      if self._nsample_y_onquad > 0:
         y_onquad = self._gdomain.sample_uniform(self._nsample_y_onquad,
                                                   on_quad=True, 
                                                   domain_num=domi)
      else:
         y_onquad = None
      if self._nsample_y_random > 0:
         y_rand = self._gdomain.sample_uniform(self._nsample_y_random,
                                                   on_quad=False,
                                                   domain_num=domi)
         if y_onquad is not None:
            y = torch.cat([y_onquad, y_rand], dim=0).detach().requires_grad_(True)
         else:
            y = y_rand.detach().requires_grad_(True)
      x_bc, u_bc = self._gdomain.sample_uniform(self._nsample_x_bc,
                                                boundary=True,
                                                target_y = y)
      x_far = self._gdomain.sample_uniform(self._nsample_x_far,
                                                target_y = y)
      x_near = self._gdomain.sample_uniform(self._nsample_x_near,
                                                rrange = (0.0, self._eps * self._r_near),
                                                target_y = y)
      x_interior = torch.cat([x_near, x_far], dim=0)
      return y, x_interior, x_bc, u_bc

   def get_loss(self, loss, domi, x_interior, x_bc, u_bc, weights):
      '''
      Compute the loss for the given domain.
      '''
      u_bc_approx = None
      for modeli in range(self._nmodels):
         if not self._models_inuse[modeli]:
            continue
         if u_bc_approx is not None:
            u_bc_approx = u_bc_approx + self._models[modeli][domi](x_bc)
            u_interior_approx = u_interior_approx + self._models[modeli][domi](x_interior)
         else:
            u_bc_approx = self._models[modeli][domi](x_bc)
            u_interior_approx = self._models[modeli][domi](x_interior)
      loss_bc = loss(u_bc_approx, u_bc)
      u_interior_grad = MSGPINNs.get_grad(u_interior_approx, x_interior, dim = self._dim)
      loss_pde = loss(self._pde(x_interior, u_interior_grad, lambda x : MSGPINNs.get_delta(x, eps = self._eps, dim = self._dim)), 
                     torch.zeros(self._nsamples_interior_total, dtype=self._dtype).to(self._device))
      loss_sym = loss(self.get_sym(x_interior, domi), torch.zeros(self._nsamples_interior_total, 1, dtype=self._dtype).to(self._device))
      loss_total = weights[0] * loss_bc + weights[1] * loss_pde + weights[2] * loss_sym
      return loss_total, loss_bc, loss_pde, loss_sym

   def plot_data(self, x_bc, x_interior, y, save_figs, fig_file_base, domi):
      '''
      Plot sample dataset for debugging.
      '''
      if self._dim == 1:
         plt.figure(figsize=(10, 1))
         plt.plot(x_bc[...,0].detach().cpu().numpy(), np.zeros(x_bc.shape[0], dtype=np.float64), marker='o', linestyle='None')
         plt.plot(x_interior[...,0].detach().cpu().numpy(), np.zeros(x_interior.shape[0], dtype=np.float64), marker='o', linestyle='None')
         plt.plot(y[...,0].detach().cpu().numpy(), np.zeros(y.shape[0], dtype=np.float64), color='red', marker='o', linestyle='None')
      elif self._dim == 2:
         plt.figure(figsize=(5, 5))
         plt.scatter(x_bc[...,0].detach().cpu().numpy(), x_bc[...,1].detach().cpu().numpy())
         plt.scatter(x_interior[...,0].detach().cpu().numpy(), x_interior[...,1].detach().cpu().numpy())
         plt.scatter(y[...,0].detach().cpu().numpy(), y[...,1].detach().cpu().numpy(), color='red')
      else:
         pass
      if save_figs:
         figname = MSGPINNs.fig_file_rule(fig_file_base, 'train', domi)
         plt.savefig(figname)
      else:
         plt.show()
      plt.close()

   def train(self,
               seed = 815,
               models_lock = [False],
               models_inuse = [True],
               weights = [(10.0, 1.0, 1.0)],
               maxits_adam = 1000,
               maxits_lbfgs = 0,
               data_update_freq = 100,
               print_loss_freq = 100,
               lr_adam_init = 1e-03,
               lr_adam_stepsize = 1000,
               lr_adam_gamma = 1.0,
               lr_lbfgs_init = -1.0,
               lr_lbfgs_stepsize = 1000,
               lr_lbfgs_gamma = 1.0,
               loss_history_file_base = None,
               save_figs = False,
               fig_file_base = None,
               disable_tqdm_bar = False,
               enable_debug = True
             ):
      '''
      Train the entire GPINNs domain by domain.
      Inputs:
         seed: Integer. Default 815.
            Random seed for training.
         models_lock: Tuple of booleans. Default [False].
            Whether to lock the models for training.
         models_inuse: Tuple of booleans. Default [True].
            Whether to use the models for output.
         weights: Tuple of tuple. Default [(10, 1, 1)]
            Weight for each model in the order (bc, pde, sym) for boundary loss, PDE loss, and symmetric loss.
         maxits_adam: Integer. Default 1000.
            Maximum iterations for Adam optimizer.
         maxits_lbfgs: Integer. Default 0.
            Maximum iterations for LBFGS optimizer.
         data_update_freq: Integer. Default 100.
            Frequency of updating the dataset.
         print_loss_freq: Integer. Default 100.
            Frequency of print the loss.
         lr_adam_init: Float. Default 1e-03.
            Initial learning rate for Adam optimizer.
         lr_adam_stepsize: Integer. Default 1000.
            Stepsize for the learning rate scheduler.
         lr_adam_gamma: Float. Default 1.0.
            Gamma for the learning rate scheduler.
         lr_lbfgs_init: Float. Default -1.0.
            Initial learning rate for LBFGS optimizer. If -1.0, the last learning rate of Adam optimizer is used.
         lr_lbfgs_stepsize: Integer. Default 1000.
            Stepsize for the learning rate scheduler.
         lr_lbfgs_gamma: Float. Default 1.0.
            Gamma for the learning rate scheduler.
         loss_history_file_base: String or None. Default None.
            File name for saving the loss history.
         save_figs: Boolean. Default False.
            Whether to save the figures.
         fig_file_base: String or None. Default None.
            File name for saving the figures.
         disable_tqdm_bar: Boolean. Default False.
            Whether to disable the tqdm bar.
      '''
      MSGPINNs.lock_check(models_lock, models_inuse, self._nmodels)
      self._models_lock = models_lock
      self._models_inuse = models_inuse
      loss_res, loss_its = MSGPINNs.load_loss(loss_history_file_base, self._ndomains, maxits_adam, maxits_lbfgs)

      # main loop
      for domi in range(self._ndomains):
         loss_its[domi] = self.train_domi(domi,
                        seed = seed,
                        loss_res_i = loss_res[domi],
                        loss_its_i = loss_its[domi],
                        models_lock = models_lock,
                        models_inuse = models_inuse,
                        weights = weights,
                        maxits_adam = maxits_adam,
                        maxits_lbfgs = maxits_lbfgs,
                        data_update_freq = data_update_freq,
                        print_loss_freq = print_loss_freq,
                        lr_adam_init = lr_adam_init,
                        lr_adam_stepsize = lr_adam_stepsize,
                        lr_adam_gamma = lr_adam_gamma,
                        lr_lbfgs_init = lr_lbfgs_init,
                        lr_lbfgs_stepsize = lr_lbfgs_stepsize,
                        lr_lbfgs_gamma = lr_lbfgs_gamma,
                        loss_history_file_base = loss_history_file_base,
                        save_figs = save_figs,
                        fig_file_base = fig_file_base,
                        disable_tqdm_bar = disable_tqdm_bar,
                        enable_debug = enable_debug
                        )

   def train_domi(self,
                  domi,
                  seed = 815,
                  loss_res_i = None,
                  loss_its_i = None,
                  models_lock = [False],
                  models_inuse = [True],
                  weights = [(10.0, 1.0, 1.0)],
                  maxits_adam = 1000,
                  maxits_lbfgs = 0,
                  data_update_freq = 100,
                  print_loss_freq = 100,
                  lr_adam_init = 1e-03,
                  lr_adam_stepsize = 1000,
                  lr_adam_gamma = 1.0,
                  lr_lbfgs_init = -1.0,
                  lr_lbfgs_stepsize = 1000,
                  lr_lbfgs_gamma = 1.0,
                  loss_history_file_base = None,
                  save_figs = False,
                  fig_file_base = None,
                  disable_tqdm_bar = False,
                  enable_debug = True,
                  fix_y = None,
                  y_sample = None
               ):
      '''
      Train one subdomain. First train with Adam optimizer, then with LBFGS optimizer.
      Inputs:
         domi: Integer.
            Domain number.
         seed: Integer. Default 815.
            Random seed for training.
         loss_res_i: List of list of floats.
            Buffer for loss history.
         loss_its_i: List of integers.
            Buffer for number of iterations.
         models_lock: Tuple of booleans. Default [False].
            Whether to lock the models for training.
         models_inuse: Tuple of booleans. Default [True].
            Whether to use the models for output.
         weights: Tuple of tuple. Default [(10, 1, 1)]
            Weight for each model in the order (bc, pde, sym) for boundary loss, PDE loss, and symmetric loss.
         maxits_adam: Integer. Default 1000.
            Maximum iterations for Adam optimizer.
         maxits_lbfgs: Integer. Default 0.
            Maximum iterations for LBFGS optimizer.
         data_update_freq: Integer. Default 100.
            Frequency of updating the dataset.
         print_loss_freq: Integer. Default 100.
            Frequency of print the loss.
         lr_adam_init: Float. Default 1e-03.
            Initial learning rate for Adam optimizer.
         lr_adam_stepsize: Integer. Default 1000.
            Stepsize for the learning rate scheduler.
         lr_adam_gamma: Float. Default 1.0.
            Gamma for the learning rate scheduler.
         lr_lbfgs_init: Float. Default -1.0.
            Initial learning rate for LBFGS optimizer. If -1.0, the last learning rate of Adam optimizer is used.
         lr_lbfgs_stepsize: Integer. Default 1000.
            Stepsize for the learning rate scheduler.
         lr_lbfgs_gamma: Float. Default 1.0.
            Gamma for the learning rate scheduler.
         loss_history_file_base: String or None. Default None.
            File name for saving the loss history.
         save_figs: Boolean. Default False.
            Whether to save the figures.
         fig_file_base: String or None. Default None.
            File name for saving the figures.
         disable_tqdm_bar: Boolean. Default False.
            Whether to disable the tqdm bar.
         fix_y: Torch tensor of size n by dim or None. Default None.
            If not None, fix the y to the given value.
         y_sample: Function or None. Default None.
            If not None, sample y from the given function as y = y_sample(num_samples).
      '''
      
      MSGPINNs.lock_check(models_lock, models_inuse, self._nmodels)
      self._models_lock = models_lock
      self._models_inuse = models_inuse
      if loss_res_i is None or loss_its_i is None:
         loss_res, loss_its = MSGPINNs.load_loss(loss_history_file_base, self._ndomains, maxits_adam, maxits_lbfgs)
         loss_res_i = loss_res[domi]
         loss_its_i = loss_its[domi]
      
      param_list = []
      if self._on_the_fly:
         # in the on-the-fly mode, we cerate and load the model here
         for modeli in range(self._nmodels):
            if self._models_inuse[modeli]:
               # create the model
               # note that we do not print info here
               self._models[modeli][domi] = GPINNs_Model(self._dim,
                                                         self._nunknown,
                                                         self._models_params[modeli][0],
                                                         self._models_params[modeli][1],
                                                         eps = self._eps,
                                                         alpha = self._models_alpha[modeli],
                                                         beta = self._models_beta[modeli],
                                                         use_diff = self._use_diff,
                                                         activation = self._models_activation[modeli],
                                                         dtype = self._dtype,
                                                         device = self._device,
                                                         init_method = self._models_init_method[modeli],
                                                         init_seed = self._init_seed[modeli],
                                                         print_info = False
                                                         )
               # try to load the model
               self._models[modeli][domi].load(self._models_state_files[modeli][domi], model_name=self._models_names[modeli][domi])
               self._models[modeli][domi].train()
            # add the parameters to the list if the model is not locked
            if not self._models_lock[modeli]:
               param_list.extend(list(self._models[modeli][domi].parameters()))
      else:
         for modeli in range(self._nmodels):
            if not self._models_lock[modeli]:
               param_list.extend(list(self._models[modeli][domi].parameters()))
            if self._models_inuse[modeli]:
               self._models[modeli][domi].to(self._device)
               self._models[modeli][domi].train()
      optimizer = torch.optim.Adam(param_list, lr=lr_adam_init)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                   step_size=lr_adam_stepsize, 
                                                   gamma=lr_adam_gamma)
      loss = nn.MSELoss()
      reset_seed(seed)
      with tqdm(total=maxits_adam, disable=disable_tqdm_bar) as pbar:
         for i in range(maxits_adam):
            if i % data_update_freq == 0:
               tqdm.write("Updating dataset")
               x_bc, u_bc, x_far, x_near, y = self.sample_data(self._nsamples, 
                                                               self._gdomain, 
                                                               domi, 
                                                               self._eps, 
                                                               self._r_near,
                                                               fix_y = fix_y,
                                                               y_sample = y_sample)
               x_interior = torch.cat([x_near, x_far], dim=0).detach().requires_grad_(True)
            optimizer.zero_grad()
            loss_total, loss_bc, loss_pde, loss_sym = self.get_loss(loss, domi, x_interior, x_bc, u_bc, weights)
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            if i == 0 and enable_debug:
               self.plot_data(x_bc, x_interior, y, save_figs, fig_file_base, domi)
            if i % print_loss_freq == 0 or i == maxits_adam - 1:
               MSGPINNs.print_loss(i, loss_total.item(), loss_bc.item(), loss_pde.item(), loss_sym.item(), weights)
            pbar.update(1)
            loss_res_i[loss_its_i] = loss_total.item()
            loss_its_i += 1
      
      for i in range(self._nmodels):
         if not self._models_lock[i]:
            self._models[i][domi].save(self._models_state_files[i][domi])
      
      # next the LBFGS training loop
      optimizer = torch.optim.LBFGS(param_list, lr=lr_lbfgs_init if lr_lbfgs_init > 0 else scheduler.get_last_lr()[0])
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                   step_size=lr_lbfgs_stepsize, 
                                                   gamma=lr_lbfgs_gamma)
      loss = nn.MSELoss()
      reset_seed(seed)
      with tqdm(total=maxits_lbfgs, disable=disable_tqdm_bar) as pbar:
         for i in range(maxits_lbfgs):
            optimizer.zero_grad()
            if i % data_update_freq == 0:
               tqdm.write("Updating dataset")
               x_bc, u_bc, x_far, x_near, y = self.sample_data(self._nsamples, 
                                                               self._gdomain, 
                                                               domi, 
                                                               self._eps, 
                                                               self._r_near,
                                                               fix_y = fix_y,
                                                               y_sample = y_sample)
               x_interior = torch.cat([x_near, x_far], dim=0).detach().requires_grad_(True)
            optimizer.zero_grad()
            loss_items = np.zeros(3)
            def closure(its, weights, print_mgs):
               loss_total, loss_bc, loss_pde, loss_sym = self.get_loss(loss, domi, x_interior, x_bc, u_bc, weights)
               if print_mgs:
                  loss_items[0] = loss_bc.item()
                  loss_items[1] = loss_pde.item()
                  loss_items[2] = loss_sym.item()
               loss_total.backward()
               return loss_total
            if i % print_loss_freq == 0 or i == maxits_lbfgs - 1:
               loss_total = optimizer.step(lambda: closure(i, weights, True))
               MSGPINNs.print_loss(i, loss_total.item(), loss_items[0], loss_items[1], loss_items[2], weights)
            else:
               loss_total = optimizer.step(lambda: closure(i, weights, False))
            pbar.update(1)
            loss_res_i[loss_its_i] = loss_total.item()
            loss_its_i += 1

      MSGPINNs.plot_and_save_loss(loss_res_i, loss_its_i, domi, loss_history_file_base, save_figs, fig_file_base)
      
      for i in range(self._nmodels):
         if not self._models_lock[i]:
            self._models[i][domi].save(self._models_state_files[i][domi])
         if self._models_inuse[i]:
            if self._on_the_fly:
               # in the on-the-fly mode, we free the model from the meory as they are stored in the file
               del self._models[i][domi]
               gc.collect()
               if torch.cuda.is_available():
                  torch.cuda.empty_cache()
               print("Model %s freed" % self._models_names[i][domi])
               self._models[i].insert(domi, None)
            else:
               self._models[i][domi].to(torch.device('cpu'))
      
      if torch.cuda.is_available():
         torch.cuda.empty_cache()

      return loss_its_i

   def eval(self,
            x,
            domain_num = 0,
            all_inuse = False):
      '''
      Evaluate the solution at x.
      Inputs:
         x: Torch tensor of size n by dim.
            Input points.
         domain_num: Integer.
            Domain number.
      '''
      u = None
      for modeli in range(self._nmodels):
         if not all_inuse and not self._models_inuse[modeli]:
            continue
         if self._on_the_fly:
            self._models[modeli][domain_num] = GPINNs_Model(self._dim,
                                                            self._nunknown,
                                                            self._models_params[modeli][0],
                                                            self._models_params[modeli][1],
                                                            eps = self._eps,
                                                            alpha = self._models_alpha[modeli],
                                                            beta = self._models_beta[modeli],
                                                            use_diff = self._use_diff,
                                                            activation = self._models_activation[modeli],
                                                            dtype = self._dtype,
                                                            device = self._device,
                                                            init_method = self._models_init_method[modeli],
                                                            init_seed = self._init_seed[modeli],
                                                            print_info = False
                                                            )
            self._models[modeli][domain_num].load(self._models_state_files[modeli][domain_num], model_name=self._models_names[modeli][domain_num])
            self._models[modeli][domain_num].eval()
         else:
            self._models[modeli][domain_num].to(self._device)
            self._models[modeli][domain_num].eval()
         print(self._models[modeli][domain_num])
         print(u is None)
         if u is not None:
            u = u + self._models[modeli][domain_num](x)
         else:
            u = self._models[modeli][domain_num](x)
         if self._on_the_fly:
            del self._models[modeli][domain_num]
            gc.collect()
            if torch.cuda.is_available():
               torch.cuda.empty_cache()
            print("Model %s freed" % self._models_names[modeli][domain_num])
            self._models[modeli].insert(domain_num, None)
         else:
            self._models[modeli][domain_num].to(torch.device('cpu'))
      return u

   def show_results(self,
                    nplots = 1,
                    nsamples = (0, 1, 50, 50, 50),
                    seed = 906,
                    save_figs = False,
                    fig_file_base = None):
      '''
      Visualize the solution.
      '''
      self._gdomain.update_device(torch.device('cpu'))
      for domi in range(self._num_domains):
         self.show_results_domi(domi, nplots, nsamples, seed, save_figs, fig_file_base)

   def show_results_domi(self,
                    domi,
                    nplots = 1,
                    nsamples = (0, 1, 50, 50, 50),
                    seed = 906,
                    save_figs = False,
                    fig_file_base = None,
                    fix_y = None,
                    y_sample = None):
      '''
      Visualize the solution on one domain.
      '''
      if self._on_the_fly:
         for modeli in range(self._nmodels):
            if self._models_inuse[modeli]:
               self._models[modeli][domi] = GPINNs_Model(self._dim,
                                                         self._nunknown,
                                                         self._models_params[modeli][0],
                                                         self._models_params[modeli][1],
                                                         eps = self._eps,
                                                         alpha = self._models_alpha[modeli],
                                                         beta = self._models_beta[modeli],
                                                         use_diff = self._use_diff,
                                                         activation = self._models_activation[modeli],
                                                         dtype = self._dtype,
                                                         device = torch.device('cpu'),
                                                         init_method = self._models_init_method[modeli],
                                                         init_seed = self._init_seed[modeli],
                                                         print_info = False
                                                         )
               self._models[modeli][domi].load(self._models_state_files[modeli][domi], model_name=self._models_names[modeli][domi])
               self._models[modeli][domi].eval()
      else:
         for modeli in range(self._nmodels):
            if self._models_inuse[modeli]:
               self._models[modeli][domi].to(torch.device('cpu'))
               self._models[modeli][domi].eval()
      reset_seed(seed)
      for i in range(nplots):
         x_bc, u_bc, x_far, x_near, y = self.sample_data(self._nsamples, 
                                                         self._gdomain, 
                                                         domi, 
                                                         self._eps, 
                                                         self._r_near,
                                                         fix_y = fix_y,
                                                         y_sample = y_sample)
         x = torch.cat([x_bc, x_near, x_far], dim=0)
         u = self.eval(x, domi)
         if self._sol is not None:
            u_exact = self._sol(x.numpy())
            fig = plt.figure(figsize=(10,10))
            if self._dim == 1:
               ax = fig.add_subplot(221)
               ax.plot(x[...,0].detach().numpy(), np.zeros(x.shape[0]), marker='o', linestyle='None')
               ax.plot(y[0,0].detach().numpy(), np.zeros(1), marker='o', linestyle='None', color='red')
               ax.set_title('Sample points')
               ax = fig.add_subplot(222)
               ax.plot(x[...,0].detach().numpy(), u.detach().numpy(), marker='o', linestyle='None')
               ax.set_title('Predicted solution')
               ax = fig.add_subplot(223)
               ax.plot(x[...,0].detach().numpy(), u_exact, marker='o', linestyle='None')
               ax.set_title('Exact Solution')
               ax = fig.add_subplot(224)
               ax.plot(x[...,0].detach().numpy(), u_exact - u.detach().numpy().squeeze(), marker='o', linestyle='None')
               ax.set_title('Error')
            elif self._dim == 2:
               ax = fig.add_subplot(221)
               ax.scatter(x[...,0].detach().numpy(), x[...,1].detach().numpy())
               plt.scatter(y[0,0].detach().numpy(), y[0,1].detach().numpy(), color='red')
               ax.set_title('Sample points')
               ax = fig.add_subplot(222, projection='3d')
               ax.scatter(x[...,0].detach().numpy(), x[...,1].detach().numpy(), u.detach().numpy().squeeze())
               ax.set_title('Predicted solution')
               ax = fig.add_subplot(223, projection='3d')
               ax.scatter(x[...,0].detach().numpy(), x[...,1].detach().numpy(), u_exact)
               ax.set_title('Exact Solution')
               ax = fig.add_subplot(224, projection='3d')
               ax.scatter(x[...,0].detach().numpy(), x[...,1].detach().numpy(), u_exact - u.detach().numpy().squeeze())
               ax.set_title('Error')
            else:
               pass
            if save_figs:
               figname = MSGPINNs.fig_file_rule(fig_file_base, 'top_test', domi, i)
               plt.savefig(figname)
            else:
               plt.show()
            plt.close()
         else:
            fig = plt.figure(figsize=(10,5))
            if self._dim == 1:
               ax = fig.add_subplot(121)
               ax.plot(x[...,0].detach().numpy(), np.zeros(x.shape[0]), marker='o', linestyle='None')
               ax.plot(y[0,0].detach().numpy(), np.zeros(1), marker='o', linestyle='None', color='red')
               ax.set_title('Sample points')
               ax = fig.add_subplot(122)
               ax.plot(x[...,0].detach().numpy(), u.detach().numpy(), marker='o', linestyle='None')
               ax.set_title('Predicted solution')
            elif self._dim == 2:
               ax = fig.add_subplot(121)
               ax.scatter(x[...,0].detach().numpy(), x[...,1].detach().numpy())
               plt.scatter(y[0,0].detach().numpy(), y[0,1].detach().numpy(), color='red')
               ax.set_title('Sample points')
               ax = fig.add_subplot(122, projection='3d')
               ax.scatter(x[...,0].detach().numpy(), x[...,1].detach().numpy(), u.detach().numpy())
               ax.set_title('Predicted solution')
            else:
               pass
            if save_figs:
               figname = MSGPINNs.fig_file_rule(fig_file_base, 'top_test', domi, i)
               plt.savefig(figname)
            else:
               plt.show()
            plt.close()

      if self._on_the_fly:
         for modeli in range(self._nmodels):
            if self._models_inuse[modeli]:
               del self._models[modeli][domi]
               gc.collect()
               if torch.cuda.is_available():
                  torch.cuda.empty_cache()
               print("Model %s freed" % self._models_names[modeli][domi])
               self._models[modeli].insert(domi, None)

   @staticmethod
   def lock_check(models_lock, models_inuse, nmodels):
      if len(models_lock) != nmodels:
         raise ValueError("Length of lock_models must be the same as the number of models")
      
      if len(models_inuse) != nmodels:
         raise ValueError("Length of inuse_models must be the same as the number of models")
      
      if all(models_lock):
         print("All models are locked, no training")
         
      return
   
   @staticmethod
   def loss_file_rule(loss_history_file_base, domi):
      return loss_history_file_base + '_' + str(domi) + '.npz'
   
   @staticmethod
   def fig_file_rule(fig_file_base, figname, domi, num2 = None):
      if num2 is None:  
         return fig_file_base + '_' + figname + '_' + str(domi) + '.png'
      else:
         return fig_file_base + '_' + figname + '_' + str(domi) + '_' + str(num2) + '.png'

   @staticmethod
   def print_loss(its, loss_total_item, loss_bc_item, loss_pde_item, loss_sym_item, weights):
      tqdm.write('Iteration %d, Training Loss: %.2e' % (its, loss_total_item))
      tqdm.write("Loss terms: bc %.2e, pde %.2e, sym %.2e" % 
         (weights[0] * loss_bc_item, 
            weights[1] * loss_pde_item, 
            weights[2] * loss_sym_item))

   @staticmethod
   def load_loss(loss_history_file_base, ndomains, maxits_adam, maxits_lbfgs):
      loss_res = []
      loss_its = np.zeros(ndomains, dtype=np.int64)
      if loss_history_file_base is not None:
         for domi in range(ndomains):
            loss_history_file_i = MSGPINNs.loss_file_rule(loss_history_file_base, domi)
            if not os.path.exists(loss_history_file_i):
               loss_its[domi] = 0
               loss_res.append(np.zeros(maxits_adam + maxits_lbfgs).astype(np.float64))
            else:
               loss_data_i = np.load(loss_history_file_i)
               loss_its[domi] = loss_data_i['total_its']
               loss_res_i = loss_data_i['loss_history']
               loss_res_i = np.concatenate((loss_res_i, np.zeros(maxits_adam + maxits_lbfgs).astype(np.float64)))
               loss_res.append(loss_res_i)
      else:
         for domi in range(ndomains):
            loss_its[domi] = 0
            loss_res.append(np.zeros(maxits_adam + maxits_lbfgs).astype(np.float64))
      return loss_res, loss_its

   @staticmethod
   def plot_and_save_loss(loss_res_i, loss_its_i, domi, loss_history_file_base, save_figs, fig_file_base):
      plt.figure(figsize=(10,5))
      plt.yscale('log')
      plt.plot(loss_res_i)
      plt.title('Loss history')
      plt.xlabel('Iteration')
      plt.ylabel('Loss')
      if save_figs:
         plt.savefig(MSGPINNs.fig_file_rule(fig_file_base, 'loss', domi))
      else:
         plt.show()
      plt.close()

      if loss_history_file_base is not None:
         loss_history_file_i = MSGPINNs.loss_file_rule(loss_history_file_base, domi)
         np.savez(loss_history_file_i, 
                  loss_history = loss_res_i, 
                  total_its = loss_its_i)
         
   @staticmethod
   def sample_data(samples, 
                   gdomain, 
                   domi, 
                   eps, 
                   r_near,
                   fix_y = None,
                   y_sample = None):
      if fix_y is not None:
         y = fix_y.detach().requires_grad_(True)
      elif y_sample is not None:
         y = y_sample(samples[0] + samples[1]).detach().requires_grad_(True)
      else:
         if samples[0] > 0:
            y_onquad = gdomain.sample_uniform(samples[0], on_quad=True, domain_num=domi)
         else:
            y_onquad = None
         if samples[1] > 0:
            y_random = gdomain.sample_uniform(samples[1], on_quad=False, domain_num=domi)
         else:
            y_random = None
         if y_onquad is not None:
            if y_random is not None:
               y = torch.cat([y_onquad, y_random], dim=0).detach().requires_grad_(True)
            else:
               y = y_onquad.detach().requires_grad_(True)
         else:
            y = y_random.detach().requires_grad_(True)
      x_bc, u_bc = gdomain.sample_uniform(samples[2], boundary=True, target_y = y)
      x_far = gdomain.sample_uniform(samples[3], target_y = y)
      x_near = gdomain.sample_uniform(samples[4], rrange = (0.0, eps * r_near), target_y = y)
      return x_bc, u_bc, x_far, x_near, y