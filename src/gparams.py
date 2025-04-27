import re
import os
import sys
import glob
from IPython import get_ipython
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_parameters(file_path):
   parameters = {}
   with open(file_path, 'r') as file:
      for line in file:
         stripped_line = line.strip()
         if not stripped_line or stripped_line.startswith('#'):
            continue
         parts = re.split(r'[ ,]+', stripped_line)
         if len(parts) < 2:
            continue
         key, type_indicator, *values = parts
         try:
            if type_indicator == 'I':
               if len(parts) == 2:
                  value = None
               elif len(parts) == 3:
                  value = int(values[0])
               else:
                  value = [int(v) for v in values]
            elif type_indicator == 'F':
               if len(parts) == 2:
                  value = None
               elif len(parts) == 3:
                  value = float(values[0])
               else:
                  value = [float(v) for v in values]
            elif type_indicator == 'S':
               if len(parts) == 2:
                  value = None
               elif len(parts) == 3:
                  value = str(values[0])
               else:
                  value = [str(v) for v in values]
            elif type_indicator == 'B':
               if len(parts) == 2:
                  value = None
               elif len(parts) == 3:
                  value = bool(int(values[0]))
               else:
                  value = [bool(int(v)) for v in values]
            else:
               print(f"Warning: Unknown type indicator '{type_indicator}' for key '{key}'")
               continue
         except ValueError as e:
            print(f"Warning: Error converting value '{value}' for key '{key}': {e}")
            continue
         parameters[key] = value
   return parameters

def validation_init_domain(params):
   print("Using domain seed: ", params['domain_seed'])
   np.random.seed(params['domain_seed'])
   torch.manual_seed(params['domain_seed'])
   torch.cuda.manual_seed(params['domain_seed'])
   # first check if we need domain for y
   if 'domain_y_type' in params:
      if params['domain_y_type'] == 'segment':
         if 'domain_y_params' in params:
            if len(params['domain_y_params']) == 2:
               print("Creating specific segment y domain: (", params['domain_y_params'][0], params['domain_y_params'][1], ")")
               import geometry
               params['domain_y'] = geometry.GSegment( (params['domain_y_params'][0],params['domain_y_params'][1]),
                                                         dtype = params['dtype'],
                                                         device = params['device'])
            else:
               params['domain_y'] = None
         else:
            params['domain_y'] = None
         if 'domaim_y_seed' in params:
            np.random.seed(params['domain_y_seed'])
         else:
            params['domain_y_seed'] = params['domain_seed']
      elif params['domain_y_type'] == 'triangle':
         if 'domain_y_params' in params:
            if len(params['domain_y_params']) == 6:
               print("Creating specific triangle y domain: (", params['domain_y_params'][0], params['domain_y_params'][1], params['domain_y_params'][2], params['domain_y_params'][3], params['domain_y_params'][4], params['domain_y_params'][5], ")")
               import geometry
               params['domain_y'] = geometry.GTriangle( torch.tensor([[params['domain_y_params'][0], params['domain_y_params'][1]], 
                                                                     [params['domain_y_params'][2], params['domain_y_params'][3]],
                                                                     [params['domain_y_params'][4], params['domain_y_params'][5]]],
                                                                     dtype = params['dtype'],
                                                                     device = params['device']),
                                                         dtype = params['dtype'],
                                                         device = params['device'])
            else:
               params['domain_y'] = None
            if 'domain_y_seed' in params:
               np.random.seed(params['domain_y_seed'])
            else:
               params['domain_y_seed'] = params['domain_seed']
         else:
            params['domain_y'] = None
      else:
         params['domain_y'] = None
   else:
      params['domain_y'] = None
   # now create the gdomain
   if params['domain_y'] is not None:
      # in this case we do not create subdomains
      params['domain_dd_params'][1] = 0
   if params['domain_type'] == 'segment':
      print("Using domain type: 1D segment")
      params['dim'] = 1
      import geometry
      params['domain'] = geometry.GDSegment((params['domain_params'][0],params['domain_params'][1]), 
                                            ncell_dd = int(params['domain_dd_params'][0]), 
                                            dlev = int(params['domain_dd_params'][1]), 
                                            ncell_quad = int(params['domain_quad_params'][0]),
                                            degree = int(params['domain_quad_params'][1]),
                                            dtype = params['dtype'], 
                                            device = params['device'])
      print("Generating plots")
      plt.figure(figsize=(10,3))
      for i in range(params['domain']._ndomains):
         plt.plot(params['domain']._mesh_points[torch.cat((params['domain']._mesh_tris[params['domain']._domains[i],0],
                                             params['domain']._mesh_tris[params['domain']._domains[i],1])
                                             )
                                             ].cpu().detach().numpy(), np.zeros((2*params['domain']._domains[i].shape[0],1)), marker='o', linestyle='None')
      plt.title("Domain")
      plt.tight_layout()
      if params['save_figs']:
         figname = params['fig_file_base'] + '_dd.png'
         plt.savefig(figname)
      else:
         plt.show()
      plt.close()
      plt.figure(figsize=(10,3))
      plt.plot(params['domain']._quad_mesh_points.cpu().detach().numpy(), np.zeros((params['domain']._quad_mesh_points.shape[0],1)), marker='o', markersize=1, linestyle='None')
      plt.plot(params['domain']._quad_points.cpu().detach().numpy(), np.zeros((params['domain']._quad_points.shape[0],1)), marker='o', markersize=1, linestyle='None')
      plt.title("Quad Points")
      plt.tight_layout()
      if params['save_figs']:
         figname = params['fig_file_base'] + '_quad.png'
         plt.savefig(figname)
      else:
         plt.show()
      plt.close()
   elif params['domain_type'] == 'polygon':
      print("Using domain type: 2D polygon")
      params['dim'] = 2
      import geometry
      print("Running DD")
      if 'domain_params_hole' in params and 'domain_hole' in params:
         params['domain'] = geometry.GDPolygon(np.reshape(params['domain_params'], (int(len(params['domain_params'])/2), 2)),
                                    vertices_hole = np.reshape(params['domain_params_hole'], (int(len(params['domain_params_hole'])/2), 2)),
                                    hole=np.reshape(params['domain_hole'], (int(len(params['domain_hole'])/2), 2)),
                                    min_area_dd = params['domain_dd_params'][0],
                                    dlev = int(params['domain_dd_params'][1]),
                                    min_area_quad = params['domain_quad_params'][0],
                                    degree = int(params['domain_quad_params'][1]),
                                    dtype = params['dtype'],
                                    device = params['device'])
      else:
         params['domain'] = geometry.GDPolygon(np.reshape(params['domain_params'], (int(len(params['domain_params'])/2), 2)),
                                       min_area_dd = params['domain_dd_params'][0], 
                                       dlev = int(params['domain_dd_params'][1]), 
                                       min_area_quad = params['domain_quad_params'][0],
                                       degree = int(params['domain_quad_params'][1]), 
                                       dtype = params['dtype'], 
                                       device = params['device'])
      print("Creating mesh")
      params['tri'] = params['domain'].get_plot_tri()
      try:
         import meshio
         meshio.write(params['mesh_file_base'] + ".xdmf", 
                      meshio.Mesh(params['domain']._mesh_points.detach().cpu().numpy(), 
                                 [("triangle", params['domain']._mesh_tris.detach().cpu().numpy())]))
      except ImportError:
         pass
      print("Generating plots")
      fig = plt.figure(figsize=(10, 5))
      ax = fig.add_subplot(121)
      for i in range(params['domain']._ndomains):
         ax.triplot(params['domain']._mesh_points[:, 0].cpu().detach().numpy(), 
                     params['domain']._mesh_points[:, 1].cpu().detach().numpy(), 
                     params['domain']._mesh_tris[params['domain']._domains[i],:].cpu().detach().numpy())
      ax.set_title("Domain")
      ax = fig.add_subplot(122)
      ax.triplot(params['domain']._quad_mesh_points[:, 0].cpu().detach().numpy(), 
                     params['domain']._quad_mesh_points[:, 1].cpu().detach().numpy(), 
                     params['domain']._quad_mesh_tris.cpu().detach().numpy(), color = 'red')
      ax.scatter(params['domain']._quad_points[:,0].cpu().detach().numpy(),
                 params['domain']._quad_points[:,1].cpu().detach().numpy(), color = 'blue',s=1)
      ax.set_title("Quad Points")
      plt.tight_layout()
      if params['save_figs']:
         figname = params['fig_file_base'] + '_dd_quad.png'
         plt.savefig(figname)
      else:
         plt.show()
      plt.close()
   elif params['domain_type'] == 'circle':
      print("Using domain type: 2D circle")
      params['dim'] = 2
      import geometry
      print("Running DD")
      params['domain'] = geometry.GDCircle((params['domain_params'][0], params['domain_params'][1]),
                                             params['domain_params'][2],
                                             min_area_dd = params['domain_dd_params'][0], 
                                             dlev = int(params['domain_dd_params'][1]), 
                                             nbd = int(params['domain_dd_params'][2]), 
                                             min_area_quad = params['domain_quad_params'][0],
                                             degree = int(params['domain_quad_params'][1]), 
                                             dtype = params['dtype'], 
                                             device = params['device'])
      print("Creating mesh")
      params['tri'] = params['domain'].get_plot_tri()
      try:
         import meshio
         meshio.write(params['mesh_file_base'] + ".xdmf", 
                      meshio.Mesh(params['domain']._mesh_points.detach().cpu().numpy(), 
                                 [("triangle", params['domain']._mesh_tris.detach().cpu().numpy())]))
      except ImportError:
         pass
      print("Generating plots")
      fig = plt.figure(figsize=(10, 5))
      ax = fig.add_subplot(121)
      for i in range(params['domain']._ndomains):
         ax.triplot(params['domain']._mesh_points[:, 0].cpu().detach().numpy(), 
                     params['domain']._mesh_points[:, 1].cpu().detach().numpy(), 
                     params['domain']._mesh_tris[params['domain']._domains[i],:].cpu().detach().numpy())
      ax.set_title("Domain")
      ax = fig.add_subplot(122)
      ax.triplot(params['domain']._quad_mesh_points[:, 0].cpu().detach().numpy(), 
                     params['domain']._quad_mesh_points[:, 1].cpu().detach().numpy(), 
                     params['domain']._quad_mesh_tris.cpu().detach().numpy(), color = 'red')
      ax.scatter(params['domain']._quad_points[:,0].cpu().detach().numpy(),
                 params['domain']._quad_points[:,1].cpu().detach().numpy(), color = 'blue',s=1)
      ax.set_title("Quad Points")
      plt.tight_layout()
      if params['save_figs']:
         figname = params['fig_file_base'] + '_dd_quad.png'
         plt.savefig(figname)
      else:
         plt.show()
      plt.close()
   
   else:
      raise ValueError("Invalid domain type")
   return params

def validation_init_pde(params):
   if params['pde_type'] == 'lap':
      if params['dim'] == 1:
         print("Using 1D Laplacian")
         import pdes
         params['pde'] = pdes.pde1d_1()
      elif params['dim'] == 2:
         print("Using 2D Laplacian")
         import pdes
         params['pde'] = pdes.pde2d_1(is_unit_circle=params['is_unit_circle'])
      else:
         raise ValueError("Invalid dimension for Laplacian PDE")
   elif params['pde_type'] == 'slap':
      if params['dim'] == 1:
         import pdes
         if 'c_value' in params:
            c_value = params['c_value']
         else:
            c_value = 0.0
         print("Using 1D Shifted Laplacian with shift = ", c_value)
         params['pde'] = pdes.pde1d_2(c_value = c_value)
      elif params['dim'] == 2:
         import pdes
         if 'c_value' in params:
            c_value = params['c_value']
         else:
            c_value = 0.0
         print("Using 2D Shifted Laplacian with shift = ", c_value)
         params['pde'] = pdes.pde2d_2(c_value = c_value, is_unit_circle=params['is_unit_circle'])
      else:
         raise ValueError("Invalid dimension for Stiff Laplacian PDE")
   elif params['pde_type'] == 'vlap':
      if params['dim'] == 1:
         import pdes
         if 'c_value' in params:
            c_value = params['c_value']
         else:
            c_value = 0.0
         print("Using 1D Laplacian with shift c * (1 + x^2) with c = ", c_value)
         params['pde'] = pdes.pde1d_3(c_value = c_value)
      elif params['dim'] == 2:
         import pdes
         if 'c_value' in params:
            c_value = params['c_value']
         else:
            c_value = 0.0
         print("Using 2D Laplacian with shift c * (1 + x^2 + y^2) with c = ", c_value)
         params['pde'] = pdes.pde2d_3(c_value = c_value, is_unit_circle=params['is_unit_circle'])
      else:
         raise ValueError("Invalid dimension for Variable Laplacian PDE")
   else:
      raise ValueError("Invalid PDE type")
   return params

def validation_init_model(params):
   import pinns
   if 'nmodels' not in params:
      params['nmodels'] = int(len(params['models_params'])/2)
   models_params_list = []
   for i in range(params['nmodels']):
      models_params_list.append(params['models_params'][2*i:2*i+2])
   if 'on_the_fly' not in params:
      params['on_the_fly'] = False
   if 'activation' in params:
      try:  
         if len(params['activation']) > 1:
            activation = params['activation']
         else:
            activation = [params['activation']] * params['nmodels']
      except:
         activation = [params['activation']] * params['nmodels']
   else:
      activation = ['Default'] * params['nmodels']
   if 'model_init_method' in params:
      try:  
         if len(params['model_init_method']) > 1:
            model_init_method = params['model_init_method']
         else:
            model_init_method = [params['model_init_method']] * params['nmodels']
      except:
         model_init_method = [params['model_init_method']] * params['nmodels']
   else:
      model_init_method = ['xavier'] * params['nmodels']
   if 'print_info' in params:
      try:
         if len(params['print_info']) > 1:
            print_info = params['print_info']
         else:
            print_info = [params['print_info']] * params['nmodels']
      except:
         print_info = [params['print_info']] * params['nmodels']      
   else:
      print_info = [True] * params['nmodels']
   model_state_file_base_list = []
   if params['nmodels'] == 1:
      model_state_file_base_list.append(params['model_file_base'] + '_' + params['models_name'])
   else:
      for i in range(params['nmodels']):
         model_state_file_base_list.append(params['model_file_base'] + '_' + params['models_name'][i])
   if params['nmodels'] == 1:
      init_seed = [params['models_seed']]
      alpha = [params['models_alpha']]
      beta = [params['models_beta']]
      model_name = [params['models_name']]
   else:
      init_seed = params['models_seed']
      alpha = params['models_alpha']
      beta = params['models_beta']
      model_name = params['models_name']
   params['model'] = pinns.MSGPINNs(
                                 pde = params['pde'].pde,
                                 nunknown = params['pde'].nunknown(),
                                 gdomain = params['domain'],
                                 eps = params['eps'],
                                 r_near = params['r_near'],
                                 nsamples = params['nsamples'],
                                 use_diff = params['use_diff'],
                                 model_params = models_params_list,
                                 model_activation = activation,
                                 model_init_method = model_init_method,
                                 model_print_info = print_info,
                                 model_name_base = model_name,
                                 model_state_file_base = model_state_file_base_list,
                                 sol = params['pde'].sol,
                                 bc = None,
                                 init_seed = init_seed,
                                 alpha = alpha,
                                 beta = beta,
                                 dtype=params['dtype'],
                                 device=params['device'],
                                 on_the_fly=params['on_the_fly'],)
   return params

def validation_train(params):
   
   ntrainings = int(len(params['training_weights'])/3)
   models_lock = []
   models_inuse = []
   models_weights = []
   for i in range(ntrainings):
      models_weights.append(params['training_weights'][i*3: (i+1)*3])
      models_weights[-1][0] /= (params['eps']**params['dim'])
      models_weights[-1][2] /= (params['eps']**params['dim'])
      if ntrainings == 1 and params['nmodels'] == 1:
         models_lock.append([params['training_lock']])
         models_inuse.append([params['training_inuse']])
      else:
         models_lock.append(params['training_lock'][i*params['nmodels']: (i+1)*params['nmodels']])
         models_inuse.append(params['training_inuse'][i*params['nmodels']: (i+1)*params['nmodels']])
   if ntrainings == 1:
      training_seed = [params['training_seed']]
      maxits_adam = [params['maxits_adam']]
      maxits_lbfgs = [params['maxits_lbfgs']]
      data_update_freq = [params['data_update_freq']]
      plot_loss_freq = [params['plot_loss_freq']]
      adam_lr = [params['adam_lr']]
      adam_lr_stepsize = [params['adam_lr_stepsize']]
      adam_lr_gamma = [params['adam_lr_gamma']]
      lbfgs_lr = [params['lbfgs_lr']]
      lbfgs_lr_stepsize = [params['lbfgs_lr_stepsize']]
      lbfgs_lr_gamma = [params['lbfgs_lr_gamma']]
   else:
      training_seed = params['training_seed']
      maxits_adam = params['maxits_adam']
      maxits_lbfgs = params['maxits_lbfgs']
      data_update_freq = params['data_update_freq']
      plot_loss_freq = params['plot_loss_freq']
      adam_lr = params['adam_lr']
      adam_lr_stepsize = params['adam_lr_stepsize']
      adam_lr_gamma = params['adam_lr_gamma']
      lbfgs_lr = params['lbfgs_lr']
      lbfgs_lr_stepsize = params['lbfgs_lr_stepsize']
      lbfgs_lr_gamma = params['lbfgs_lr_gamma']
   loss_history_file_base_list = []
   fig_file_base_list = []
   if ntrainings == 1:
      training_name = [params['training_name']]
   else:
      training_name = params['training_name']
   for i in range(ntrainings):
      #loss_history_file_base_list.append(params['training_file_base'] + '_' + training_name[i])
      loss_history_file_base_list.append(params['training_file_base']) # overwrite, append to the same loss
      fig_file_base_list.append(params['fig_file_base'] + '_' + training_name[i])
   for i in range(ntrainings):
      print("Training number: %d with training weights: %f %f %f" % (i, models_weights[i][0], models_weights[i][1], models_weights[i][2]))
      print("Lock status: %s" % models_lock[i])
      print("Inuse status: %s" % models_inuse[i])
      if params['domain_y'] is None:
         # apply the entire traiming
         params['model'].train(seed = training_seed[i],
                              models_lock = models_lock[i],
                              models_inuse = models_inuse[i],
                              weights = models_weights[i],
                              maxits_adam = maxits_adam[i],
                              maxits_lbfgs = maxits_lbfgs[i],
                              data_update_freq = data_update_freq[i],
                              print_loss_freq = plot_loss_freq[i],
                              lr_adam_init = adam_lr[i],
                              lr_adam_stepsize = adam_lr_stepsize[i],
                              lr_adam_gamma = adam_lr_gamma[i],
                              lr_lbfgs_init = lbfgs_lr[i],
                              lr_lbfgs_stepsize = lbfgs_lr_stepsize[i],
                              lr_lbfgs_gamma = lbfgs_lr_gamma[i],
                              loss_history_file_base = loss_history_file_base_list[i],
                              save_figs = params['save_figs'],
                              fig_file_base = fig_file_base_list[i],
                              disable_tqdm_bar = params['hide_tqdm_bar'],
                              enable_debug = True)
      else:
         # Only train on a specific domain
         params['model'].train_domi(0,
                                    seed = training_seed[i],
                                    loss_res_i = None,
                                    loss_its_i = None,
                                    models_lock = models_lock[i],
                                    models_inuse = models_inuse[i],
                                    weights = models_weights[i],
                                    maxits_adam = maxits_adam[i],
                                    maxits_lbfgs = maxits_lbfgs[i],
                                    data_update_freq = data_update_freq[i],
                                    print_loss_freq = plot_loss_freq[i],
                                    lr_adam_init = adam_lr[i],
                                    lr_adam_stepsize = adam_lr_stepsize[i],
                                    lr_adam_gamma = adam_lr_gamma[i],
                                    lr_lbfgs_init = lbfgs_lr[i],
                                    lr_lbfgs_stepsize = lbfgs_lr_stepsize[i],
                                    lr_lbfgs_gamma = lbfgs_lr_gamma[i],
                                    loss_history_file_base = loss_history_file_base_list[i],
                                    save_figs = params['save_figs'],
                                    fig_file_base = fig_file_base_list[i],
                                    disable_tqdm_bar = params['hide_tqdm_bar'],
                                    enable_debug = True,
                                    y_sample = params['domain_y'].sample_uniform)
   return params

def plot_model_params(params):
   '''
   Plot histogram of the model parameters.
   '''
   all_params = None
   for i in range(params['model']._nmodels):
      for j in range(params['model']._ndomains):
         if all_params is not None:
            all_params = torch.cat([all_params, torch.cat([p.flatten() for p in params['model']._models[i][j].parameters()])])
         else:
            all_params = torch.cat([torch.cat([p.flatten() for p in params['model']._models[i][j].parameters()])])
   all_params = all_params.detach().cpu().numpy()
   fig, axs = plt.subplots(1, 1, figsize=(10, 5), facecolor='w', edgecolor='k')
   axs.hist(all_params, bins=20)
   axs.set_xlabel('Parameter value')
   axs.set_ylabel('Frequency')
   axs.set_title('Histogram of model parameters')
   if params['save_figs']:
      figname = params['fig_file_base'] + '_model_params.png'
      plt.savefig(figname)
   else:
      plt.show()
   plt.close()
   return params

def plot_2dgreen_single_source(params, err_max = 0.0):
   if params['dim'] == 2 and params['domain_y'] is not None and params['pde']._is_unit_circle:
      ntotal = params['domain']._quad_mesh_points.shape[0]
      y = params['domain_y'].sample_uniform(ntotal)
      params['G_exact_plot'] = params['pde'].sol(np.concatenate((params['domain']._quad_mesh_points.detach().cpu().numpy(), 
                                                                y.detach().cpu().numpy()), axis=1))
      params['G_pred_plot'] = params['model'].eval(torch.cat((params['domain']._quad_mesh_points, 
                                                              y.to(params['domain']._quad_mesh_points.device)), dim=1)).detach().cpu().numpy()[:,0]
      if params['G_exact_plot'] is None:
         fig = plt.figure(figsize=(5,5))
         ax = fig.add_subplot(111)
         im = ax.tricontourf(params['tri'], params['G_pred_plot'], cmap='jet')
         ax.set_title('Prediction')
         fig.colorbar(im, ax=ax)
         ax.set_aspect('equal', adjustable='box')
      else:
         G_error_plot = np.abs(params['G_exact_plot'] - params['G_pred_plot'])
         params['G_min_plot'] = min(min(np.min(params['G_exact_plot']), np.min(params['G_pred_plot'])), np.min(G_error_plot))
         params['G_max_plot'] = max(max(np.max(params['G_exact_plot']), np.max(params['G_pred_plot'])), np.max(G_error_plot))
         fig = plt.figure(figsize=(15,5))
         ax = fig.add_subplot(131)
         im = ax.tricontourf(params['tri'], params['G_exact_plot'], cmap='jet', vmin=params['G_min_plot'], vmax=params['G_max_plot'])
         ax.set_title('Exact')
         fig.colorbar(im, ax=ax)
         ax.set_aspect('equal', adjustable='box')
         ax = fig.add_subplot(132)
         im = ax.tricontourf(params['tri'], params['G_pred_plot'], cmap='jet', vmin=params['G_min_plot'], vmax=params['G_max_plot'])
         ax.set_title('Prediction')
         fig.colorbar(im, ax=ax)
         ax.set_aspect('equal', adjustable='box')
         ax = fig.add_subplot(133)
         im = ax.tricontourf(params['tri'], np.abs(params['G_exact_plot'] - params['G_pred_plot']), cmap='jet')
         ax.set_title('Error')
         fig.colorbar(im, ax=ax)
         ax.set_aspect('equal', adjustable='box')
      if params['save_figs']:
         figname = params['fig_file_base'] + '_green' + '.png'
         plt.savefig(figname)
         npzname = params['fig_file_base'] + '_green.npz'
         np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(), 
                                 G_exact=params['G_exact_plot'], 
                                 G_pred=params['G_pred_plot'])
         npzname = params['fig_file_base'] + '_tri.npz'
         np.savez(npzname, x = params['domain']._quad_mesh_points[...,0].detach().cpu().numpy(), 
                              y = params['domain']._quad_mesh_points[...,1].detach().cpu().numpy(), 
                              tri = params['domain']._quad_mesh_tris.detach().cpu().numpy())
      else:
         plt.show()
      plt.close()
      if params['G_exact_plot'] is not None:
         params['G_l1_error'] = np.linalg.norm(params['G_exact_plot'] - params['G_pred_plot'], ord=1)
         params['G_l2_error'] = np.linalg.norm(params['G_exact_plot'] - params['G_pred_plot'], ord=2)
         params['G_linf_error'] = np.max(np.abs(params['G_exact_plot'] - params['G_pred_plot']))
   else:
      raise ValueError("Invalid option for plotting 2D Green function")
   return params

def plot_2dgreen_single_source_fem(params, err_max = 0.0, refine_level = 1, fem_only = False):
   if fem_only:
      if params['dim'] == 2 and params['domain_y'] is not None:
         ntotal = params['domain']._quad_mesh_points.shape[0]
         y = params['domain_y'].sample_uniform(ntotal)
         
         rhs_fenics = (
            f"1.0 / (pow({params['eps']}, 2) * pi) * "
            f"exp(- (pow(x[0] - {params['domain_y']._vertices[0,0]}, 2) + "
            f"pow(x[1] - {params['domain_y']._vertices[0,1]}, 2)) "
            f"/ pow({params['eps']}, 2))"
         )
         try: 
            import fenics
            mesh = fenics.Mesh()
            with fenics.XDMFFile(params['mesh_file_base'] + ".xdmf") as infile:
               infile.read(mesh)
            for _ in range(refine_level): mesh = fenics.refine(mesh)
            V = fenics.FunctionSpace(mesh, 'P', 1)
            u_D = fenics.Constant(0)
            bc = fenics.DirichletBC(V, u_D, 'on_boundary')
            u = fenics.TrialFunction(V)
            v = fenics.TestFunction(V)
            f = fenics.Expression(rhs_fenics, degree=5, domain=mesh)
            L = params['pde'].weak_form(V,V)
            R = f * v * fenics.dx
            sol_func = fenics.Function(V)
            fenics.solve(L == R, sol_func, bc)
            params['G_fem_plot'] = np.array([sol_func(fenics.Point(*p)) for p in params['domain']._quad_mesh_points.detach().cpu().numpy()])
         except ImportError:
            print("FEniCS not installed, skipped")
            return params

         params['G_min_plot_fem'] = np.min(params['G_fem_plot'])
         params['G_max_plot_fem'] = np.max(params['G_fem_plot'])
         fig = plt.figure(figsize=(5,5))
         ax = fig.add_subplot(111)
         im = ax.tricontourf(params['tri'], params['G_fem_plot'], cmap='jet', vmin=params['G_min_plot_fem'], vmax=params['G_max_plot_fem'])
         ax.set_title('FEM')
         fig.colorbar(im, ax=ax)
         ax.set_aspect('equal', adjustable='box')
         if params['save_figs']:
            figname = params['fig_file_base'] + '_green_fem' + '.png'
            plt.savefig(figname)
            npzname = params['fig_file_base'] + '_green_fem.npz'
            np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(), 
                                    G_fem=params['G_fem_plot'])
            npzname = params['fig_file_base'] + '_tri.npz'
            np.savez(npzname, x = params['domain']._quad_mesh_points[...,0].detach().cpu().numpy(), 
                                 y = params['domain']._quad_mesh_points[...,1].detach().cpu().numpy(), 
                                 tri = params['domain']._quad_mesh_tris.detach().cpu().numpy())
         else:
            plt.show()
         plt.close()
      else:
         raise ValueError("Invalid option for plotting 2D Green function")
   else:
      if params['dim'] == 2 and params['domain_y'] is not None:
         ntotal = params['domain']._quad_mesh_points.shape[0]
         y = params['domain_y'].sample_uniform(ntotal)
         params['G_pred_plot'] = params['model'].eval(torch.cat((params['domain']._quad_mesh_points, 
                                                               y.to(params['domain']._quad_mesh_points.device)), dim=1)).detach().cpu().numpy()[:,0]
         
         rhs_fenics = (
            f"1.0 / (pow({params['eps']}, 2) * pi) * "
            f"exp(- (pow(x[0] - {params['domain_y']._vertices[0,0]}, 2) + "
            f"pow(x[1] - {params['domain_y']._vertices[0,1]}, 2)) "
            f"/ pow({params['eps']}, 2))"
         )
         try: 
            import fenics
            mesh = fenics.Mesh()
            with fenics.XDMFFile(params['mesh_file_base'] + ".xdmf") as infile:
               infile.read(mesh)
            for _ in range(refine_level): mesh = fenics.refine(mesh)
            V = fenics.FunctionSpace(mesh, 'P', 1)
            u_D = fenics.Constant(0)
            bc = fenics.DirichletBC(V, u_D, 'on_boundary')
            u = fenics.TrialFunction(V)
            v = fenics.TestFunction(V)
            f = fenics.Expression(rhs_fenics, degree=5, domain=mesh)
            L = params['pde'].weak_form(V,V)
            R = f * v * fenics.dx
            sol_func = fenics.Function(V)
            fenics.solve(L == R, sol_func, bc)
            params['G_fem_plot'] = np.array([sol_func(fenics.Point(*p)) for p in params['domain']._quad_mesh_points.detach().cpu().numpy()])
         except ImportError:
            print("FEniCS not installed, skipped")
            return params

         G_error_plot = np.abs(params['G_fem_plot'] - params['G_pred_plot'])
         params['G_min_plot_fem'] = min(min(np.min(params['G_fem_plot']), np.min(params['G_pred_plot'])), np.min(G_error_plot))
         params['G_max_plot_fem'] = max(max(np.max(params['G_fem_plot']), np.max(params['G_pred_plot'])), np.max(G_error_plot))
         fig = plt.figure(figsize=(15,5))
         ax = fig.add_subplot(131)
         im = ax.tricontourf(params['tri'], params['G_fem_plot'], cmap='jet', vmin=params['G_min_plot_fem'], vmax=params['G_max_plot_fem'])
         ax.set_title('FEM')
         fig.colorbar(im, ax=ax)
         ax.set_aspect('equal', adjustable='box')
         ax = fig.add_subplot(132)
         im = ax.tricontourf(params['tri'], params['G_pred_plot'], cmap='jet', vmin=params['G_min_plot_fem'], vmax=params['G_max_plot_fem'])
         ax.set_title('Prediction')
         fig.colorbar(im, ax=ax)
         ax.set_aspect('equal', adjustable='box')
         ax = fig.add_subplot(133)
         im = ax.tricontourf(params['tri'], G_error_plot, cmap='jet')
         ax.set_title('Error')
         fig.colorbar(im, ax=ax)
         ax.set_aspect('equal', adjustable='box')
         if params['save_figs']:
            figname = params['fig_file_base'] + '_green_fem' + '.png'
            plt.savefig(figname)
            npzname = params['fig_file_base'] + '_green_fem.npz'
            np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(), 
                                    G_fem=params['G_fem_plot'], 
                                    G_pred=params['G_pred_plot'])
            npzname = params['fig_file_base'] + '_tri.npz'
            np.savez(npzname, x = params['domain']._quad_mesh_points[...,0].detach().cpu().numpy(), 
                                 y = params['domain']._quad_mesh_points[...,1].detach().cpu().numpy(), 
                                 tri = params['domain']._quad_mesh_tris.detach().cpu().numpy())
         else:
            plt.show()
         plt.close()
         params['G_l1_error_fem'] = np.linalg.norm(params['G_fem_plot'] - params['G_pred_plot'], ord=1)
         params['G_l2_error_fem'] = np.linalg.norm(params['G_fem_plot'] - params['G_pred_plot'], ord=2)
         params['G_linf_error_fem'] = np.max(np.abs(params['G_fem_plot'] - params['G_pred_plot']))
      else:
         raise ValueError("Invalid option for plotting 2D Green function")
   return params

def plot_1dgreen(params, err_max = 0.0):
   if params['dim'] == 1:
      if params['domain_y'] is not None and params['domain_y']._range[0] == params['domain_y']._range[1]:
         ntotal = params['domain']._quad_mesh_points.shape[0]
         y = params['domain_y'].sample_uniform(1)[0].repeat(ntotal)
         x = torch.vstack([params['domain']._quad_mesh_points, y]).T
         params['G_exact_plot'] = params['pde'].sol(x.detach().cpu().numpy())
         if params['G_exact_plot'] is None:
            params['G_pred_plot'] = np.zeros((1,ntotal)).astype(params['dtype_np'])
         else:
            params['G_exact_plot'] = params['G_exact_plot'].reshape((1,ntotal), order='F')
            params['G_pred_plot'] = np.zeros_like(params['G_exact_plot'])
         params['G_pred_plot'] = params['model'].eval(x).detach().cpu().numpy().squeeze().reshape((1, ntotal), order='F')
         if torch.cuda.is_available():
            torch.cuda.empty_cache()
         if params['G_exact_plot'] is not None:
            fig, axs = plt.subplots(1, 1, figsize=(10, 5), facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace = .2, wspace=.2)
            axs.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(), params['G_exact_plot'][0,:], label='Exact')
            axs.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(), params['G_pred_plot'][0,:], label='Prediction')
            axs.set_xlabel('x')
            axs.set_ylabel('G(x,y)')
            axs.set_title('1D Green function')
            axs.legend()
            if params['save_figs']:
               figname = params['fig_file_base'] + '_green.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_green.npz'
               np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(), 
                                 G_exact=params['G_exact_plot'][0,:], 
                                 G_pred=params['G_pred_plot'][0,:])
            else:
               plt.show()
            plt.close()
            # also print the l1 and l2 norm of the error to the terminal
            params['G_l1_error'] = np.linalg.norm(params['G_exact_plot'][0,:] - params['G_pred_plot'][0,:], ord=1)
            params['G_l2_error'] = np.linalg.norm(params['G_exact_plot'][0,:] - params['G_pred_plot'][0,:], ord=2)
            params['G_linf_error'] = np.max(np.abs(params['G_exact_plot'][0,:] - params['G_pred_plot'][0,:]))
         else:
            fig, axs = plt.subplots(1, 1, figsize=(10, 5), facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace = .2, wspace=.2)
            axs.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(), params['G_pred_plot'][0,:], label='Prediction')
            axs.set_xlabel('x')
            axs.set_ylabel('G(x,y)')
            axs.set_title('1D Green function')
            axs.legend()
            if params['save_figs']:
               figname = params['fig_file_base'] + '_green.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_green.npz'
               np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(), 
                                 G_pred=params['G_pred_plot'][0,:])
            else:
               plt.show()
            plt.close()
      else:
         ntotal = params['domain']._quad_points.shape[0]
         repeat_x = params['domain']._quad_points.repeat_interleave(ntotal)
         repeat_y = params['domain']._quad_points.repeat(ntotal)
         x = torch.vstack([repeat_x, repeat_y]).T
         params['G_exact_plot'] = params['pde'].sol(x.detach().cpu().numpy())
         if params['G_exact_plot'] is None:
            params['G_pred_plot'] = np.zeros((ntotal,ntotal)).astype(params['dtype_np'])
         else:
            params['G_exact_plot'] = params['G_exact_plot'].reshape((ntotal,ntotal), order='F')
            params['G_pred_plot'] = np.zeros_like(params['G_exact_plot'])
         for domi in range(params['domain']._ndomains):
            idx = np.where(params['domain']._quad_domain.detach().cpu().numpy() == domi)[0]
            ndomi = idx.shape[0]
            xi = torch.vstack([params['domain']._quad_points.repeat_interleave(ndomi),
                              params['domain']._quad_points[idx].repeat(ntotal)]).T
            params['G_pred_plot'][idx,:] = params['model'].eval(xi, domain_num=domi).detach().cpu().numpy().squeeze().reshape((ndomi, ntotal), order='F')
            if torch.cuda.is_available():
               torch.cuda.empty_cache()
         if params['domain_y'] is not None:
            # in this case we set all the y outside the domain to 0
            out_points_y = ~params['domain_y'].isinside(params['domain']._quad_points).repeat(ntotal).detach().cpu().numpy().reshape((ntotal,ntotal), order='F')
            if params['G_exact_plot'] is not None:
               params['G_exact_plot'][out_points_y] = 0
            params['G_pred_plot'][out_points_y] = 0
         if params['G_exact_plot'] is None:
            fig, axs = plt.subplots(1, 1, figsize=(5, 5), facecolor='w', edgecolor='k')
            im = axs.imshow(params['G_pred_plot'], cmap='jet')
            axs.set_title('Prediction')
            axs.set_ylim(axs.get_ylim()[::-1])
            axs.axis('off')
            fig.colorbar(im, ax=axs)
            if params['save_figs']:
               figname = params['fig_file_base'] + '_green.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_green.npz'
               np.savez(npzname, x=params['domain']._quad_points.detach().cpu().numpy(), 
                                 G_pred=params['G_pred_plot'])
            else:
               plt.show()
            plt.close()
         else:
            G_error_plot = np.abs(params['G_exact_plot'] - params['G_pred_plot'])
            params['G_min_plot'] = min(min(np.min(params['G_exact_plot']), np.min(params['G_pred_plot'])), np.min(G_error_plot))
            params['G_max_plot'] = max(max(np.max(params['G_exact_plot']), np.max(params['G_pred_plot'])), np.max(G_error_plot))
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace = .2, wspace=.2)
            axs = axs.ravel()
            im = axs[0].imshow(params['G_exact_plot'], cmap='jet', vmin=params['G_min_plot'], vmax=params['G_max_plot'])
            axs[0].set_title('Exact')
            axs[0].set_ylim(axs[0].get_ylim()[::-1])
            axs[0].axis('off')
            fig.colorbar(im, ax=axs[0])
            im = axs[1].imshow(params['G_pred_plot'], cmap='jet', vmin=params['G_min_plot'], vmax=params['G_max_plot'])
            axs[1].set_title('Prediction')
            axs[1].set_ylim(axs[1].get_ylim()[::-1])
            axs[1].axis('off')
            fig.colorbar(im, ax=axs[1])
            if err_max == 0:
               im = axs[2].imshow(G_error_plot, cmap='jet')
            elif err_max <= 0:
               im = axs[2].imshow(G_error_plot, cmap='jet', vmin=params['G_min_plot'], vmax=params['G_max_plot'])
            else:
               im = axs[2].imshow(G_error_plot, cmap='jet', vmin=0, vmax=err_max)
            axs[2].set_title('Error')
            axs[2].set_ylim(axs[2].get_ylim()[::-1])
            axs[2].axis('off')
            fig.colorbar(im, ax=axs[2])
            if params['save_figs']:
               figname = params['fig_file_base'] + '_green.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_green.npz'
               np.savez(npzname, x=params['domain']._quad_points.detach().cpu().numpy(), 
                                 G_exact=params['G_exact_plot'], 
                                 G_pred=params['G_pred_plot'])
            else:
               plt.show()
            plt.close()
   return params

def plot_1dgreen_fem(params, err_max = 0.0, refine_level = 1, fem_only = False):
   '''
   Compare GPINNs solution with fem solution.
   '''
   if fem_only:
      if params['dim'] == 1:
         if params['domain_y'] is not None and params['domain_y']._range[0] == params['domain_y']._range[1]:
            ntotal = params['domain']._quad_mesh_points.shape[0]
            y = params['domain_y'].sample_uniform(1)[0].repeat(ntotal)
            x = torch.vstack([params['domain']._quad_mesh_points, y]).T
            
            if torch.cuda.is_available():
               torch.cuda.empty_cache()
            
            # next we need to compute the corresponding numerical solution.
            # we need a fine mesh for computing that 
            # Our right-hand side for 1D is the Gaussian 

            rhs_fenics = f"1.0 / ({params['eps']} * sqrt(pi)) * exp(- pow(x[0] - {params['domain_y']._range[0]}, 2) / pow({params['eps']}, 2))"

            try: 
               import fenics
               mesh = fenics.IntervalMesh(params['domain']._quad_mesh_points.shape[0], 
                                          params['domain']._range.detach().cpu().numpy()[0], 
                                          params['domain']._range.detach().cpu().numpy()[1])
               for _ in range(refine_level): mesh = fenics.refine(mesh)
               V = fenics.FunctionSpace(mesh, 'P', 1)
               u_D = fenics.Constant(0)
               bc = fenics.DirichletBC(V, u_D, 'on_boundary')
               u = fenics.TrialFunction(V)
               v = fenics.TestFunction(V)
               f = fenics.Expression(rhs_fenics, degree=5, domain=mesh)
               L = params['pde'].weak_form(V,V)
               R = f * v * fenics.dx
               sol_func = fenics.Function(V)
               fenics.solve(L == R, sol_func, bc)
               params['G_fem_plot'] = np.array([sol_func(fenics.Point(p)) for p in params['domain']._quad_mesh_points.detach().cpu().numpy()])
            except ImportError:
               print("FEniCS not installed, skipped")
               return params

            fig, axs = plt.subplots(1, 1, figsize=(10, 5), facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace = .2, wspace=.2)
            axs.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(), params['G_fem_plot'], label='FEM')
            axs.set_xlabel('x')
            axs.set_ylabel('G(x,y)')
            axs.set_title('1D Green function')
            axs.legend()
            if params['save_figs']:
               figname = params['fig_file_base'] + '_green_fem.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_green_fem.npz'
               np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(), 
                                 G_fem=params['G_fem_plot'])
            else:
               plt.show()
            plt.close()
         else:
            # run FEM for each y value! This can be expensive
            ntotal = params['domain']._quad_points.shape[0]
            repeat_x = params['domain']._quad_points.repeat_interleave(ntotal)
            repeat_y = params['domain']._quad_points.repeat(ntotal)
            x = torch.vstack([repeat_x, repeat_y]).T
            
            # next compute FEM solution
            if params['domain_y'] is not None:
               # in this case we set all the y outside the domain to 0
               
               try:
                  import fenics
                  params['G_fem_plot'] = np.zeros((ntotal,ntotal)).astype(params['dtype_np'])
                  mesh = fenics.IntervalMesh(params['domain']._quad_mesh_points.shape[0], 
                                             params['domain']._range.detach().cpu().numpy()[0], 
                                             params['domain']._range.detach().cpu().numpy()[1])
                  for _ in range(refine_level): mesh = fenics.refine(mesh)
                  V = fenics.FunctionSpace(mesh, 'P', 1)
                  u_D = fenics.Constant(0)
                  bc = fenics.DirichletBC(V, u_D, 'on_boundary')
                  u = fenics.TrialFunction(V)
                  v = fenics.TestFunction(V)
                  L = params['pde'].weak_form(V,V)
                  sol_func = fenics.Function(V)
                  from tqdm.auto import tqdm 
                  with tqdm(total=params['domain']._quad_points.shape[0], disable = params['hide_tqdm_bar']) as pbar:
                     for i in range(params['domain']._quad_points.shape[0]):
                        if params['domain_y'].isinside(params['domain']._quad_points[i]):
                           rhs_fenics = f"1.0 / ({params['eps']} * sqrt(pi)) * exp(- pow(x[0] - {params['domain']._quad_points[i]}, 2) / pow({params['eps']}, 2))"
                           f = fenics.Expression(rhs_fenics, degree=5, domain=mesh)
                           R = f * v * fenics.dx
                           fenics.solve(L == R, sol_func, bc)
                           params['G_fem_plot'][i,:] = np.array([sol_func(fenics.Point(p)) for p in params['domain']._quad_points.detach().cpu().numpy()])
                        pbar.update(1)
               except ImportError:
                  print("FEniCS not installed, skipped")
                  return params
            else:
               try:
                  import fenics
                  params['G_fem_plot'] = np.zeros((ntotal,ntotal)).astype(params['dtype_np'])
                  mesh = fenics.IntervalMesh(params['domain']._quad_mesh_points.shape[0], 
                                             params['domain']._range.detach().cpu().numpy()[0], 
                                             params['domain']._range.detach().cpu().numpy()[1])
                  for _ in range(refine_level): mesh = fenics.refine(mesh)
                  V = fenics.FunctionSpace(mesh, 'P', 1)
                  u_D = fenics.Constant(0)
                  bc = fenics.DirichletBC(V, u_D, 'on_boundary')
                  u = fenics.TrialFunction(V)
                  v = fenics.TestFunction(V)
                  L = params['pde'].weak_form(V,V)
                  sol_func = fenics.Function(V)
                  from tqdm.auto import tqdm
                  with tqdm(total=params['domain']._quad_points.shape[0], disable = params['hide_tqdm_bar']) as pbar:
                     for i in range(params['domain']._quad_points.shape[0]):
                        rhs_fenics = f"1.0 / ({params['eps']} * sqrt(pi)) * exp(- pow(x[0] - {params['domain']._quad_points[i]}, 2) / pow({params['eps']}, 2))"
                        f = fenics.Expression(rhs_fenics, degree=5, domain=mesh)
                        R = f * v * fenics.dx
                        fenics.solve(L == R, sol_func, bc)
                        params['G_fem_plot'][i,:] = np.array([sol_func(fenics.Point(p)) for p in params['domain']._quad_points.detach().cpu().numpy()])
                        pbar.update(1)
               except ImportError:
                  print("FEniCS not installed, skipped")
                  return params

            params['G_min_plot_fem'] = np.min(params['G_fem_plot'])
            params['G_max_plot_fem'] = np.max(params['G_fem_plot'])
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace = .2, wspace=.2)
            im = ax.imshow(params['G_fem_plot'], cmap='jet', vmin=params['G_min_plot_fem'], vmax=params['G_max_plot_fem'])
            ax.set_title('FEM')
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.axis('off')
            fig.colorbar(im, ax=ax)
            if params['save_figs']:
               figname = params['fig_file_base'] + '_green_fem.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_green_fem.npz'
               np.savez(npzname, x=params['domain']._quad_points.detach().cpu().numpy(), 
                                 G_fem=params['G_fem_plot'])
            else:
               plt.show()
            plt.close()
   else:
      if params['dim'] == 1:
         if params['domain_y'] is not None and params['domain_y']._range[0] == params['domain_y']._range[1]:
            ntotal = params['domain']._quad_mesh_points.shape[0]
            y = params['domain_y'].sample_uniform(1)[0].repeat(ntotal)
            x = torch.vstack([params['domain']._quad_mesh_points, y]).T
            
            params['G_pred_plot'] = np.zeros((1,ntotal)).astype(params['dtype_np'])
            params['G_pred_plot'] = params['model'].eval(x).detach().cpu().numpy().squeeze().reshape((1, ntotal), order='F')
            
            if torch.cuda.is_available():
               torch.cuda.empty_cache()
            
            # next we need to compute the corresponding numerical solution.
            # we need a fine mesh for computing that 
            # Our right-hand side for 1D is the Gaussian 

            rhs_fenics = f"1.0 / ({params['eps']} * sqrt(pi)) * exp(- pow(x[0] - {params['domain_y']._range[0]}, 2) / pow({params['eps']}, 2))"

            try: 
               import fenics
               mesh = fenics.IntervalMesh(params['domain']._quad_mesh_points.shape[0], 
                                          params['domain']._range.detach().cpu().numpy()[0], 
                                          params['domain']._range.detach().cpu().numpy()[1])
               for _ in range(refine_level): mesh = fenics.refine(mesh)
               V = fenics.FunctionSpace(mesh, 'P', 1)
               u_D = fenics.Constant(0)
               bc = fenics.DirichletBC(V, u_D, 'on_boundary')
               u = fenics.TrialFunction(V)
               v = fenics.TestFunction(V)
               f = fenics.Expression(rhs_fenics, degree=5, domain=mesh)
               L = params['pde'].weak_form(V,V)
               R = f * v * fenics.dx
               sol_func = fenics.Function(V)
               fenics.solve(L == R, sol_func, bc)
               params['G_fem_plot'] = np.array([sol_func(fenics.Point(p)) for p in params['domain']._quad_mesh_points.detach().cpu().numpy()])
            except ImportError:
               print("FEniCS not installed, skipped")
               return params

            fig, axs = plt.subplots(1, 1, figsize=(10, 5), facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace = .2, wspace=.2)
            axs.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(), params['G_fem_plot'], label='FEM')
            axs.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(), params['G_pred_plot'][0,:], label='Prediction')
            axs.set_xlabel('x')
            axs.set_ylabel('G(x,y)')
            axs.set_title('1D Green function')
            axs.legend()
            if params['save_figs']:
               figname = params['fig_file_base'] + '_green_fem.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_green_fem.npz'
               np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(), 
                                 G_fem=params['G_fem_plot'], 
                                 G_pred=params['G_pred_plot'][0,:])
               npzname = params['fig_file_base'] + '_tri.npz'
               np.savez(npzname, x = params['domain']._quad_mesh_points[...,0].detach().cpu().numpy(), 
                                    y = params['domain']._quad_mesh_points[...,1].detach().cpu().numpy(), 
                                    tri = params['domain']._quad_mesh_tris.detach().cpu().numpy())
            else:
               plt.show()
            plt.close()
            # also print the l1 and l2 norm of the error to the terminal
            params['G_l1_error_fem'] = np.linalg.norm(params['G_fem_plot'] - params['G_pred_plot'][0,:], ord=1)
            params['G_l2_error_fem'] = np.linalg.norm(params['G_fem_plot']- params['G_pred_plot'][0,:], ord=2)
            params['G_linf_error_fem'] = np.max(np.abs(params['G_fem_plot'] - params['G_pred_plot'][0,:]))
         else:
            # run FEM for each y value! This can be expensive
            ntotal = params['domain']._quad_points.shape[0]
            repeat_x = params['domain']._quad_points.repeat_interleave(ntotal)
            repeat_y = params['domain']._quad_points.repeat(ntotal)
            x = torch.vstack([repeat_x, repeat_y]).T
            
            params['G_pred_plot'] = np.zeros((ntotal,ntotal)).astype(params['dtype_np'])

            for domi in range(params['domain']._ndomains):
               idx = np.where(params['domain']._quad_domain.detach().cpu().numpy() == domi)[0]
               ndomi = idx.shape[0]
               xi = torch.vstack([params['domain']._quad_points.repeat_interleave(ndomi),
                                 params['domain']._quad_points[idx].repeat(ntotal)]).T
               params['G_pred_plot'][idx,:] = params['model'].eval(xi, domain_num=domi).detach().cpu().numpy().squeeze().reshape((ndomi, ntotal), order='F')
               if torch.cuda.is_available():
                  torch.cuda.empty_cache()

            # next compute FEM solution
            if params['domain_y'] is not None:
               # in this case we set all the y outside the domain to 0
               out_points_y = ~params['domain_y'].isinside(params['domain']._quad_points).repeat(ntotal).detach().cpu().numpy().reshape((ntotal,ntotal), order='F')
               if params['G_exact_plot'] is not None:
                  params['G_exact_plot'][out_points_y] = 0
               params['G_pred_plot'][out_points_y] = 0
               
               try:
                  import fenics
                  params['G_fem_plot'] = np.zeros_like(params['G_pred_plot'])
                  mesh = fenics.IntervalMesh(params['domain']._quad_mesh_points.shape[0], 
                                             params['domain']._range.detach().cpu().numpy()[0], 
                                             params['domain']._range.detach().cpu().numpy()[1])
                  for _ in range(refine_level): mesh = fenics.refine(mesh)
                  V = fenics.FunctionSpace(mesh, 'P', 1)
                  u_D = fenics.Constant(0)
                  bc = fenics.DirichletBC(V, u_D, 'on_boundary')
                  u = fenics.TrialFunction(V)
                  v = fenics.TestFunction(V)
                  L = params['pde'].weak_form(V,V)
                  sol_func = fenics.Function(V)
                  from tqdm.auto import tqdm 
                  with tqdm(total=params['domain']._quad_points.shape[0], disable = params['hide_tqdm_bar']) as pbar:
                     for i in range(params['domain']._quad_points.shape[0]):
                        if params['domain_y'].isinside(params['domain']._quad_points[i]):
                           rhs_fenics = f"1.0 / ({params['eps']} * sqrt(pi)) * exp(- pow(x[0] - {params['domain']._quad_points[i]}, 2) / pow({params['eps']}, 2))"
                           f = fenics.Expression(rhs_fenics, degree=5, domain=mesh)
                           R = f * v * fenics.dx
                           fenics.solve(L == R, sol_func, bc)
                           params['G_fem_plot'][i,:] = np.array([sol_func(fenics.Point(p)) for p in params['domain']._quad_points.detach().cpu().numpy()])
                        pbar.update(1)
               except ImportError:
                  print("FEniCS not installed, skipped")
                  return params
            else:
               try:
                  import fenics
                  params['G_fem_plot'] = np.zeros_like(params['G_pred_plot'])
                  mesh = fenics.IntervalMesh(params['domain']._quad_mesh_points.shape[0], 
                                             params['domain']._range.detach().cpu().numpy()[0], 
                                             params['domain']._range.detach().cpu().numpy()[1])
                  for _ in range(refine_level): mesh = fenics.refine(mesh)
                  V = fenics.FunctionSpace(mesh, 'P', 1)
                  u_D = fenics.Constant(0)
                  bc = fenics.DirichletBC(V, u_D, 'on_boundary')
                  u = fenics.TrialFunction(V)
                  v = fenics.TestFunction(V)
                  L = params['pde'].weak_form(V,V)
                  sol_func = fenics.Function(V)
                  from tqdm.auto import tqdm
                  with tqdm(total=params['domain']._quad_points.shape[0], disable = params['hide_tqdm_bar']) as pbar:
                     for i in range(params['domain']._quad_points.shape[0]):
                        rhs_fenics = f"1.0 / ({params['eps']} * sqrt(pi)) * exp(- pow(x[0] - {params['domain']._quad_points[i]}, 2) / pow({params['eps']}, 2))"
                        f = fenics.Expression(rhs_fenics, degree=5, domain=mesh)
                        R = f * v * fenics.dx
                        fenics.solve(L == R, sol_func, bc)
                        params['G_fem_plot'][i,:] = np.array([sol_func(fenics.Point(p)) for p in params['domain']._quad_points.detach().cpu().numpy()])
                        pbar.update(1)
               except ImportError:
                  print("FEniCS not installed, skipped")
                  return params

            G_error_plot = np.abs(params['G_fem_plot'] - params['G_pred_plot'])
            params['G_min_plot_fem'] = min(min(np.min(params['G_fem_plot']), np.min(params['G_pred_plot'])), np.min(G_error_plot))
            params['G_max_plot_fem'] = max(max(np.max(params['G_fem_plot']), np.max(params['G_pred_plot'])), np.max(G_error_plot))
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace = .2, wspace=.2)
            axs = axs.ravel()
            im = axs[0].imshow(params['G_fem_plot'], cmap='jet', vmin=params['G_min_plot_fem'], vmax=params['G_max_plot_fem'])
            axs[0].set_title('FEM')
            axs[0].set_ylim(axs[0].get_ylim()[::-1])
            axs[0].axis('off')
            fig.colorbar(im, ax=axs[0])
            im = axs[1].imshow(params['G_pred_plot'], cmap='jet', vmin=params['G_min_plot_fem'], vmax=params['G_max_plot_fem'])
            axs[1].set_title('Prediction')
            axs[1].set_ylim(axs[1].get_ylim()[::-1])
            axs[1].axis('off')
            fig.colorbar(im, ax=axs[1])
            if err_max == 0:
               im = axs[2].imshow(G_error_plot, cmap='jet')
            elif err_max <= 0:
               im = axs[2].imshow(G_error_plot, cmap='jet', vmin=params['G_min_plot'], vmax=params['G_max_plot'])
            else:
               im = axs[2].imshow(G_error_plot, cmap='jet', vmin=0, vmax=err_max)
            axs[2].set_title('Error')
            axs[2].set_ylim(axs[2].get_ylim()[::-1])
            axs[2].axis('off')
            fig.colorbar(im, ax=axs[2])
            if params['save_figs']:
               figname = params['fig_file_base'] + '_green_fem.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_green_fem.npz'
               np.savez(npzname, x=params['domain']._quad_points.detach().cpu().numpy(), 
                                 G_fem=params['G_fem_plot'], 
                                 G_pred=params['G_pred_plot'])
            else:
               plt.show()
            plt.close()
   return params
   
def validation_eval(params):
   if params['dim'] == 1:
      nquad = params['domain']._quad_points.shape[0]
      ngrid = params['domain']._quad_mesh_points.shape[0]
      repeat_x = params['domain']._quad_mesh_points.repeat_interleave(nquad)
      repeat_y = params['domain']._quad_points.repeat(ngrid)
      x = torch.vstack([repeat_x, repeat_y]).T
      params['G_exact'] = params['pde'].sol(x.detach().cpu().numpy())
      if params['G_exact'] is None:
         params['G_pred'] = np.zeros((nquad,ngrid)).astype(params['dtype_np'])
      else:
         params['G_exact'] = params['G_exact'].reshape((nquad,ngrid), order='F')
         params['G_pred'] = np.zeros_like(params['G_exact'])
      for domi in range(params['domain']._ndomains):
         idx = np.where(params['domain']._quad_domain.detach().cpu().numpy() == domi)[0]
         ndomi = idx.shape[0]
         xi = torch.vstack([params['domain']._quad_mesh_points.repeat_interleave(ndomi),
                           params['domain']._quad_points[idx].repeat(ngrid)]).T
         params['G_pred'][idx,:] = params['model'].eval(xi, domain_num=domi, all_inuse = True).detach().cpu().numpy().squeeze().reshape((ndomi, ngrid), order='F')
         if torch.cuda.is_available():
            torch.cuda.empty_cache()
   elif params['dim'] == 2:
      def eval_2d(params):
         nquad = params['domain']._quad_points.shape[0]
         ngrid = params['domain']._quad_mesh_points.shape[0]
         repeat_x = params['domain']._quad_mesh_points.repeat_interleave(nquad,dim=0)
         repeat_y = params['domain']._quad_points.repeat(ngrid,1)
         x = torch.hstack([repeat_x, repeat_y])
         params['G_exact'] = params['pde'].sol(x.detach().cpu().numpy())
         if params['G_exact'] is None:
            params['G_pred'] = np.zeros((nquad,ngrid)).astype(params['dtype_np'])
         else:
            params['G_exact'] = params['G_exact'].reshape((nquad,ngrid), order='F')
            params['G_pred'] = np.zeros_like(params['G_exact'])
         for domi in range(params['domain']._ndomains):
            idx = np.where(params['domain']._quad_domain.detach().cpu().numpy() == domi)[0]
            ndomi = idx.shape[0]
            xi = torch.hstack([params['domain']._quad_mesh_points.repeat_interleave(ndomi,dim=0),
                              params['domain']._quad_points[idx,:].repeat(ngrid,1)])
            params['G_pred'][idx,:] = params['model'].eval(xi, domain_num=domi, all_inuse = True).detach().cpu().numpy().squeeze().reshape((ndomi, ngrid), order='F')
            if torch.cuda.is_available():
               torch.cuda.empty_cache()
         return params
      try:
         params = eval_2d(params)
      except RuntimeError as err:
         if 'CUDA' in str(err):
            print('CUDA error, try to run on cpu')
            params['model'].update_device(torch.device('cpu'))
            params['device'] = torch.device('cpu')
            params = eval_2d(params)
         else:
            print("Error: ", err)
            pass
      # nothin to visualize in this case
   return params

def validation_solve(params, rhs_func, rhs_fenics = None, test_name = "", refine_level = 0):
   if params['dim'] == 1:
      if params['G_exact'] is None:

         if rhs_fenics is not None:
            try: 
               import fenics
               mesh = fenics.IntervalMesh(params['domain']._quad_mesh_points.shape[0], 
                                          params['domain']._range.detach().cpu().numpy()[0], 
                                          params['domain']._range.detach().cpu().numpy()[1])
               for _ in range(refine_level): mesh = fenics.refine(mesh)
               V = fenics.FunctionSpace(mesh, 'P', 1)
               u_D = fenics.Constant(0)
               bc = fenics.DirichletBC(V, u_D, 'on_boundary')
               u = fenics.TrialFunction(V)
               v = fenics.TestFunction(V)
               f = fenics.Expression(rhs_fenics, degree=5, domain=mesh)
               L = params['pde'].weak_form(V,V)
               R = f * v * fenics.dx
               sol_func = fenics.Function(V)
               fenics.solve(L == R, sol_func, bc)
               sol_fem = np.array([sol_func(fenics.Point(p)) for p in params['domain']._quad_mesh_points.detach().cpu().numpy()])
            except ImportError:
               print("FEniCS not installed, skipping FEM sol")
               sol_fem = None
         else:
            sol_fem = None
         rhs = rhs_func(params['domain']._quad_points.detach().cpu().numpy())
         sol = np.einsum('ba,b,b->a', params['G_pred'], rhs, params['domain']._quad_weights.detach().cpu().numpy())
         if sol_fem is not None:
            plt.figure(figsize=(5,5))
            plt.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(),sol, label='MSGPINNs')
            plt.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(),sol_fem, label='FEM')
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.title('1D Problem Solution')
            plt.legend()
            if params['save_figs']:
               figname = params['fig_file_base'] + '_sol_' + test_name + '.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_sol_' + test_name + '.npz'
               np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(), 
                                 sol=sol, 
                                 sol_fem=sol_fem)
            else:
               plt.show()
            plt.close()
         else:
            plt.figure(figsize=(5,5))
            plt.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(),sol, label='MSGPINNs')
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.title('1D Problem Solution')
            plt.legend()
            if params['save_figs']:
               figname = params['fig_file_base'] + '_sol_' + test_name + '.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_sol_' + test_name + '.npz'
               np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(),
                                 sol=sol)
            else:
               plt.show()
            plt.close()
      else:
         rhs = rhs_func(params['domain']._quad_points.detach().cpu().numpy())
         sol = np.einsum('ba,b,b->a', params['G_pred'], rhs, params['domain']._quad_weights.detach().cpu().numpy())
         sol_exact = np.einsum('ba,b,b->a', params['G_exact'], rhs, params['domain']._quad_weights.detach().cpu().numpy())

         plt.figure(figsize=(5,5))
         plt.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(),sol, label='MSGPINNs')
         plt.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(),sol_exact, label='Exact')
         plt.xlabel('x')
         plt.ylabel('u(x)')
         plt.title('1D Problem Solution')
         plt.legend()
         if params['save_figs']:
            figname = params['fig_file_base'] + '_sol_' + test_name + '.png'
            plt.savefig(figname)
            npzname = params['fig_file_base'] + '_sol_' + test_name + '.npz'
            np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(),
                          sol=sol,
                          sol_exact=sol_exact)
         else:
            plt.show()
         plt.close()
   elif params['dim'] == 2:
      if params['G_exact'] is None:
         if rhs_fenics is not None:
            try: 
               import fenics
               mesh = fenics.Mesh()
               with fenics.XDMFFile(params['mesh_file_base'] + ".xdmf") as infile:
                  infile.read(mesh)
               for _ in range(refine_level): mesh = fenics.refine(mesh)
               V = fenics.FunctionSpace(mesh, 'P', 1)
               u_D = fenics.Constant(0)
               bc = fenics.DirichletBC(V, u_D, 'on_boundary')
               u = fenics.TrialFunction(V)
               v = fenics.TestFunction(V)
               f = fenics.Expression(rhs_fenics, degree=5, domain=mesh)
               L = params['pde'].weak_form(V,V)
               R = f * v * fenics.dx
               sol_func = fenics.Function(V)
               fenics.solve(L == R, sol_func, bc)
               sol_fem = np.array([sol_func(fenics.Point(*p)) for p in params['domain']._quad_mesh_points.detach().cpu().numpy()])
            except ImportError:
               print("FEniCS not installed, skipping FEM sol")
               sol_fem = None
         
         rhs = rhs_func(params['domain']._quad_points.detach().cpu().numpy())
         sol = np.einsum('ba,b,b->a', params['G_pred'], rhs, params['domain']._quad_weights.detach().cpu().numpy())

         if sol_fem is not None:
            vmin = min(sol.min(), sol_fem.min())
            vmax = max(sol.max(), sol_fem.max())
            fig = plt.figure(figsize=(15,5))
            ax = fig.add_subplot(131)
            im = ax.tricontourf(params['tri'], sol_fem, cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title('FEM solution')
            fig.colorbar(im, ax=ax)
            ax.set_aspect('equal', adjustable='box')
            ax = fig.add_subplot(132)
            im = ax.tricontourf(params['tri'], sol, cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title('Predicted solution')
            fig.colorbar(im, ax=ax)
            ax.set_aspect('equal', adjustable='box')
            ax = fig.add_subplot(133)
            im = ax.tricontourf(params['tri'], sol-sol_fem, cmap='jet')
            ax.set_title('Diff')
            fig.colorbar(im, ax=ax)
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            if params['save_figs']:
               figname = params['fig_file_base'] + '_sol_' + test_name + '.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_sol_' + test_name + '.npz'
               np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(),
                                 sol=sol,
                                 sol_fem=sol_fem)
               npzname = params['fig_file_base'] + '_tri.npz'
               np.savez(npzname, x = params['domain']._quad_mesh_points[...,0].detach().cpu().numpy(), 
                                    y = params['domain']._quad_mesh_points[...,1].detach().cpu().numpy(), 
                                    tri = params['domain']._quad_mesh_tris.detach().cpu().numpy())
            else:
               plt.show()
            plt.close()
         else:
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111)
            ax.tricontourf(params['tri'], sol, cmap='jet')
            ax.set_title('Predicted solution')
            ax.set_aspect('equal', adjustable='box')
            if params['save_figs']:
               figname = params['fig_file_base'] + '_sol_' + test_name + '.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_sol_' + test_name + '.npz'
               np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(),
                                 sol=sol)
               npzname = params['fig_file_base'] + '_tri.npz'
               np.savez(npzname, x = params['domain']._quad_mesh_points[...,0].detach().cpu().numpy(), 
                                    y = params['domain']._quad_mesh_points[...,1].detach().cpu().numpy(), 
                                    tri = params['domain']._quad_mesh_tris.detach().cpu().numpy())
            else:
               plt.show()
            plt.close()
      else:
         rhs = rhs_func(params['domain']._quad_points.detach().cpu().numpy())
         sol = np.einsum('ba,b,b->a', params['G_pred'], rhs, params['domain']._quad_weights.detach().cpu().numpy())
         sol_exact = np.einsum('ba,b,b->a', params['G_exact'], rhs, params['domain']._quad_weights.detach().cpu().numpy())

         vmin = min(sol.min(), sol_exact.min())
         vmax = max(sol.max(), sol_exact.max())
         fig = plt.figure(figsize=(15,5))
         ax = fig.add_subplot(131)
         im = ax.tricontourf(params['tri'], sol_exact, cmap='jet', vmin=vmin, vmax=vmax)
         ax.set_title('Exact Green solution')
         fig.colorbar(im, ax=ax)
         ax.set_aspect('equal', adjustable='box')
         ax = fig.add_subplot(132)
         im = ax.tricontourf(params['tri'], sol, cmap='jet', vmin=vmin, vmax=vmax)
         ax.set_title('Predicted solution')
         fig.colorbar(im, ax=ax)
         ax.set_aspect('equal', adjustable='box')
         ax = fig.add_subplot(133)
         im = ax.tricontourf(params['tri'], sol-sol_exact, cmap='jet')
         ax.set_title('Diff')
         fig.colorbar(im, ax=ax)
         ax.set_aspect('equal', adjustable='box')
         plt.tight_layout()
         if params['save_figs']:
            figname = params['fig_file_base'] + '_sol_' + test_name + '.png'
            plt.savefig(figname)
            npzname = params['fig_file_base'] + '_sol_' + test_name + '.npz'
            np.savez(npzname, x=params['domain']._quad_mesh_points.detach().cpu().numpy(),
                          sol=sol,
                          sol_exact=sol_exact)
            npzname = params['fig_file_base'] + '_tri.npz'
            np.savez(npzname, x = params['domain']._quad_mesh_points[...,0].detach().cpu().numpy(), 
                                 y = params['domain']._quad_mesh_points[...,1].detach().cpu().numpy(), 
                                 tri = params['domain']._quad_mesh_tris.detach().cpu().numpy())
         else:
            plt.show()
         plt.close()
   else:
      pass

def numerical_solve(params, rhs_func, rhs_fenics = None, test_name = ""):
   if params['dim'] == 1:
      if 'G_exact' not in params or params['G_exact'] is None:
         print("No G_exact, generating FEM solution")
         if rhs_fenics is not None:
            try: 
               import fenics
               mesh = fenics.IntervalMesh(params['domain']._quad_mesh_points.shape[0], 
                                          params['domain']._range.detach().cpu().numpy()[0], 
                                          params['domain']._range.detach().cpu().numpy()[1])
               for _ in range(refine_level): mesh = fenics.refine(mesh)
               V = fenics.FunctionSpace(mesh, 'P', 1)
               u_D = fenics.Constant(0)
               bc = fenics.DirichletBC(V, u_D, 'on_boundary')
               u = fenics.TrialFunction(V)
               v = fenics.TestFunction(V)
               f = fenics.Expression(rhs_fenics, degree=5, domain=mesh)
               L = params['pde'].weak_form(V,V)
               R = f * v * fenics.dx
               sol_func = fenics.Function(V)
               fenics.solve(L == R, sol_func, bc)
               sol_fem = np.array([sol_func(fenics.Point(p)) for p in params['domain']._quad_mesh_points.detach().cpu().numpy()])
            except ImportError:
               print("FEniCS not installed, skipping FEM sol")
               sol_fem = None
         else:
            sol_fem = None
         if sol_fem is not None:
            plt.figure(figsize=(5,5))
            plt.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(), sol_fem, label='FEM')
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.title('1D Problem Solution')
            plt.legend()
            if params['save_figs']:
               figname = params['fig_file_base'] + '_numerical_sol_' + test_name + '.png'
               plt.savefig(figname)
            else:
               plt.show()
            plt.close()
         else:
            print("Notiong to plot")
      else:
         rhs = rhs_func(params['domain']._quad_points.detach().cpu().numpy())
         sol_exact = np.einsum('ba,b,b->a', params['G_exact'], rhs, params['domain']._quad_weights.detach().cpu().numpy())

         plt.figure(figsize=(5,5))
         plt.plot(params['domain']._quad_mesh_points.detach().cpu().numpy(),sol_exact, label='Exact')
         plt.xlabel('x')
         plt.ylabel('u(x)')
         plt.title('1D Problem Solution')
         plt.legend()
         if params['save_figs']:
            figname = params['fig_file_base'] + '_numerical_sol_' + test_name + '.png'
            plt.savefig(figname)
         else:
            plt.show()
         plt.close()
   elif params['dim'] == 2:
      if 'G_exact' not in params or params['G_exact'] is None:
         print("No G_exact, generating FEM solution")
         if rhs_fenics is not None:
            try: 
               import fenics
               mesh = fenics.Mesh()
               with fenics.XDMFFile(params['mesh_file_base'] + ".xdmf") as infile:
                  infile.read(mesh)
               for _ in range(refine_level): mesh = fenics.refine(mesh)
               V = fenics.FunctionSpace(mesh, 'P', 1)
               u_D = fenics.Constant(0)
               bc = fenics.DirichletBC(V, u_D, 'on_boundary')
               u = fenics.TrialFunction(V)
               v = fenics.TestFunction(V)
               f = fenics.Expression(rhs_fenics, degree=5, domain=mesh)
               L = params['pde'].weak_form(V,V)
               R = f * v * fenics.dx
               sol_func = fenics.Function(V)
               fenics.solve(L == R, sol_func, bc)
               sol_fem = np.array([sol_func(fenics.Point(*p)) for p in params['domain']._quad_mesh_points.detach().cpu().numpy()])
            except ImportError:
               print("FEniCS not installed, skipping FEM sol")
               sol_fem = None
         if sol_fem is not None:
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111)
            im = ax.tricontourf(params['tri'], sol_fem, cmap='jet')
            ax.set_title('FEM solution')
            fig.colorbar(im, ax=ax)
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            if params['save_figs']:
               figname = params['fig_file_base'] + '_numerical_sol_' + test_name + '.png'
               plt.savefig(figname)
               npzname = params['fig_file_base'] + '_tri.npz'
               np.savez(npzname, x = params['domain']._quad_mesh_points[...,0].detach().cpu().numpy(), 
                                    y = params['domain']._quad_mesh_points[...,1].detach().cpu().numpy(), 
                                    tri = params['domain']._quad_mesh_tris.detach().cpu().numpy())
            else:
               plt.show()
            plt.close()
         else:
            print("Notiong to plot")
      else:
         rhs = rhs_func(params['domain']._quad_points.detach().cpu().numpy())
         sol_exact = np.einsum('ba,b,b->a', params['G_exact'], rhs, params['domain']._quad_weights.detach().cpu().numpy())

         fig = plt.figure(figsize=(5,5))
         ax = fig.add_subplot(111)
         im = ax.tricontourf(params['tri'], sol_exact, cmap='jet')
         ax.set_title('Exact Green solution')
         fig.colorbar(im, ax=ax)
         ax.set_aspect('equal', adjustable='box')
         plt.tight_layout()
         if params['save_figs']:
            figname = params['fig_file_base'] + '_numerical_sol_' + test_name + '.png'
            plt.savefig(figname)
            npzname = params['fig_file_base'] + '_tri.npz'
            np.savez(npzname, x = params['domain']._quad_mesh_points[...,0].detach().cpu().numpy(), 
                                 y = params['domain']._quad_mesh_points[...,1].detach().cpu().numpy(), 
                                 tri = params['domain']._quad_mesh_tris.detach().cpu().numpy())
         else:
            plt.show()
         plt.close()
   else:
      pass