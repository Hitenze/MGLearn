import os
import sys
import glob
from IPython import get_ipython
import torch
import numpy as np
import matplotlib.pyplot as plt

def is_using_notebook():
    try:
        if 'IPKernelApp' not in get_ipython().config:
            return False
        return True
    except Exception:
        return False

def add_parent_dir(base_name, folder_name, num_parent_dirs):
   for i in range(num_parent_dirs):
      base_name = os.path.abspath(os.path.join(base_name, '..'))
   base_name = os.path.join(base_name, folder_name)
   return base_name

def validation_init(paramstr, base_name = None, folder_level = 2):
   using_notebook = is_using_notebook()
   if base_name is None:
      base_name = os.path.abspath('') if using_notebook else os.path.dirname(__file__)
   sys.path.append(add_parent_dir(base_name, 'src', folder_level))
   import gparams
   params = gparams.load_parameters(os.path.join(base_name, paramstr))
   params['import_path'] = add_parent_dir(base_name, 'src', folder_level)
   params['using_notebook'] = using_notebook
   params['base_name'] = base_name
   data_folder = os.path.join(base_name, 'data')
   training_folder = os.path.join(base_name, 'training')
   figs_folder = os.path.join(base_name, 'figs')
   mesh_folder = os.path.join(base_name, 'meshes')
   os.makedirs(data_folder, exist_ok=True)
   os.makedirs(training_folder, exist_ok=True)
   os.makedirs(figs_folder, exist_ok=True)
   os.makedirs(mesh_folder, exist_ok=True)

   print("Running test name: ", params['test_name'])

   model_file_base = os.path.join(data_folder, params['test_name'])
   training_file_base = os.path.join(training_folder, params['test_name'])
   fig_file_base = os.path.join(figs_folder, params['test_name'])
   mesh_file_base = os.path.join(mesh_folder, params['test_name'])

   if params['clear_data']:
      print("Clearing data ...")
      model_files = glob.glob(model_file_base+"*")
      for filename in model_files:
         os.remove(filename)
      training_files = glob.glob(training_file_base+"*")
      for filename in training_files:
         os.remove(filename)
      fig_files = glob.glob(fig_file_base+"*")
      for filename in fig_files:
         os.remove(filename)
      mesh_files = glob.glob(mesh_file_base+"*")
      for filename in mesh_files:
         os.remove(filename)
   else:
      print("Will not clear data ...")

   params['model_file_base'] = model_file_base
   params['training_file_base'] = training_file_base
   params['fig_file_base'] = fig_file_base
   params['mesh_file_base'] = mesh_file_base
   
   params['device'] = torch.device("cuda:0" if torch.cuda.is_available() and params['using_cuda'] else "cpu")
   params['dtype'] = torch.float32 if params['using_fp32'] else torch.float64
   params['dtype_np'] = np.float32 if params['using_fp32'] else np.float64

   print("Running on device: ", params['device'])  
   print("Data type: ", params['dtype'])

   return params