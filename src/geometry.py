# this file contains the basic geometry and data generation functions
import torch
import numpy as np
import meshpy.triangle as triangle
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.tri as tri

class GGeometry:
   '''
   Basic gemotry for Greens functions, which includes:
      1. Line segment for 1D cases, and;
      2. Triangle for 2D cases.
   Variables:
      _d: Integer.
         Dimension of the geometry.
      _device: Pytorch device.
         Pytorch device.
      _dtype: Pytorch data type.
         Data type. (FP64 or FP32)
   '''
   def __init__(self,
                  dim,
                  dtype = torch.float32,
                  device = 'cpu'):
      '''
      Initialize the GGometry, leave constrain to other specific type to handle.
      Inputs:
         dim: Integer.
            Dimension of the geometry. Typically can be obtained via the staticmethod get_dim()
         dtype: Pytorch data type. (Optional, default torch.float32).
            Data type. Can only be torch.float32 or torch.float64.
         device: Pytorch device. (Optional, default 'cpu').
            Pytorch device.
      '''
      self._d = dim
      self._device = device
      self._dtype = dtype

   def update_dtype(self, dtype):
      '''
      Update the data type.
      Inputs:
         dtype: Pytorch data type.
            Data type. Can only be torch.float32 or torch.float64.
      '''
      self._dtype = dtype

   def update_device(self, device):
      '''
      Update the device.
      Inputs:
         device: Pytorch device.
            Pytorch device.
      '''
      self._device = device
   
   def isinside(self, x):
      '''
      Given x which is n by d, check if is inside the geometry.
      Inputs:
         x: Float number.
            Point to check.
      Outputs:
         isinside: Boolean.
            True if in inteval. Otherwise false.
      '''
      return False
   
   def sample_uniform(self, 
                        nx,
                        target_y = None,
                        boundary = False,
                        bc = None,
                        rrange = None,
                        ):
      '''
      Uniformly sample some points inside the geometry, optinally near some target points.
      The output can be either n by d (x or y only) or n by 2d ([x,y] for Greens function).
      Inputs:
         nx: Integer,
            Number of points to sample.
         target_y: Torch tensor, ny by d. (Optional, default None).
            1. None:
                  If target_y is None, output would be a torch tensor of nx by d.
            2. Torch tensor of ny by d: 
                  We sample nx points for each point in target_y, that is a total of n = nx * ny points.
                  The putput would be n by 2d
         boundary: Boolean, optional (Optional, default False).
            If true, we sample on the boundary. Otherwise in the domain.
         bc: Boolean, optional (default None).
            Boundary condition function on $x$ u = bc(x). Takes inpute n by d.
            If None, zero Dirichlet boundary condition will be used
         rrange: Tuple of two numbers. (Optional, default None).
            rrange = (r_min, r_max).
            r_min:
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance smaller than r_min to target_y.
            r_max: Float number, optional (Optional, default -1.0).
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance larger than r_max to target_y.
      Outputs:
         y: n by d tensor or n by 2d tensor.
      '''
      return None
   
   @staticmethod
   def get_dim():
      return 1

class GSegment(GGeometry):
   '''
   Line segment (interval), basic 1D geometry.
   Variables:
      _range: 1D torch tensor of 2 float numbers.
         Range of the line segment (left, right).
   '''
   def __init__(self, 
                  range,
                  dtype = torch.float32,
                  device = 'cpu'):
      '''
      Initialize the line segment.
      Inputs:
         range: Tuple with 2 float numbers.
            Range of the line segment like (0.0, 1.0)
         dtype: Pytorch data type. (Optional, default torch.float32).
            Data type. Can only be torch.float32 or torch.float64.
         device: Pytorch device. (Optional, default 'cpu').
            Pytorch device.
      '''
      super().__init__( 1, dtype = dtype, device = device)
      self._range = torch.tensor(range, dtype = dtype, device = device)

   def update_dtype(self, dtype):
      '''
      Update the data type.
      Inputs:
         dtype: Pytorch data type.
            Data type. Can only be torch.float32 or torch.float64.
      '''
      super().update_dtype(dtype)
      self._range = self._range.to(dtype)

   def update_device(self, device):
      '''
      Update the device.
      Inputs:
         device: Pytorch device.
            Pytorch device.
      '''
      super().update_device(device)
      self._range = self._range.to(device)
   
   def isinside(self, x):
      '''
      Given x which is n by 1, check if in inteval.
      Inputs:
         x: Float number.
            Point to check.
      Outputs:
         isinside: Boolean.
            True if in inteval. Otherwise false.
      '''
      return torch.logical_and(x >= self._range[0], x <= self._range[1])

   def sample_uniform(self, 
                        nx,
                        target_y = None,
                        boundary = False,
                        bc = None,
                        rrange = None,
                        ):
      '''
      Uniformly sample some points inside the geometry, optinally near some target points.
      The output can be either n by d (x or y only) or n by 2d ([x,y] for Greens function).
      Inputs:
         nx: Integer,
            Number of points to sample.
         target_y: Torch tensor, ny by d. (Optional, default None).
            1. None:
                  If target_y is None, output would be a torch tensor of nx by d.
            2. Torch tensor of ny by d: 
                  We sample nx points for each point in target_y, that is a total of n = nx * ny points.
                  The putput would be n by 2d
         boundary: Boolean, optional (Optional, default False).
            If true, we sample on the boundary. Otherwise in the domain.
         bc: Boolean, optional (default None).
            Boundary condition function on $x$ u = bc(x). Takes inpute n by d.
            If None, zero Dirichlet boundary condition will be used
         rrange: Tuple of two numbers. (Optional, default None).
            rrange = (r_min, r_max).
            r_min:
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance smaller than r_min to target_y.
            r_max: Float number, optional (Optional, default -1.0).
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance larger than r_max to target_y.
      Outputs:
         when boundary is False:
            y: n by d tensor or n by 2d tensor.
         when boundary is True:
            y, u_y: n by d tensor or n by 2d tensor together with function value on the boundary
      '''
      if target_y is None:
         # 1. On the entire domain
         if boundary:
            # 1.1 sample on the boundary
            y = self._range.clone().detach().requires_grad_(True).view(-1, 1)
            if bc is not None:
               u_y = bc(y)
            else:
               u_y = torch.zeros(2, 1, dtype=self._dtype).to(self._device)
            return y, u_y
         else:
            # 1.2 sample uniformly in the inteval
            y = torch.rand(nx, 1, dtype=self._dtype, device = self._device) * \
               (self._range[1]-self._range[0]) + self._range[0]
            return y.detach().requires_grad_(True)
      else:
         # 2. Sample near target points
         if rrange is None:
            # 2.1 Sample without limit
            if boundary:
               # 2.1.1 Sample on the boundary
               x = torch.cat([
                  self._range.repeat(target_y.shape[0], 1).detach().view(-1, 1),
                  target_y.repeat_interleave(2, dim = 0).detach().requires_grad_(True)
               ], dim = 1)
               if bc is not None:
                  u_x = bc(x[:self.get_dim(),:])
               else:
                  u_x = torch.zeros(x.shape[0], 1, dtype=self._dtype).to(self._device)
               return x, u_x
            else:
               # 2.1.2 Sample uniformly in the inteval
               return torch.cat([
                  self.sample_uniform(nx * target_y.shape[0]),
                  target_y.repeat_interleave(nx, dim = 0).detach().requires_grad_(True)
               ], dim = 1)
         else:
            # 2.2 Sample within a distance
            r_min = 0.0 if rrange[0] < 0 else rrange[0]
            r_max = r_min if rrange[1] < r_min else rrange[1]
            x = torch.cat([
               torch.zeros(nx * target_y.shape[0], 1, dtype=self._dtype).to(self._device),
               target_y.repeat_interleave(nx, dim = 0).detach().requires_grad_(True)
            ], dim = 1)
            idx = 0
            for i in range(target_y.shape[0]):
               left = torch.max(self._range[0], target_y[i, 0] - r_max)
               right = torch.min(self._range[1], target_y[i, 0] + r_max)
               x[idx:idx+nx, 0] = torch.rand(nx, dtype=self._dtype).to(self._device) * (right - left) + left
               idx += nx
            x.detach().requires_grad_(True)
            return x

   def get_mesh(self, ncell = 20):
      '''
      Generate mesh on the domain.
      Inputs:
         ncell: Integer. (Optional, default 20).
            Number of cells in each direction. The total number of points would be ncell + 1 as we have boundary.
      Outputs:
         mesh_points: Numpy array, 1D.
            The location of the grid points.
         mesh_tris: Numpy integer array, n by 2.
            The start/end location of each 1D element
      '''
      np_dtype = np.float32 if self._dtype == torch.float32 else np.float64
      mesh_points = np.linspace(self._range[0], self._range[1], ncell+1).astype(np_dtype)
      mesh_tris = np.stack((np.arange(ncell), np.arange(1,ncell+1)), axis=1)
      return mesh_points, mesh_tris

class GTriangle(GGeometry):
   '''
   Triangle, basic 2D geometry.
   Variables:
      _vertices: 3 by 2 array.
         Three vertices of the triangle.
         Counterclockwise!
   '''
   def __init__(self, 
                  vertices,
                  dtype = torch.float32,
                  device = 'cpu'):
      '''
      Initialize the line segment.
      Inputs:
         vertices: Torch tensor of size 3 by 2.
            The location of three vertices of the triangle (counterclockwise)
         dtype: Pytorch data type. (Optional, default torch.float32).
            Data type. Can only be torch.float32 or torch.float64.
         device: Pytorch device. (Optional, default 'cpu').
            Pytorch device.
      '''
      super().__init__(2, dtype = dtype, device = device)
      self._vertices = vertices.clone().detach()
      self._vertices_diffs = torch.vstack([
         self._vertices[0,:] - self._vertices[1,:],
         self._vertices[1,:] - self._vertices[2,:],
         self._vertices[2,:] - self._vertices[0,:]
      ])
      self._2center12 = 2 * self._vertices[2,:] + self._vertices_diffs[1,:]
      self._edge_length = torch.norm(self._vertices_diffs, p=2, dim=1)
      self._edge_length /= torch.sum(self._edge_length)
      self._edge_acculength = torch.cumsum(self._edge_length, dim=0)

   def update_dtype(self, dtype):
      '''
      Update the data type.
      Inputs:
         dtype: Pytorch data type.
            Data type. Can only be torch.float32 or torch.float64.
      '''
      super().update_dtype(dtype)
      self._vertices = self._vertices.to(dtype)
      self._vertices_diffs = self._vertices_diffs.to(dtype)
      self._2center12 = self._2center12.to(dtype)
      self._edge_length = self._edge_length.to(dtype)
      self._edge_acculength = self._edge_acculength.to(dtype)
   
   def update_device(self, device):
      '''
      Update the device.
      Inputs:
         device: Pytorch device.
            Pytorch device.
      '''
      super().update_device(device)
      self._vertices = self._vertices.to(device)
      self._vertices_diffs = self._vertices_diffs.to(device)
      self._2center12 = self._2center12.to(device)
      self._edge_length = self._edge_length.to(device)
      self._edge_acculength = self._edge_acculength.to(device)

   def isinside(self, x):
      '''
      Given x which is n by 2, check if in inteval.
      Inputs:
         x: Float number.
            Point to check.
      Outputs:
         isinside: Boolean.
            True if in inteval. Otherwise false.
      '''
      d = torch.vstack([(x[:, 0] - self._vertices[1, 0]) * self._vertices_diffs[0, 1] - (x[:, 1] - self._vertices[1, 1]) * self._vertices_diffs[0, 0],
                     (x[:, 0] - self._vertices[2, 0]) * self._vertices_diffs[1, 1] - (x[:, 1] - self._vertices[2, 1]) * self._vertices_diffs[1, 0],
                     (x[:, 0] - self._vertices[0, 0]) * self._vertices_diffs[2, 1] - (x[:, 1] - self._vertices[0, 1]) * self._vertices_diffs[2, 0]
      ])
      return torch.logical_or(torch.all(d >= 0, dim = 0), torch.all(d <= 0, dim = 0))
   
   def sample_uniform(self, 
                        nx,
                        target_y = None,
                        boundary = False,
                        bc = None,
                        rrange = None,
                        ):
      '''
      Uniformly sample some points inside the geometry, optinally near some target points.
      The output can be either n by d (x or y only) or n by 2d ([x,y] for Greens function).
      Inputs:
         nx: Integer,
            Number of points to sample.
         target_y: Torch tensor, ny by d. (Optional, default None).
            1. None:
                  If target_y is None, output would be a torch tensor of nx by d.
            2. Torch tensor of ny by d: 
                  We sample nx points for each point in target_y, that is a total of n = nx * ny points.
                  The putput would be n by 2d
         boundary: Boolean, optional (Optional, default False).
            If true, we sample on the boundary. Otherwise in the domain.
         bc: Boolean, optional (default None).
            Boundary condition function on $x$ u = bc(x). Takes inpute n by d.
            If None, zero Dirichlet boundary condition will be used
         rrange: Tuple of two numbers. (Optional, default None).
            rrange = (r_min, r_max).
            r_min:
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance smaller than r_min to target_y.
            r_max: Float number, optional (Optional, default -1.0).
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance larger than r_max to target_y.
      Outputs:
         when boundary is False:
            y: n by d tensor or n by 2d tensor.
         when boundary is True:
            y, u_y: n by d tensor or n by 2d tensor together with function value on the boundary
      '''
      if target_y is None:
         # 1. On the entire domain
         if boundary:
            # 1.1 sample on the boundary
            disp = torch.rand(nx, dtype=self._dtype, device = self._device)
            e1 = disp <= self._edge_acculength[0]
            e3 = disp > self._edge_acculength[1]
            e2 = torch.logical_and(~e1, ~e3)
            y = torch.vstack([
               self._vertices[1,:] + torch.einsum('a,b->ab', (disp[e1] / self._edge_length[0]), self._vertices_diffs[0,:]),
               self._vertices[2,:] + torch.einsum('a,b->ab', (disp[e2] - self._edge_acculength[0]) / self._edge_length[1], self._vertices_diffs[1,:]),
               self._vertices[0,:] + torch.einsum('a,b->ab', (disp[e3] - self._edge_acculength[1]) / self._edge_length[2], self._vertices_diffs[2,:])
            ]).detach().requires_grad_(True)
            if bc is not None:
               u_y = bc(y)
            else:
               u_y = torch.zeros(nx, 1, dtype=self._dtype).to(self._device)
            return y, u_y
         else:
            # 1.2 sample uniformly in the triangle
            y = self._vertices[0,:] - \
                  torch.einsum('a,b->ba', self._vertices_diffs[0,:], torch.rand(nx, dtype=self._dtype, device = self._device)) + \
                  torch.einsum('a,b->ba', self._vertices_diffs[2,:], torch.rand(nx, dtype=self._dtype, device = self._device))
            idx_out = ~self.isinside(y)
            y[idx_out,:] =  self._2center12 - y[idx_out,:]
            return y.detach().requires_grad_(True)
      else:
         # 2. Sample near target points
         if rrange is None:
            # 2.1 Sample without limit
            if boundary:
               # 2.1.1 Sample on the boundary
               x_b, _ = self.sample_uniform(nx * target_y.shape[0], boundary=True)
               x = torch.cat([
                  x_b,
                  target_y.repeat_interleave(nx, dim = 0).detach().requires_grad_(True)
               ], dim = 1)
               if bc is not None:
                  u_x = bc(x[:2,:])
               else:
                  u_x = torch.zeros(x.shape[0], 1, dtype=self._dtype).to(self._device)
               return x, u_x
            else:
               # 2.1.2 Sample uniformly in the inteval
               return torch.cat([
                  self.sample_uniform(nx * target_y.shape[0]),
                  target_y.repeat_interleave(nx, dim = 0).detach().requires_grad_(True)
               ], dim = 1)
         else:
            # 2.2 Sample within a distance
            r_min = 0.0 if rrange[0] < 0 else rrange[0]
            r_max = r_min if rrange[1] < r_min else rrange[1]
            x = torch.cat([
               torch.zeros(nx * target_y.shape[0], 2, dtype=self._dtype).to(self._device),
               target_y.repeat_interleave(nx, dim = 0).detach().requires_grad_(True)
            ], dim = 1)
            idx_start = 0
            n_expanded = nx * 4
            for i in range(target_y.shape[0]):
               # resample, reject, repeat
               # WARNING: might got trapped here forever if r_min is too large. We do not check this.
               theta = torch.rand(n_expanded, dtype=self._dtype).to(self._device) * 2 * np.pi
               r = torch.rand(n_expanded, dtype=self._dtype).to(self._device) * (r_max - r_min) + r_min
               x_tmp = torch.stack([target_y[i, 0] + r * torch.cos(theta), target_y[i, 1] + r * torch.sin(theta)], dim=1)
               # search first n points inside the circle
               idx = torch.where(self.isinside(x_tmp))[0]
               # resample if not enough points
               while len(idx) < nx:
                  # TODO: this is a lazy implementation
                  theta = torch.rand(n_expanded, dtype=self._dtype).to(self._device) * 2 * np.pi
                  r = torch.rand(n_expanded, dtype=self._dtype).to(self._device) * (r_max - r_min) + r_min
                  x_tmp = torch.cat([x_tmp, torch.stack([target_y[i, 0] + r * torch.cos(theta), target_y[i, 1] + r * torch.sin(theta)], dim=1)], dim=0)
                  idx = torch.where(self.isinside(x_tmp))[0]
               x[idx_start:idx_start+nx, 0] = x_tmp[idx[:nx], 0]
               x[idx_start:idx_start+nx, 1] = x_tmp[idx[:nx], 1]
               idx_start += nx
            x.detach().requires_grad_(True)
            return x

   def get_mesh(self, min_area = 0.01):
      '''
      Get the mesh of the rectangle.
      Inputs:
         min_area: Float number.
            Minimum area of the mesh.
      Outputs:
         mesh_points: Numpy array, n by 2.
            Points of the mesh.
         mesh_tris: Numpy array, n by 3.
            Triangles of the mesh.
      '''
      def round_trip_connect(start, end):
            return [(i, i + 1) for i in range(start, end)] + [(end, start)]
      points = [(self._vertices[0,0],self._vertices[0,1]),
                (self._vertices[1,0],self._vertices[1,1]),
                (self._vertices[2,0],self._vertices[2,1])]
      facets = round_trip_connect(0, len(points) - 1)
      def needs_refinement(vertices, area):
            return area > min_area
      info = triangle.MeshInfo()
      info.set_points(points)
      info.set_facets(facets)
      mesh = triangle.build(info, refinement_func=needs_refinement)
      mesh_points = np.array(mesh.points)
      mesh_tris = np.array(mesh.elements)
      return mesh_points, mesh_tris
   
   @staticmethod
   def get_dim():
      return 2
   
class GDomain:
   '''
   The Domain for Greens functions, more complex geometry.
   1D Domain is the union of 1D line segments.
   2D Domain is the union of 2D triangles.
   '''
   def __init__(self,
                  dim,
                  dtype = torch.float32,
                  device = 'cpu'):
      '''
      Initialize the GGometry, leave constrain to other specific type to handle.
      Inputs:
         dim: Integer.
            Dimension of the geometry. Typically can be obtained via the staticmethod get_dim()
         dtype: Pytorch data type. (Optional, default torch.float32).
            Data type. Can only be torch.float32 or torch.float64.
         device: Pytorch device. (Optional, default 'cpu').
            Pytorch device.
      '''
      self._d = dim
      self._device = device
      self._dtype = dtype

   def update_dtype(self, dtype):
      '''
      Update the data type.
      Inputs:
         dtype: Pytorch data type.
            Data type. Can only be torch.float32 or torch.float64.
      '''
      self._dtype = dtype

   def update_device(self, device):
      '''
      Update the device.
      Inputs:
         device: Pytorch device.
            Pytorch device.
      '''
      self._device = device

   def isinside(self, x):
      '''
      Given x which is n by 2, check if in inteval.
      Inputs:
         x: Float number.
            Point to check.
      Outputs:
         isinside: Boolean.
            True if in inteval. Otherwise false.
      '''
      return False
   
   def sample_uniform(self, 
                        nx,
                        domain_num = -1,
                        on_quad = False,
                        target_y = None,
                        boundary = False,
                        bc = None,
                        rrange = None,
                        ):
      '''
      Uniformly sample some points inside the geometry, optinally near some target points.
      The output can be either n by d (x or y only) or n by 2d ([x,y] for Greens function).
      Inputs:
         nx: Integer,
            Number of points to sample.
         domain_num: Integer, optional (default -1).
            If >= 0, we sample in the i-th domain.
            Othersize sample in the entire region.
         on_quad: Boolean, optional (default False).
            If true, we sample only on quad points based on their weights.
         target_y: Torch tensor, ny by d. (Optional, default None).
            1. None:
                  If target_y is None, output would be a torch tensor of nx by d.
            2. Torch tensor of ny by d: 
                  We sample nx points for each point in target_y, that is a total of n = nx * ny points.
                  The putput would be n by 2d
         boundary: Boolean, optional (Optional, default False).
            If true, we sample on the boundary. Otherwise in the domain.
         bc: Boolean, optional (default None).
            Boundary condition function on $x$ u = bc(x). Takes inpute n by d.
            If None, zero Dirichlet boundary condition will be used
         rrange: Tuple of two numbers. (Optional, default None).
            rrange = (r_min, r_max).
            r_min:
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance smaller than r_min to target_y.
            r_max: Float number, optional (Optional, default -1.0).
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance larger than r_max to target_y.
      Outputs:
         y: n by d tensor or n by 2d tensor.
      '''
      return None
   
   @staticmethod
   def setup_geometry(mesh_points,
                        mesh_tris,
                        dtype = torch.float64,
                        device = 'cpy',
                        dim = 1):
      '''
      Setup basic geometry for each triangle.
      Inputs:
         mesh_points: Numpy array, n by 2.
            Points of the mesh.
         mesh_tris: Numpy array, n by 3.
            Triangles of the mesh.
         dtype: Pytorch data type. (Optional, default torch.float64).
            Data type. Can only be torch.float32 or torch.float64.
         device: Pytorch device. (Optional, default 'cpu').
            Pytorch device.
      Outputs:
         mesh_geometries: List of GSegment or GTriangle.
      '''
      if dim == 1:
         ntriangles = mesh_tris.shape[0]
         mesh_geometries = []
         for i in range(ntriangles):
            mesh_geometries.append(GSegment(range = (mesh_points[mesh_tris[i,0]],mesh_points[mesh_tris[i,1]]), 
                                                dtype = dtype, 
                                                device=device))
         return mesh_geometries
      elif dim == 2:
         ntriangles = mesh_tris.shape[0]
         mesh_geometries = []
         for i in range(ntriangles):
            mesh_geometries.append(GTriangle(
               vertices = mesh_points[mesh_tris[i, :], :],
               dtype = dtype,
               device = device
            ))
         return mesh_geometries
      else:
         raise ValueError('Wrong dimension.')

   @staticmethod
   def setup_quad(mesh_points,
                  mesh_tris,
                  mesh_areas,
                  mesh_geometries,
                  domain_num,
                  dtype = torch.float64,
                  device = 'cpu',
                  degree = 3,
                  dim = 1,
                  ):
      '''
      Setup points and weights for integral.
      Inputs:
         mesh_points: Numpy array, n by 2.
            Points of the mesh.
         mesh_tris: Numpy array, n by 3.
            Triangles of the mesh.
         mesh_areas: Numpy array, n by 1.
            Areas of the mesh.
         mesh_geometries: List of GSegment or GTriangle.
         domain_num: Numpy array, n by 1.
            Domain number of each triangle.
         dtype: Pytorch data type. (Optional, default torch.float64).
            Data type. Can only be torch.float32 or torch.float64.
         device: Pytorch device. (Optional, default 'cpu').
            Pytorch device.
         degree: Integer. (Optional, default 3).
            Degree of the quadrature.
      Outputs:
         quad_points: N by d tensor.
            Points for quadrature.
         quad_weights: N by 1 tensor.
            Weights for quadrature.
         quad_prob: N by 1 tensor.
            Probability based on weight, used when sampling.
         quad_domain: N by 1 tensor.
            Domain number of each quad point.
      '''
      if dim == 1:
         ntriangles = mesh_tris.shape[0]
         # Gauss Legendre
         if degree == 1:
            points = torch.tensor([0.00000000000000000000], dtype=dtype, device=device)
            weights = torch.tensor([2.00000000000000000000], dtype=dtype, device=device)
         elif degree == 2:
            points = torch.tensor([-0.57735025882720947266,0.57735025882720947266], dtype=dtype, device=device)
            weights = torch.tensor([1.00000000000000000000,1.00000000000000000000], dtype=dtype, device=device)
         elif degree == 3:
            points = torch.tensor([-0.77459669113159179688,0.00000000000000000000,0.77459669113159179688], dtype=dtype, device=device)
            weights = torch.tensor([0.55555558204650878906,0.88888889551162719727,0.55555558204650878906], dtype=dtype, device=device)
         elif degree == 4:
            points = torch.tensor([-0.86113631725311279297,-0.33998104929924011230,0.33998104929924011230,
                                    0.86113631725311279297], dtype=dtype, device=device)
            weights = torch.tensor([0.34785485267639160156,0.65214514732360839844,0.65214514732360839844,
                                    0.34785485267639160156], dtype=dtype, device=device)
         elif degree == 5:
            points = torch.tensor([-0.90617984533309936523,-0.53846931457519531250,0.00000000000000000000,
                                    0.53846931457519531250,0.90617984533309936523], dtype=dtype, device=device)
            weights = torch.tensor([0.23692688345909118652,0.47862866520881652832,0.56888890266418457031,
                                    0.47862866520881652832,0.23692688345909118652], dtype=dtype, device=device)
         else:
            raise ValueError('Degree not implemented.')
         absweights = torch.abs(weights)
         point_unit = weights.shape[0]
         quad_points = torch.zeros(point_unit*ntriangles, dtype = dtype, device = device)
         quad_weights = torch.zeros(point_unit*ntriangles, dtype = dtype, device = device)
         quad_prob = torch.zeros(point_unit*ntriangles, dtype = dtype, device = device)
         quad_domain = torch.zeros(point_unit*ntriangles, dtype = dtype, device = device)
         idx_start = 0
         for i in range(ntriangles):
            pointsi = 0.5 * mesh_areas[i] * points + 0.5 * (mesh_points[mesh_tris[i,0]] + mesh_points[mesh_tris[i,1]])
            quad_points[idx_start:idx_start+point_unit] = pointsi
            quad_weights[idx_start:idx_start+point_unit] = 0.5  * mesh_areas[i] * weights
            quad_prob[idx_start:idx_start+point_unit] = 0.5  * mesh_areas[i] * absweights
            idx_start += point_unit
         for i in range(len(mesh_geometries)):
            quad_domain[mesh_geometries[i].isinside(quad_points)] = domain_num[i]
         quad_prob /= torch.sum(quad_prob)
         return quad_points, quad_weights, quad_prob, quad_domain
      elif dim == 2:
         ntriangles = mesh_tris.shape[0]
         #  
         if degree == 1:
            points = torch.tensor([[0.33333333333333331483],
                                    [0.33333333333333331483],
                                    [0.33333333333333331483]], dtype=dtype, device=device)
            weights = torch.tensor([1.00000000000000000000], dtype=dtype, device=device)
         elif degree == 2:
            points = torch.tensor([[0.16666666666666665741,0.16666666666666665741,0.66666666666666662966],
                                    [0.16666666666666665741,0.66666666666666662966,0.16666666666666665741],
                                    [0.66666666666666662966,0.16666666666666665741,0.16666666666666665741]], dtype=dtype, device=device)
            weights = torch.tensor([0.33333333333333331483,0.33333333333333331483,0.33333333333333331483], dtype=dtype, device=device)
         elif degree == 3:
            points = torch.tensor([[0.33333333333333331483,0.20000000000000001110,0.20000000000000001110,
                                    0.59999999999999997780],
                                    [0.33333333333333331483,0.20000000000000001110,0.59999999999999997780,
                                    0.20000000000000001110],
                                    [0.33333333333333331483,0.59999999999999997780,0.20000000000000001110,
                                    0.20000000000000001110]], dtype=dtype, device=device)
            weights = torch.tensor([-0.56250000000000000000,0.52083333333333337034,0.52083333333333337034,
                                    0.52083333333333337034], dtype=dtype, device=device)
         elif degree == 4:
            points = torch.tensor([[0.44594849091596489021,0.09157621350977071528,0.44594849091596489021,
                                    0.09157621350977071528,0.10810301816807021957,0.81684757298045851392],
                                    [0.44594849091596489021,0.09157621350977071528,0.10810301816807021957,
                                    0.81684757298045851392,0.44594849091596489021,0.09157621350977071528],
                                    [0.10810301816807021957,0.81684757298045851392,0.44594849091596489021,
                                    0.09157621350977071528,0.44594849091596489021,0.09157621350977071528]], dtype=dtype, device=device)
            weights = torch.tensor([0.22338158967801161059,0.10995174365532189853,0.22338158967801161059,
                                    0.10995174365532189853,0.22338158967801161059,0.10995174365532189853], dtype=dtype, device=device)
         elif degree == 5:
            points = torch.tensor([[0.33333333333333331483,0.47014206410511505396,0.10128650732345631513,
                                    0.47014206410511505396,0.10128650732345631513,0.05971587178976989208,
                                    0.79742698535308731422],
                                    [0.33333333333333331483,0.47014206410511505396,0.10128650732345631513,
                                    0.05971587178976989208,0.79742698535308731422,0.47014206410511505396,
                                    0.10128650732345631513],
                                    [0.33333333333333331483,0.05971587178976989208,0.79742698535308731422,
                                    0.47014206410511505396,0.10128650732345631513,0.47014206410511505396,
                                    0.10128650732345631513]], dtype=dtype, device=device)
            weights = torch.tensor([0.22500000000000006106,0.13239415278850619195,0.12593918054482716729,
                                    0.13239415278850619195,0.12593918054482716729,0.13239415278850619195,
                                    0.12593918054482716729], dtype=dtype, device=device)
         else:
            raise ValueError('Degree not implemented.')
         absweights = torch.abs(weights)
         point_unit = weights.shape[0]
         quad_points = torch.zeros((point_unit*ntriangles, 2), dtype = dtype, device = device)
         quad_weights = torch.zeros(point_unit*ntriangles, dtype = dtype, device = device)
         quad_prob = torch.zeros(point_unit*ntriangles, dtype = dtype, device = device)
         quad_domain = torch.zeros(point_unit*ntriangles, dtype = dtype, device = device)
         idx_start = 0
         for i in range(ntriangles):
            pointsi = mesh_points[mesh_tris[i, 0], :] + \
                        torch.einsum('a,b->ab',points[0, :], mesh_points[mesh_tris[i, 1], :] - mesh_points[mesh_tris[i, 0], :]) + \
                        torch.einsum('a,b->ab',points[1, :], mesh_points[mesh_tris[i, 2], :] - mesh_points[mesh_tris[i, 0], :])
            quad_points[idx_start:idx_start+point_unit] = pointsi
            quad_weights[idx_start:idx_start+point_unit] = weights * mesh_areas[i]
            quad_prob[idx_start:idx_start+point_unit] = absweights * mesh_areas[i]
            idx_start += point_unit
         for i in range(len(mesh_geometries)):
            quad_domain[mesh_geometries[i].isinside(quad_points)] = domain_num[i]
         quad_prob /= torch.sum(quad_prob)
         return quad_points, quad_weights, quad_prob, quad_domain
      else:
         raise ValueError('Wrong dimension!')

   @staticmethod
   def naive_dd(mesh_points,
                mesh_tris,
                dlev,
                dim = 1):
      '''
      Naive domain decomposition method, can definitly be improved.
      Inputs:
         mesh_points: Torch tensor, n by d.
            Points of the mesh.
         mesh_tris: Torch tensor, n by 3.
            Triangles of the mesh.
         dlev: Integer.
            Level of the domain decomposition. Total # of subdomain is 2**dlev.
         dim: Integer.
            Dimension of the geometry. Typically can be obtained via the staticmethod get_dim()
      Outputs:
         ndomains: Integer.
            Number of subdomains.
         domain_num: Torch tensor, n by 1.
            Domain number of each point.
         domains: List of Torch tensors.
            List of subdomain indices.
      '''
      if dim == 1 or dim == 2:
         mesh_points = mesh_points.cpu().detach().numpy().astype(np.float64)
         mesh_tris = mesh_tris.cpu().detach().numpy()
         n = mesh_tris.shape[0]
         ndomains = 2**dlev
         if ndomains > 1:
            if n <= ndomains:
               ndomains = n
               domain_num = np.arange(n).astype(np.int64)
               domains = domain_num.reshape(-1, 1)
            else:
               L = np.zeros((n, n)).astype(np.float64)
               for i in range(n):
                  L[:,i] = -np.any(np.isin(mesh_tris, mesh_tris[i, :]), axis=1).astype(np.float64)
                  L[i,i] = 0
               domain_num = np.zeros(n, dtype=np.int64)
               domains = []
               for i in range(dlev):
                  for j in range(2**i):
                     idx = domain_num == j
                     nij = np.sum(idx)
                     Lsub = L[idx,:][:,idx]
                     for k in range(nij):
                        Lsub[k,k] = -np.sum(Lsub[:,k])
                     _, v = eigsh(sp.csr_matrix(Lsub), k=2, which='SM', ncv=min(nij,30), maxiter=300, tol=1e-6)
                     fv = v[:,1] * np.sign(v[0,1])
                     split = np.median(fv)
                     new_domain = np.zeros(nij, dtype=np.int64)
                     new_domain[fv <= split] = j
                     new_domain[fv > split] = j + 2**i
                     domain_num[idx] = new_domain
               for i in range(ndomains):
                  domains.append(np.where(domain_num == i)[0])
         else:
            ndomains = 1
            domain_num = np.zeros(n).astype(np.int64)
            domains = np.arange(n).astype(np.int64).reshape(1, -1)
      else:
         raise ValueError('Wrong dimension!')
      return ndomains, domain_num, domains

   @staticmethod
   def get_dim():
      return 1

class GDSegment(GGeometry):
   '''
   1D Domain, line segment.
   '''
   def __init__(self,
                  rrange,
                  ncell_dd = 32,
                  dlev = 4,
                  ncell_quad = 128,
                  degree = 2,
                  dtype = torch.float32,
                  device = 'cpu'):
      super().__init__( 2, dtype = dtype, device = device)
      self._range = torch.tensor(rrange, dtype = dtype, device = device)
      np_dtype = np.float32 if self._dtype == torch.float32 else np.float64
      # DD
      self._mesh_points = torch.linspace(self._range[0], self._range[1], ncell_dd+1, dtype = dtype, device = device)
      self._mesh_tris = torch.tensor(np.stack((np.arange(ncell_dd), np.arange(1,ncell_dd+1)), axis=1), dtype = torch.int64, device = device)
      self._mesh_areas = self._mesh_points[self._mesh_tris[:,1]] - self._mesh_points[self._mesh_tris[:,0]]
      self._ntriangles = self._mesh_tris.shape[0]
      self._ndomains, self._domain_num, self._domains = GDomain.naive_dd(self._mesh_points, self._mesh_tris, dlev, dim = GDSegment.get_dim())
      self._mesh_geometries = GDomain.setup_geometry(self._mesh_points, self._mesh_tris, dtype = self._dtype, device = self._device, dim = GDSegment.get_dim())
      # Quad
      self._quad_mesh_points = torch.linspace(self._range[0], self._range[1], ncell_quad+1, dtype = dtype, device = device)
      self._quad_mesh_tris = torch.tensor(np.stack((np.arange(ncell_quad), np.arange(1, ncell_quad+1)), axis=1), dtype = torch.int64, device = device)
      self._quad_mesh_areas = self._quad_mesh_points[self._quad_mesh_tris[:,1]] - self._quad_mesh_points[self._quad_mesh_tris[:,0]]
      self._quad_ntriangles = self._quad_mesh_tris.shape[0]
      self._degree = degree
      self._quad_points, self._quad_weights, self._quad_prob, self._quad_domain = \
         GDomain.setup_quad(self._quad_mesh_points,
                            self._quad_mesh_tris,
                            self._quad_mesh_areas,
                            self._mesh_geometries,
                            self._domain_num,
                            dtype = self._dtype,
                            device = self._device,
                            degree = self._degree,
                            dim = GDSegment.get_dim())

   def update_dtype(self, dtype):
      super().update_dtype(dtype)
      self._range = self._range.to(dtype)
      self._mesh_points = self._mesh_points.to(dtype)
      self._mesh_areas = self._mesh_areas.to(dtype)
      self._quad_mesh_points = self._quad_mesh_points.to(dtype)
      self._quad_mesh_areas = self._quad_mesh_areas.to(dtype)
      self._quad_points = self._quad_points.to(dtype)
      self._quad_weights = self._quad_weights.to(dtype)
      self._quad_prob = self._quad_prob.to(dtype)
      for i in range(self._ntriangles):
         self._mesh_geometries[i].update_dtype(dtype)
   
   def update_device(self, device):
      super().update_device(device)
      self._range = self._range.to(device)
      self._mesh_points = self._mesh_points.to(device)
      self._mesh_tris = self._mesh_tris.to(device)
      self._mesh_areas = self._mesh_areas.to(device)
      self._quad_mesh_points = self._quad_mesh_points.to(device)
      self._quad_mesh_tris = self._quad_mesh_tris.to(device)
      self._quad_mesh_areas = self._quad_mesh_areas.to(device)
      self._quad_points = self._quad_points.to(device)
      self._quad_weights = self._quad_weights.to(device)
      self._quad_prob = self._quad_prob.to(device)
      for i in range(self._ntriangles):
         self._mesh_geometries[i].update_device(device)

   def isinside(self, x):
      '''
      Given x which is n by 1, check if in inteval.
      Inputs:
         x: Float number.
            Point to check.
      Outputs:
         isinside: Boolean.
            True if in inteval. Otherwise false.
      '''
      return torch.logical_and(x >= self._range[0], x <= self._range[1])

   def sample_uniform(self, 
                        nx,
                        domain_num = -1,
                        on_quad = False,
                        target_y = None,
                        boundary = False,
                        bc = None,
                        rrange = None,
                        ):
      '''
      Uniformly sample some points inside the geometry, optinally near some target points.
      The output can be either n by d (x or y only) or n by 2d ([x,y] for Greens function).
      Inputs:
         nx: Integer,
            Number of points to sample.
         domain_num: Integer, optional (default -1).
            If >= 0, we sample in the i-th domain.
            Othersize sample in the entire region.
         on_quad: Boolean, optional (default False).
            If true, we sample only on quad points based on their weights.
         target_y: Torch tensor, ny by d. (Optional, default None).
            1. None:
                  If target_y is None, output would be a torch tensor of nx by d.
            2. Torch tensor of ny by d: 
                  We sample nx points for each point in target_y, that is a total of n = nx * ny points.
                  The putput would be n by 2d
         boundary: Boolean, optional (Optional, default False).
            If true, we sample on the boundary. Otherwise in the domain.
         bc: Boolean, optional (default None).
            Boundary condition function on $x$ u = bc(x). Takes inpute n by d.
            If None, zero Dirichlet boundary condition will be used
         rrange: Tuple of two numbers. (Optional, default None).
            rrange = (r_min, r_max).
            r_min:
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance smaller than r_min to target_y.
            r_max: Float number, optional (Optional, default -1.0).
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance larger than r_max to target_y.
      Outputs:
         y: n by d tensor or n by 2d tensor.
      '''
      if target_y is None:
         # 1. On the entire domain
         if domain_num < 0:
            # 1.1 No domain specified
            if boundary:
               # 1.1.1 sample on the boundary
               y = self._range.clone().detach().requires_grad_(True).view(-1, 1)
               if bc is not None:
                  u_y = bc(y)
               else:
                  u_y = torch.zeros(2, 1, dtype=self._dtype).to(self._device)
               return y, u_y
            else:
               # 1.1.2 sample uniformly in the inteval
               if on_quad:
                  # 1.1.2.1 sample only on quad points
                  if nx >= self._quad_points.shape[0]:
                     return self._quad_points.detach().requires_grad_(True).view(-1, 1)
                  else:
                     return self._quad_points[torch.multinomial(self._quad_prob, num_samples=nx, replacement=False)].detach().requires_grad_(True).view(-1, 1)
               else:
                  # 1.1.2.2 sample any points
                  y = torch.rand(nx, 1, dtype=self._dtype, device = self._device) * \
                     (self._range[1]-self._range[0]) + self._range[0]
                  return y.detach().requires_grad_(True)
         else:
            # 1.2 we sample within each triangle based, note that in this case we cannot sample on boundary
            if domain_num >= self._ndomains:
               raise ValueError('Wrong domain number!')
            if on_quad:
               # 1.2.1 sample only on quad points
               idx = self._quad_domain == domain_num
               prob = self._quad_prob[idx]
               prob /= prob.sum()
               if nx >= torch.sum(idx):
                  return self._quad_points[idx].detach().requires_grad_(True).view(-1, 1)
               else:
                  return self._quad_points[idx][torch.multinomial(prob, num_samples=nx, replacement=False)].detach().requires_grad_(True).view(-1, 1)
            else:
               # 1.2.2 sample any points
               idx = self._domains[domain_num]
               ndomaini = idx.shape[0]
               prob = self._mesh_areas[idx]
               prob /= prob.sum()
               domain_list = idx[torch.multinomial(prob, num_samples=nx, replacement=True).cpu().numpy()]
               y = torch.zeros(nx, 1, dtype=self._dtype, device=self._device)
               idx_start = 0
               for i in range(ndomaini):
                  nsamplei = np.sum(domain_list == idx[i])
                  y[idx_start:idx_start+nsamplei] = self._mesh_geometries[idx[i]].sample_uniform(nsamplei)
                  idx_start += nsamplei
               return y.detach().requires_grad_(True)
      else:
         # 2. Sample near target points
         if rrange is None:
            # 2.1 Sample without limit
            if boundary:
               # 2.1.1 Sample on the boundary
               x = torch.cat([
                  self._range.repeat(target_y.shape[0], 1).detach().view(-1, 1),
                  target_y.repeat_interleave(2, dim = 0).detach().requires_grad_(True)
               ], dim = 1)
               if bc is not None:
                  u_x = bc(x[:self.get_dim(),:])
               else:
                  u_x = torch.zeros(x.shape[0], 1, dtype=self._dtype).to(self._device)
               return x, u_x
            else:
               # 2.1.2 Sample uniformly in the inteval
               return torch.cat([
                  self.sample_uniform(nx * target_y.shape[0]),
                  target_y.repeat_interleave(nx, dim = 0).detach().requires_grad_(True)
               ], dim = 1)
         else:
            # 2.2 Sample within a distance
            r_min = 0.0 if rrange[0] < 0 else rrange[0]
            r_max = r_min if rrange[1] < r_min else rrange[1]
            x = torch.cat([
               torch.zeros(nx * target_y.shape[0], 1, dtype=self._dtype).to(self._device),
               target_y.repeat_interleave(nx, dim = 0).detach().requires_grad_(True)
            ], dim = 1)
            idx = 0
            for i in range(target_y.shape[0]):
               left = torch.max(self._range[0], target_y[i, 0] - r_max)
               right = torch.min(self._range[1], target_y[i, 0] + r_max)
               x[idx:idx+nx, 0] = torch.rand(nx, dtype=self._dtype).to(self._device) * (right - left) + left
               idx += nx
            return x.detach().requires_grad_(True)
            
class GDCircle(GDomain):
   '''
   2D Domain, circle.
   '''
   def __init__(self,
                  center, 
                  radius,
                  nbd = 30,
                  min_area_dd = 0.05,
                  dlev = 2,
                  min_area_quad = 0.001,
                  degree = 2,
                  dtype = torch.float32,
                  device = 'cpu'):
      super().__init__( 2, dtype = dtype, device = device)
      self._center = torch.tensor(center, dtype = dtype, device = device)
      self._radius = torch.tensor(radius, dtype = dtype, device = device)
      def round_trip_connect(start, end):
         return [(i, i + 1) for i in range(start, end)] + [(end, start)]
      points = []
      points.extend((center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, nbd, endpoint=False))
      facets = round_trip_connect(0, len(points) - 1)
      def needs_refinement_dd(vertices, area):
         return area > min_area_dd
      def needs_refinement_quad(vertices, area):
         return area > min_area_quad
      info = triangle.MeshInfo()
      info.set_points(points)
      info.set_facets(facets)
      # DD
      mesh = triangle.build(info, refinement_func=needs_refinement_dd)
      self._mesh_points = torch.tensor(mesh.points, dtype = dtype, device = device)
      self._mesh_tris = torch.tensor(mesh.elements, dtype = torch.int64, device = device)
      self._mesh_areas = 0.5 * torch.abs(
         self._mesh_points[self._mesh_tris[:, 0]][:, 0] * (self._mesh_points[self._mesh_tris[:, 1]][:, 1] - self._mesh_points[self._mesh_tris[:, 2]][:, 1]) +
         self._mesh_points[self._mesh_tris[:, 1]][:, 0] * (self._mesh_points[self._mesh_tris[:, 2]][:, 1] - self._mesh_points[self._mesh_tris[:, 0]][:, 1]) +
         self._mesh_points[self._mesh_tris[:, 2]][:, 0] * (self._mesh_points[self._mesh_tris[:, 0]][:, 1] - self._mesh_points[self._mesh_tris[:, 1]][:, 1])
      )
      self._ntriangles = self._mesh_tris.shape[0]
      self._ndomains, self._domain_num, self._domains = GDomain.naive_dd(self._mesh_points, self._mesh_tris, dlev, dim = GDCircle.get_dim())
      self._mesh_geometries = GDomain.setup_geometry(self._mesh_points, self._mesh_tris, dtype = self._dtype, device = self._device, dim = GDCircle.get_dim())
      # Quad
      mesh_quad = triangle.build(info, refinement_func=needs_refinement_quad)
      self._quad_mesh_points = torch.tensor(mesh_quad.points, dtype = dtype, device = device)
      self._quad_mesh_tris = torch.tensor(mesh_quad.elements, dtype = torch.int64, device = device)
      self._quad_mesh_areas = 0.5 * torch.abs(
         self._quad_mesh_points[self._quad_mesh_tris[:, 0]][:, 0] * (self._quad_mesh_points[self._quad_mesh_tris[:, 1]][:, 1] - self._quad_mesh_points[self._quad_mesh_tris[:, 2]][:, 1]) +
         self._quad_mesh_points[self._quad_mesh_tris[:, 1]][:, 0] * (self._quad_mesh_points[self._quad_mesh_tris[:, 2]][:, 1] - self._quad_mesh_points[self._quad_mesh_tris[:, 0]][:, 1]) +
         self._quad_mesh_points[self._quad_mesh_tris[:, 2]][:, 0] * (self._quad_mesh_points[self._quad_mesh_tris[:, 0]][:, 1] - self._quad_mesh_points[self._quad_mesh_tris[:, 1]][:, 1])
      )
      self._quad_ntriangles = self._quad_mesh_tris.shape[0]
      self._degree = degree
      self._quad_points, self._quad_weights, self._quad_prob, self._quad_domain = \
         GDomain.setup_quad(self._quad_mesh_points,
                            self._quad_mesh_tris,
                            self._quad_mesh_areas,
                            self._mesh_geometries,
                            self._domain_num,
                            dtype = self._dtype,
                            device = self._device,
                            degree = self._degree,
                            dim = GDCircle.get_dim())

   def update_dtype(self, dtype):
      super().update_dtype(dtype)
      self._center = self._center.to(dtype)
      self._radius = self._radius.to(dtype)
      self._mesh_points = self._mesh_points.to(dtype)
      self._mesh_areas = self._mesh_areas.to(dtype)
      self._quad_mesh_points = self._quad_mesh_points.to(dtype)
      self._quad_mesh_areas = self._quad_mesh_areas.to(dtype)
      self._quad_points = self._quad_points.to(dtype)
      self._quad_prob = self._quad_prob.to(dtype)
      self._quad_weights = self._quad_weights.to(dtype)
      for i in range(self._ntriangles):
         self._mesh_geometries[i].update_dtype(dtype)
   
   def update_device(self, device):
      super().update_device(device)
      self._center = self._center.to(device)
      self._radius = self._radius.to(device)
      self._mesh_points = self._mesh_points.to(device)
      self._mesh_tris = self._mesh_tris.to(device)
      self._mesh_areas = self._mesh_areas.to(device)
      self._quad_mesh_points = self._quad_mesh_points.to(device)
      self._quad_mesh_tris = self._quad_mesh_tris.to(device)
      self._quad_mesh_areas = self._quad_mesh_areas.to(device)
      self._quad_points = self._quad_points.to(device)
      self._quad_prob = self._quad_prob.to(device)
      self._quad_weights = self._quad_weights.to(device)
      for i in range(self._ntriangles):
         self._mesh_geometries[i].update_device(device)

   def isinside(self, x):
      '''
      Given x which is n by 2, check if the points are inside the circle.
      Inputs:
         x: Float number.
            Point to check.
      Outputs:
         isinside: Boolean.
            True if in inteval. Otherwise false.
      '''
      return torch.norm(x - self._center, dim=1) < self._radius

   def sample_uniform(self, 
                        nx,
                        domain_num = -1,
                        on_quad = False,
                        target_y = None,
                        boundary = False,
                        bc = None,
                        rrange = None,
                        ):
      '''
      Uniformly sample some points inside the geometry, optinally near some target points.
      The output can be either n by d (x or y only) or n by 2d ([x,y] for Greens function).
      Inputs:
         nx: Integer,
            Number of points to sample.
         domain_num: Integer, optional (default -1).
            If >= 0, we sample in the i-th domain.
            Othersize sample in the entire region.
         on_quad: Boolean, optional (default False).
            If true, we sample only on quad points based on their weights.
         target_y: Torch tensor, ny by d. (Optional, default None).
            1. None:
                  If target_y is None, output would be a torch tensor of nx by d.
            2. Torch tensor of ny by d: 
                  We sample nx points for each point in target_y, that is a total of n = nx * ny points.
                  The putput would be n by 2d
         boundary: Boolean, optional (Optional, default False).
            If true, we sample on the boundary. Otherwise in the domain.
         bc: Boolean, optional (default None).
            Boundary condition function on $x$ u = bc(x). Takes inpute n by d.
            If None, zero Dirichlet boundary condition will be used
         rrange: Tuple of two numbers. (Optional, default None).
            rrange = (r_min, r_max).
            r_min:
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance smaller than r_min to target_y.
            r_max: Float number, optional (Optional, default -1.0).
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance larger than r_max to target_y.
      Outputs:
         y: n by d tensor or n by 2d tensor.
      '''
      if target_y is None:
         # 1. On the entire domain
         if domain_num < 0:
            # 1.1 No domain specified
            if boundary:
               # 1.1.1 sample on the boundary
               theta = torch.rand(nx, dtype=self._dtype).to(self._device) * 2 * np.pi
               y = torch.stack([self._center[0] + self._radius * torch.cos(theta), 
                              self._center[1] + self._radius * torch.sin(theta)], dim=1).detach().requires_grad_(True)
               if bc is not None:
                  u_y = bc(y)
               else:
                  u_y = torch.zeros(nx, 1, dtype=self._dtype).to(self._device)
               return y, u_y
            else:
               # 1.1.2 sample uniformly in the inteval
               if on_quad:
                  # 1.1.2.1 sample only on quad points
                  if nx >= self._quad_points.shape[0]:
                     return self._quad_points.detach().requires_grad_(True)
                  else:
                     return self._quad_points[torch.multinomial(self._quad_prob, num_samples=nx, replacement=False), :].detach().requires_grad_(True)
               else:
                  # 1.1.2.2 sample any points
                  theta = torch.rand(nx, dtype=self._dtype).to(self._device) * 2 * np.pi
                  r = torch.sqrt(torch.rand(nx, dtype=self._dtype).to(self._device)) * self._radius
                  return torch.stack([self._center[0] + r * torch.cos(theta), 
                                       self._center[1] + r * torch.sin(theta)], dim=1).detach().requires_grad_(True)
         else:
            # 1.2 we sample within each triangle based, note that in this case we cannot sample on boundary
            if domain_num >= self._ndomains:
               raise ValueError('Wrong domain number!')
            if on_quad:
               # 1.2.1 sample only on quad points
               idx = self._quad_domain == domain_num
               prob = self._quad_prob[idx]
               prob /= prob.sum()
               if nx >= torch.sum(idx):
                  return self._quad_points[idx].detach().requires_grad_(True)
               else:
                  return self._quad_points[idx][torch.multinomial(prob, num_samples=nx, replacement=False), :].detach().requires_grad_(True)
            else:
               # 1.2.2 sample any points
               idx = self._domains[domain_num]
               ndomaini = idx.shape[0]
               prob = self._mesh_areas[idx]
               prob /= prob.sum()
               domain_list = idx[torch.multinomial(prob, num_samples=nx, replacement=False).cpu().numpy()]
               y = torch.zeros(nx, 2, dtype=self._dtype, device=self._device)
               idx_start = 0
               for i in range(ndomaini):
                  nsamplei = np.sum(domain_list == idx[i])
                  y[idx_start:idx_start+nsamplei, :] = self._mesh_geometries[idx[i]].sample_uniform(nsamplei)
                  idx_start += nsamplei
               return y.detach().requires_grad_(True)
      else:
         # 2. Sample near target points
         if rrange is None:
            # 2.1 Sample without limit
            if boundary:
               # 2.1.1 Sample on the boundary
               theta = torch.rand(nx * target_y.shape[0], dtype=self._dtype).to(self._device) * 2 * np.pi
               x = torch.cat([
                  torch.stack([self._center[0] + self._radius * torch.cos(theta), 
                              self._center[1] + self._radius * torch.sin(theta)], dim=1),
                  target_y.repeat_interleave(nx, dim = 0)
               ], dim = 1).detach().requires_grad_(True)
               if bc is not None:
                  u_x = bc(x[:self.get_dim(),:])
               else:
                  u_x = torch.zeros(x.shape[0], 1, dtype=self._dtype).to(self._device)
               return x, u_x
            else:
               # 2.1.2 Sample uniformly in the inteval
               theta = torch.rand(nx * target_y.shape[0], dtype=self._dtype).to(self._device) * 2 * np.pi
               r = torch.sqrt(torch.rand(nx * target_y.shape[0], dtype=self._dtype).to(self._device)) * self._radius
               return torch.cat([
                  torch.stack([self._center[0] + r * torch.cos(theta), 
                              self._center[1] + r * torch.sin(theta)], dim=1),
                  target_y.repeat_interleave(nx, dim = 0)
               ], dim = 1).detach().requires_grad_(True)
         else:
            # 2.2 Sample within a distance
            r_min = 0.0 if rrange[0] < 0 else rrange[0]
            r_max = r_min if rrange[1] < r_min else rrange[1]
            x = torch.cat([
               torch.zeros(nx * target_y.shape[0], 2, dtype=self._dtype).to(self._device),
               target_y.repeat_interleave(nx, dim = 0).detach().requires_grad_(True)
            ], dim = 1)
            idx_start = 0
            n_expanded = nx * 3
            for i in range(target_y.shape[0]):
               # resample, reject, repeat
               # WARNING: might got trapped here forever if r_min is too large. We do not check this.
               theta = torch.rand(n_expanded, dtype=self._dtype).to(self._device) * 2 * np.pi
               r = torch.rand(n_expanded, dtype=self._dtype).to(self._device) * (r_max - r_min) + r_min
               x_tmp = torch.stack([target_y[i, 0] + r * torch.cos(theta), target_y[i, 1] + r * torch.sin(theta)], dim=1)
               # search first n points inside the circle
               idx = torch.where(self.isinside(x_tmp))[0]
               # resample if not enough points
               while len(idx) < nx:
                  # TODO: this is a lazy implementation
                  theta = torch.rand(n_expanded, dtype=self._dtype).to(self._device) * 2 * np.pi
                  r = torch.rand(n_expanded, dtype=self._dtype).to(self._device) * (r_max - r_min) + r_min
                  x_tmp = torch.cat([x_tmp, torch.stack([target_y[i, 0] + r * torch.cos(theta), target_y[i, 1] + r * torch.sin(theta)], dim=1)], dim=0)
                  idx = torch.where(self.isinside(x_tmp))[0]
               x[idx_start:idx_start+nx, 0] = x_tmp[idx[:nx], 0]
               x[idx_start:idx_start+nx, 1] = x_tmp[idx[:nx], 1]
               idx_start += nx
            return x.detach().requires_grad_(True)
   
   def get_plot_tri(self):
      return tri.Triangulation(self._quad_mesh_points[...,0].detach().cpu().numpy(), 
                                 self._quad_mesh_points[...,1].detach().cpu().numpy(), 
                                 triangles=self._quad_mesh_tris.detach().cpu().numpy())

   @staticmethod
   def get_dim():
      return 2

class GDPolygon(GDomain):
   '''
   2D Domain, polygon.
   '''
   def __init__(self,
                  vertices,
                  vertices_hole = None,
                  hole = None,
                  min_area_dd = 0.05,
                  dlev = 2,
                  min_area_quad = 0.001,
                  degree = 2,
                  dtype = torch.float32,
                  device = 'cpu'):
      super().__init__(2, dtype = dtype, device = device)
      self._nvertices = vertices.shape[0]
      self._vertices = torch.tensor(vertices, dtype = dtype, device = device)
      self._vertices_diffs = self._vertices[0,:] - self._vertices[1,:]
      for i in range(1, self._nvertices):
         self._vertices_diffs = torch.vstack([
            self._vertices_diffs,
            self._vertices[i,:] - self._vertices[((i+1) % self._nvertices),:]
         ])
      self._vertices_lengths = torch.norm(self._vertices_diffs, p=2, dim=1)
      def round_trip_connect(start, end):
            return [(i, i + 1) for i in range(start, end)] + [(end, start)]
      points = []
      points.extend((vertices[i,0], vertices[i,1]) for i in range(self._nvertices))
      facets = round_trip_connect(0, len(points) - 1)
      if vertices_hole is not None:
         self._has_hole = True
         self._nvertices_hole = vertices_hole.shape[0]
         self._vertices_hole = torch.tensor(vertices_hole, dtype = dtype, device = device)
         self._vertices_diffs_hole = self._vertices_hole[0,:] - self._vertices_hole[1,:]
         for i in range(1, self._nvertices_hole):
            self._vertices_diffs_hole = torch.vstack([
               self._vertices_diffs_hole,
               self._vertices_hole[i,:] - self._vertices_hole[((i+1) % self._nvertices_hole),:]
            ])
         self._vertices_lengths_hole = torch.norm(self._vertices_diffs_hole, p=2, dim=1)
         points.extend((vertices_hole[i, 0], vertices_hole[i, 1]) for i in range(self._nvertices_hole))
         facets.extend(round_trip_connect(len(points) - vertices_hole.shape[0], len(points) - 1))
      else:
         self._has_hole = False

      def needs_refinement_dd(vertices, area):
            return area > min_area_dd
      def needs_refinement_quad(vertices, area):
            return area > min_area_quad
      def noneed_refinement(vertices, area):
            return False
      info = triangle.MeshInfo()
      info.set_points(points)
      info.set_facets(facets)
      if self._has_hole:
         if hole is None:
            raise ValueError('hole should be specified')
         else:
            info.set_holes(hole)
      # DD
      mesh = triangle.build(info, refinement_func=needs_refinement_dd)
      mesh_naive = triangle.build(info, refinement_func=noneed_refinement)
      self._mesh_points = torch.tensor(mesh.points, dtype = dtype, device = device)
      self._mesh_tris = torch.tensor(mesh.elements, dtype = torch.int64, device = device)
      self._mesh_areas = 0.5 * torch.abs(
         self._mesh_points[self._mesh_tris[:, 0]][:, 0] * (self._mesh_points[self._mesh_tris[:, 1]][:, 1] - self._mesh_points[self._mesh_tris[:, 2]][:, 1]) +
         self._mesh_points[self._mesh_tris[:, 1]][:, 0] * (self._mesh_points[self._mesh_tris[:, 2]][:, 1] - self._mesh_points[self._mesh_tris[:, 0]][:, 1]) +
         self._mesh_points[self._mesh_tris[:, 2]][:, 0] * (self._mesh_points[self._mesh_tris[:, 0]][:, 1] - self._mesh_points[self._mesh_tris[:, 1]][:, 1])
      )
      self._ntriangles = self._mesh_tris.shape[0]
      self._mesh_naive_points = torch.tensor(mesh_naive.points, dtype = dtype, device = device)
      self._mesh_naive_tris = torch.tensor(mesh_naive.elements, dtype = torch.int64, device = device)
      self._mesh_naive_areas = 0.5 * torch.abs(
         self._mesh_naive_points[self._mesh_naive_tris[:, 0]][:, 0] * (self._mesh_naive_points[self._mesh_naive_tris[:, 1]][:, 1] - self._mesh_naive_points[self._mesh_naive_tris[:, 2]][:, 1]) +
         self._mesh_naive_points[self._mesh_naive_tris[:, 1]][:, 0] * (self._mesh_naive_points[self._mesh_naive_tris[:, 2]][:, 1] - self._mesh_naive_points[self._mesh_naive_tris[:, 0]][:, 1]) +
         self._mesh_naive_points[self._mesh_naive_tris[:, 2]][:, 0] * (self._mesh_naive_points[self._mesh_naive_tris[:, 0]][:, 1] - self._mesh_naive_points[self._mesh_naive_tris[:, 1]][:, 1])
      )
      self._ntriangles_naive = self._mesh_naive_tris.shape[0]
      self._mesh_naive_geometries = GDomain.setup_geometry(self._mesh_naive_points, self._mesh_naive_tris, dtype = self._dtype, device = self._device, dim = GDPolygon.get_dim())
      self._ndomains, self._domain_num, self._domains = GDomain.naive_dd(self._mesh_points, self._mesh_tris, dlev, dim = GDPolygon.get_dim())
      self._mesh_geometries = GDomain.setup_geometry(self._mesh_points, self._mesh_tris, dtype = self._dtype, device = self._device, dim = GDPolygon.get_dim())
      # Quad
      mesh_quad = triangle.build(info, refinement_func=needs_refinement_quad)
      self._quad_mesh_points = torch.tensor(mesh_quad.points, dtype = dtype, device = device)
      self._quad_mesh_tris = torch.tensor(mesh_quad.elements, dtype = torch.int64, device = device)
      self._quad_mesh_areas = 0.5 * torch.abs(
         self._quad_mesh_points[self._quad_mesh_tris[:, 0]][:, 0] * (self._quad_mesh_points[self._quad_mesh_tris[:, 1]][:, 1] - self._quad_mesh_points[self._quad_mesh_tris[:, 2]][:, 1]) +
         self._quad_mesh_points[self._quad_mesh_tris[:, 1]][:, 0] * (self._quad_mesh_points[self._quad_mesh_tris[:, 2]][:, 1] - self._quad_mesh_points[self._quad_mesh_tris[:, 0]][:, 1]) +
         self._quad_mesh_points[self._quad_mesh_tris[:, 2]][:, 0] * (self._quad_mesh_points[self._quad_mesh_tris[:, 0]][:, 1] - self._quad_mesh_points[self._quad_mesh_tris[:, 1]][:, 1])
      )
      self._quad_ntriangles = self._quad_mesh_tris.shape[0]
      self._degree = degree
      self._quad_points, self._quad_weights, self._quad_prob, self._quad_domain = \
         GDomain.setup_quad(self._quad_mesh_points,
                            self._quad_mesh_tris,
                            self._quad_mesh_areas,
                            self._mesh_geometries,
                            self._domain_num,
                            dtype = self._dtype,
                            device = self._device,
                            degree = self._degree,
                            dim = GDPolygon.get_dim())

   def update_dtype(self, dtype):
      super().update_dtype(dtype)
      self._vertices = self._vertices.to(dtype)
      self._mesh_points = self._mesh_points.to(dtype)
      self._mesh_areas = self._mesh_areas.to(dtype)
      self._quad_mesh_points = self._quad_mesh_points.to(dtype)
      self._quad_mesh_areas = self._quad_mesh_areas.to(dtype)
      self._quad_points = self._quad_points.to(dtype)
      self._quad_prob = self._quad_prob.to(dtype)
      self._quad_weights = self._quad_weights.to(dtype)
      for i in range(self._ntriangles):
         self._mesh_geometries[i].update_dtype(dtype)
      for i in range(self._ntriangles_naive):
         self._mesh_naive_geometries[i].update_dtype(dtype)
      self._vertices_diffs = self._vertices_diffs.to(dtype)
      self._vertices_lengths = self._vertices_lengths.to(dtype)
      if self._has_hole:
         self._vertices_hole = self._vertices_hole.to(dtype)
         self._vertices_diffs_hole = self._vertices_diffs_hole.to(dtype)
         self._vertices_lengths_hole = self._vertices_lengths_hole.to(dtype)

   def update_device(self, device):
      super().update_device(device)
      self._vertices = self._vertices.to(device)
      self._mesh_points = self._mesh_points.to(device)
      self._mesh_tris = self._mesh_tris.to(device)
      self._mesh_areas = self._mesh_areas.to(device)
      self._quad_mesh_points = self._quad_mesh_points.to(device)
      self._quad_mesh_tris = self._quad_mesh_tris.to(device)
      self._quad_mesh_areas = self._quad_mesh_areas.to(device)
      self._quad_points = self._quad_points.to(device)
      self._quad_prob = self._quad_prob.to(device)
      self._quad_weights = self._quad_weights.to(device)
      for i in range(self._ntriangles):
         self._mesh_geometries[i].update_device(device)
      for i in range(self._ntriangles_naive):
         self._mesh_naive_geometries[i].update_device(device)
      self._vertices_diffs = self._vertices_diffs.to(device)
      self._vertices_lengths = self._vertices_lengths.to(device)
      if self._has_hole:
         self._vertices_hole = self._vertices_hole.to(device)
         self._vertices_diffs_hole = self._vertices_diffs_hole.to(device)
         self._vertices_lengths_hole = self._vertices_lengths_hole.to(device)

   def isinside(self, x):
      '''
      Given x which is n by 2, check if in inteval.
      Inputs:
         x: Float number.
            Point to check.
      Outputs:
         isinside: Boolean.
            True if in inteval. Otherwise false.
      '''
      d = self._mesh_naive_geometries[0].isinside(x)
      for i in range(1, self._ntriangles_naive):
         d = torch.logical_or(d, self._mesh_naive_geometries[i].isinside(x))
      return d
   
   def sample_uniform(self, 
                        nx,
                        domain_num = -1,
                        on_quad = False,
                        target_y = None,
                        boundary = False,
                        bc = None,
                        rrange = None,
                        ):
      '''
      Uniformly sample some points inside the geometry, optinally near some target points.
      The output can be either n by d (x or y only) or n by 2d ([x,y] for Greens function).
      Inputs:
         nx: Integer,
            Number of points to sample.
         domain_num: Integer, optional (default -1).
            If >= 0, we sample in the i-th domain.
            Othersize sample in the entire region.
         on_quad: Boolean, optional (default False).
            If true, we sample only on quad points based on their weights.
         target_y: Torch tensor, ny by d. (Optional, default None).
            1. None:
                  If target_y is None, output would be a torch tensor of nx by d.
            2. Torch tensor of ny by d: 
                  We sample nx points for each point in target_y, that is a total of n = nx * ny points.
                  The putput would be n by 2d
         boundary: Boolean, optional (Optional, default False).
            If true, we sample on the boundary. Otherwise in the domain.
         bc: Boolean, optional (default None).
            Boundary condition function on $x$ u = bc(x). Takes inpute n by d.
            If None, zero Dirichlet boundary condition will be used
         rrange: Tuple of two numbers. (Optional, default None).
            rrange = (r_min, r_max).
            r_min:
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance smaller than r_min to target_y.
            r_max: Float number, optional (Optional, default -1.0).
               1. Negative:
                     Not in use.
               2. Otherwise:
                     No points will be sample with a distance larger than r_max to target_y.
      Outputs:
         y: n by d tensor or n by 2d tensor.
      '''
      if target_y is None:
         # 1. On the entire domain
         if domain_num < 0:
            # 1.1 No domain specified
            if boundary:
               # 1.1.1 sample on the boundary
               if self._has_hole:
                  nboundary = self._nvertices
                  nboundary_hole = self._nvertices_hole
                  prob = torch.cat([self._vertices_lengths, self._vertices_lengths_hole])
                  prob = prob / prob.sum()
                  boundary_list = torch.multinomial(prob, num_samples=nx, replacement=True)
                  y = torch.zeros(nx, 2, dtype=self._dtype, device=self._device)
                  idx_start = 0
                  for i in range(nboundary):
                     nsamplei = torch.sum(boundary_list == i)
                     y[idx_start:idx_start+nsamplei, :] = \
                        self._vertices[(i+1) % nboundary] + \
                        self._vertices_diffs[i] * \
                        torch.rand(nsamplei, 1, dtype=self._dtype, device=self._device)
                     idx_start += nsamplei
                  for i in range(nboundary_hole):
                     nsamplei = torch.sum(boundary_list == i+nboundary)
                     y[idx_start:idx_start+nsamplei, :] = \
                        self._vertices_hole[(i+1) % nboundary_hole] + \
                        self._vertices_diffs_hole[i] * \
                        torch.rand(nsamplei, 1, dtype=self._dtype, device=self._device)
                     idx_start += nsamplei
                  y = y.detach().requires_grad_(True)
                  if bc is not None:
                     u_y = bc(y)
                  else:
                     u_y = torch.zeros(nx, 1, dtype=self._dtype).to(self._device)
                  return y, u_y
               else:
                  nboundary = self._nvertices
                  prob = self._vertices_lengths / self._vertices_lengths.sum()
                  boundary_list = torch.multinomial(prob, num_samples=nx, replacement=True)
                  y = torch.zeros(nx, 2, dtype=self._dtype, device=self._device)
                  idx_start = 0
                  for i in range(nboundary):
                     nsamplei = torch.sum(boundary_list == i)
                     y[idx_start:idx_start+nsamplei, :] = \
                        self._vertices[(i+1) % nboundary] + \
                        self._vertices_diffs[i] * \
                        torch.rand(nsamplei, 1, dtype=self._dtype, device=self._device)
                     idx_start += nsamplei
                  y = y.detach().requires_grad_(True)
                  if bc is not None:
                     u_y = bc(y)
                  else:
                     u_y = torch.zeros(nx, 1, dtype=self._dtype).to(self._device)
                  return y, u_y
            else:
               # 1.1.2 sample uniformly in the inteval
               if on_quad:
                  # 1.1.2.1 sample only on quad points
                  if nx >= self._quad_points.shape[0]:
                     return self._quad_points.detach().requires_grad_(True)
                  else:
                     return self._quad_points[torch.multinomial(self._quad_prob, num_samples=nx, replacement=False), :].detach().requires_grad_(True)
               else:
                  # 1.1.2.2 sample any points
                  ntriangles = self._ntriangles_naive
                  prob = self._mesh_naive_areas / self._mesh_naive_areas.sum()
                  triangle_list = torch.multinomial(prob, num_samples=nx, replacement=True)
                  y = torch.zeros(nx, 2, dtype=self._dtype, device=self._device)
                  idx_start = 0
                  for i in range(ntriangles):
                     nsamplei = torch.sum(triangle_list == i)
                     y[idx_start:idx_start+nsamplei, :] = self._mesh_naive_geometries[i].sample_uniform(nsamplei)
                     idx_start += nsamplei
                  return y.detach().requires_grad_(True)
         else:
            # 1.2 we sample within each triangle based, note that in this case we cannot sample on boundary
            if domain_num >= self._ndomains:
               raise ValueError('Wrong domain number!')
            if on_quad:
               # 1.2.1 sample only on quad points
               idx = self._quad_domain == domain_num
               prob = self._quad_prob[idx]
               prob /= prob.sum()
               if nx >= torch.sum(idx):
                  return self._quad_points[idx].detach().requires_grad_(True)
               else:
                  return self._quad_points[idx][torch.multinomial(prob, num_samples=nx, replacement=False), :].detach().requires_grad_(True)
            else:
               # 1.2.2 sample any points
               idx = self._domains[domain_num]
               ndomaini = idx.shape[0]
               prob = self._mesh_areas[idx]
               prob /= prob.sum()
               domain_list = idx[torch.multinomial(prob, num_samples=nx, replacement=False).cpu().numpy()]
               y = torch.zeros(nx, 2, dtype=self._dtype, device=self._device)
               idx_start = 0
               for i in range(ndomaini):
                  nsamplei = np.sum(domain_list == idx[i])
                  y[idx_start:idx_start+nsamplei, :] = self._mesh_geometries[idx[i]].sample_uniform(nsamplei)
                  idx_start += nsamplei
               return y.detach().requires_grad_(True)
      else:
         # 2. Sample near target points
         if rrange is None:
            # 2.1 Sample without limit
            if boundary:
               # 2.1.1 Sample on the boundary
               if self._has_hole:
                  nboundary = self._nvertices
                  nboundary_hole = self._nvertices_hole
                  prob = torch.cat([self._vertices_lengths, self._vertices_lengths_hole])
                  prob = prob / prob.sum()
                  boundary_list = torch.multinomial(prob, num_samples=nx * target_y.shape[0], replacement=True)
                  y = torch.zeros(nx * target_y.shape[0], 2, dtype=self._dtype, device=self._device)
                  idx_start = 0
                  for i in range(nboundary):
                     nsamplei = torch.sum(boundary_list == i)
                     y[idx_start:idx_start+nsamplei, :] = \
                        self._vertices[(i+1) % nboundary] + \
                        self._vertices_diffs[i] * \
                        torch.rand(nsamplei, 1, dtype=self._dtype, device=self._device)
                     idx_start += nsamplei
                  for i in range(nboundary_hole):
                     nsamplei = torch.sum(boundary_list == i+nboundary)
                     y[idx_start:idx_start+nsamplei, :] = \
                        self._vertices_hole[(i+1) % nboundary_hole] + \
                        self._vertices_diffs_hole[i] * \
                        torch.rand(nsamplei, 1, dtype=self._dtype, device=self._device)
                     idx_start += nsamplei
                  x = torch.cat([
                     y.detach().requires_grad_(True),
                     target_y.repeat_interleave(nx, dim = 0).detach().requires_grad_(True)
                  ], dim = 1)
                  if bc is not None:
                     u_x = bc(x[:self.get_dim(),:])
                  else:
                     u_x = torch.zeros(x.shape[0], 1, dtype=self._dtype).to(self._device)
                  return x, u_x
               else:
                  nboundary = self._nvertices
                  prob = self._vertices_lengths / self._vertices_lengths.sum()
                  boundary_list = torch.multinomial(prob, num_samples=nx * target_y.shape[0], replacement=True)
                  y = torch.zeros(nx * target_y.shape[0], 2, dtype=self._dtype, device=self._device)
                  idx_start = 0
                  for i in range(nboundary):
                     nsamplei = torch.sum(boundary_list == i)
                     y[idx_start:idx_start+nsamplei, :] = \
                        self._vertices[(i+1) % nboundary] + \
                        self._vertices_diffs[i] * \
                        torch.rand(nsamplei, 1, dtype=self._dtype, device=self._device)
                     idx_start += nsamplei
                  x = torch.cat([
                     y.detach().requires_grad_(True),
                     target_y.repeat_interleave(nx, dim = 0).detach().requires_grad_(True)
                  ], dim = 1)
                  if bc is not None:
                     u_x = bc(x[:self.get_dim(),:])
                  else:
                     u_x = torch.zeros(x.shape[0], 1, dtype=self._dtype).to(self._device)
                  return x, u_x
            else:
               # 2.1.2 Sample uniformly in the inteval
               ntriangles = self._ntriangles_naive
               prob = self._mesh_naive_areas / self._mesh_naive_areas.sum()
               triangle_list = torch.multinomial(prob, num_samples=nx * target_y.shape[0], replacement=True)
               y = torch.zeros(nx * target_y.shape[0], 2, dtype=self._dtype, device=self._device)
               idx_start = 0
               for i in range(ntriangles):
                  nsamplei = torch.sum(triangle_list == i)
                  y[idx_start:idx_start+nsamplei, :] = self._mesh_naive_geometries[i].sample_uniform(nsamplei)
                  idx_start += nsamplei
               return torch.cat([
                  y, target_y.repeat_interleave(nx, dim = 0)
               ], dim = 1).detach().requires_grad_(True)
         else:
            # 2.2 Sample within a distance
            r_min = 0.0 if rrange[0] < 0 else rrange[0]
            r_max = r_min if rrange[1] < r_min else rrange[1]
            x = torch.cat([
               torch.zeros(nx * target_y.shape[0], 2, dtype=self._dtype).to(self._device),
               target_y.repeat_interleave(nx, dim = 0).detach().requires_grad_(True)
            ], dim = 1)
            idx_start = 0
            n_expanded = nx * 3
            for i in range(target_y.shape[0]):
               # resample, reject, repeat
               # WARNING: might got trapped here forever if r_min is too large. We do not check this.
               theta = torch.rand(n_expanded, dtype=self._dtype).to(self._device) * 2 * np.pi
               r = torch.rand(n_expanded, dtype=self._dtype).to(self._device) * (r_max - r_min) + r_min
               x_tmp = torch.stack([target_y[i, 0] + r * torch.cos(theta), target_y[i, 1] + r * torch.sin(theta)], dim=1)
               # search first n points inside the circle
               idx = torch.where(self.isinside(x_tmp))[0]
               # resample if not enough points
               while len(idx) < nx:
                  # TODO: this is a lazy implementation
                  theta = torch.rand(n_expanded, dtype=self._dtype).to(self._device) * 2 * np.pi
                  r = torch.rand(n_expanded, dtype=self._dtype).to(self._device) * (r_max - r_min) + r_min
                  x_tmp = torch.cat([x_tmp, torch.stack([target_y[i, 0] + r * torch.cos(theta), target_y[i, 1] + r * torch.sin(theta)], dim=1)], dim=0)
                  idx = torch.where(self.isinside(x_tmp))[0]
               x[idx_start:idx_start+nx, 0] = x_tmp[idx[:nx], 0]
               x[idx_start:idx_start+nx, 1] = x_tmp[idx[:nx], 1]
               idx_start += nx
            return x.detach().requires_grad_(True)
   
   def get_plot_tri(self):
      return tri.Triangulation(self._quad_mesh_points[...,0].detach().cpu().numpy(), 
                                 self._quad_mesh_points[...,1].detach().cpu().numpy(), 
                                 triangles=self._quad_mesh_tris.detach().cpu().numpy())

   @staticmethod
   def get_dim():
      return 2