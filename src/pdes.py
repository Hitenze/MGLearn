import numpy as np

class GPDE():
   '''
   Base class for General PDEs.
   This class provides a template for defining PDEs, including methods for the solution, PDE operator, weak form, and the number of unknowns.
   '''
   def __init__(self):
      super().__init__()
   
   def sol(self, x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
      None: This is a placeholder method and should be overridden by subclasses.
      '''
      return None
   def pde(self, x, u_grad, rhs):
      '''
      x: numpy.ndarray.
         Input array of points.
      u_grad: numpy.ndarray.
         Gradient of the solution.
      rhs: function.
         Right-hand side function.
      
      Returns:
      None: This is a placeholder method and should be overridden by subclasses.
      '''
      return None
   def weak_form(self, U, V):
      '''
      U: fenics.FunctionSpace.
         Function space for the solution.
      V: fenics.FunctionSpace.
         Function space for the test function.
      
      Returns:
      None: This is a placeholder method and should be overridden by subclasses.
      '''
      return None
   @staticmethod
   def nunknown():
      '''
      Returns:
      int: Number of unknowns.
      '''
      return 0

class pde1d_1():
   '''
   1D Laplacian
   '''
   def __init__(self):
      super().__init__()

   def sol(self, x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
         numpy.ndarray: Solution values at the input points.
      '''
      return np.where( x[:,0] < x[:,1] , x[:,0] * (1.0 - x[:,1]), x[:,1] * (1.0 - x[:, 0]))
   
   def pde(self, x, u_grad, rhs):
      '''
      x: numpy.ndarray.
         Input array of points.
      u_grad: numpy.ndarray.
         Gradient of the solution.
      rhs: function.
         Right-hand side function.
      
      Returns:
         numpy.ndarray: PDE values at the input points.
      '''
      return u_grad[2] + rhs(x)
   
   def weak_form(self, U, V):
      '''
      U: fenics.FunctionSpace.
         Function space for the solution.
      V: fenics.FunctionSpace.
         Function space for the test function.
      
      Returns:
         fenics.Form: Weak form of the PDE.
      '''
      try:
         import fenics
      except ImportError:
         print("FEniCS not installed!")
         return None
      u = fenics.TrialFunction(U)
      v = fenics.TestFunction(V)
      return fenics.dot(fenics.grad(u), fenics.grad(v)) * fenics.dx
   
   @staticmethod
   def nunknown():
      '''
      Returns:
         int: Number of unknowns.
      '''
      return 1

class pde1d_2():
   '''
   1D shifted Laplacian -uxx - cu = rhs => rhs + uxx + cu = 0
   '''
   def __init__(self, c_value = 0.0):
      '''
      c_value: float, optional.
         The coefficient c in the shifted Laplacian. Default is 0.0.
      '''
      super().__init__()
      self._c_value = c_value

   def sol(self, x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
         numpy.ndarray or None: Solution values at the input points if c_value is 0.0, otherwise None.
      '''
      return np.where( x[:,0] < x[:,1] , x[:,0] * (1.0 - x[:,1]), x[:,1] * (1.0 - x[:, 0])) \
               if self._c_value == 0.0 else None
   
   def pde(self, x, u_grad, rhs):
      '''
      x: numpy.ndarray.
         Input array of points.
      u_grad: numpy.ndarray.
         Gradient of the solution.
      rhs: function.
         Right-hand side function.
      
      Returns:
         numpy.ndarray: PDE values at the input points.
      '''
      return self._c_value * u_grad[0].squeeze() + u_grad[2] + rhs(x)
   
   def weak_form(self, U, V):
      '''
      U: fenics.FunctionSpace.
         Function space for the solution.
      V: fenics.FunctionSpace.
         Function space for the test function.
      
      Returns:
         fenics.Form or None: Weak form of the PDE. Returns None if FEniCS is not installed.
      '''
      try:
         import fenics
      except ImportError:
         print("FEniCS not installed!")
         return None
      u = fenics.TrialFunction(U)
      v = fenics.TestFunction(V)
      return fenics.dot(fenics.grad(u), fenics.grad(v)) * fenics.dx \
            - self._c_value * u * v * fenics.dx
   
   @staticmethod
   def nunknown():
      '''
      Returns:
         int: Number of unknowns.
      '''
      return 1

class pde1d_3():
   '''
   1D advection-diffusion with variable coefficient c * (1 + x^2)
   '''
   def __init__(self, c_value = 0.0):
      '''
      c_value: float, optional.
         The coefficient c in the advection-diffusion equation. Default is 0.0.
      '''
      super().__init__()
      self._c_value = c_value

   def sol(self, x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
         numpy.ndarray or None: Solution values at the input points if c_value is 0.0, otherwise None.
      '''
      return np.where( x[:,0] < x[:,1] , x[:,0] * (1.0 - x[:,1]), x[:,1] * (1.0 - x[:, 0])) \
               if self._c_value == 0.0 else None
   
   def pde(self, x, u_grad, rhs):
      '''
      x: numpy.ndarray.
         Input array of points.
      u_grad: numpy.ndarray.
         Gradient of the solution.
      rhs: function.
         Right-hand side function.
      
      Returns:
         numpy.ndarray: PDE values at the input points.
      '''
      return self._c_value * (1.0 + x[:,0]**2) * u_grad[0].squeeze() + u_grad[2] + rhs(x)
   
   def weak_form(self, U, V):
      '''
      U: fenics.FunctionSpace.
         Function space for the solution.
      V: fenics.FunctionSpace.
         Function space for the test function.
      
      Returns:
         fenics.Form or None: Weak form of the PDE. Returns None if FEniCS is not installed.
      '''
      try:
         import fenics
      except ImportError:
         print("FEniCS not installed!")
         return None
      u = fenics.TrialFunction(U)
      v = fenics.TestFunction(V)
      return fenics.dot(fenics.grad(u), fenics.grad(v)) * fenics.dx \
            - self._c_value * (1 + fenics.SpatialCoordinate(U.mesh())[0]**2) * u * v * fenics.dx

   @staticmethod
   def nunknown():
      '''
      Returns:
         int: Number of unknowns.
      '''
      return 1
class pde2d_1():
   '''
   2D Laplacian
   '''
   def __init__(self, is_unit_circle = False):
      '''
      is_unit_circle: bool, optional.
         Indicates if the domain is a unit circle. Default is False.
      '''
      super().__init__()
      self._is_unit_circle = is_unit_circle

   @staticmethod
   def nunknown():
      '''
      Returns:
         int: Number of unknowns.
      '''
      return 1
   
   @staticmethod
   def Phi(x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
         numpy.ndarray: Values of the fundamental solution of the Laplacian at the input points.
      '''
      eps_phi = 1e-6 if x.dtype == np.float32 else 1e-12
      return -1/(2*np.pi) * np.log(np.linalg.norm(x, axis=1) + eps_phi)

   @staticmethod
   def DualPoint(x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
         numpy.ndarray: Dual points corresponding to the input points.
      '''
      return x / np.linalg.norm(x, axis=1)[:, np.newaxis]**2

   def sol(self, x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
         numpy.ndarray or None: Solution values at the input points if is_unit_circle is True, otherwise None.
      '''
      return pde2d_1.Phi(x[...,2:] - x[...,:2]) - \
               pde2d_1.Phi(np.linalg.norm(x[...,:2], axis=1)[:, np.newaxis] * \
               (x[...,2:] - pde2d_1.DualPoint(x[...,:2]))) \
               if self._is_unit_circle else None
   
   def pde(self, x, u_grad, rhs):
      '''
      x: numpy.ndarray.
         Input array of points.
      u_grad: numpy.ndarray.
         Gradient of the solution.
      rhs: function.
         Right-hand side function.
      
      Returns:
         numpy.ndarray: PDE values at the input points.
      '''
      return u_grad[3] + u_grad[5] + rhs(x)
   
   def weak_form(self, U, V):
      '''
      U: fenics.FunctionSpace.
         Function space for the solution.
      V: fenics.FunctionSpace.
         Function space for the test function.
      
      Returns:
         fenics.Form or None: Weak form of the PDE. Returns None if FEniCS is not installed.
      '''
      try:
         import fenics
      except ImportError:
         print("FEniCS not installed!")
         return None
      u = fenics.TrialFunction(U)
      v = fenics.TestFunction(V)
      return fenics.dot(fenics.grad(u), fenics.grad(v)) * fenics.dx

class pde2d_2():
   '''
   2D shifted Laplacian - Delta u - cu = f
   '''
   def __init__(self, c_value = 0.0, is_unit_circle = False):
      '''
      c_value: float, optional.
         The coefficient c in the shifted Laplacian. Default is 0.0.
      is_unit_circle: bool, optional.
         Indicates if the domain is a unit circle. Default is False.
      '''
      super().__init__()
      self._c_value = c_value
      self._is_unit_circle = is_unit_circle

   @staticmethod
   def nunknown():
      '''
      Returns:
         int: Number of unknowns.
      '''
      return 1
   
   @staticmethod
   def Phi(x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
         numpy.ndarray: Values of the fundamental solution of the Laplacian at the input points.
      '''
      eps_phi = 1e-6 if x.dtype == np.float32 else 1e-12
      return -1/(2*np.pi) * np.log(np.linalg.norm(x, axis=1) + eps_phi)

   @staticmethod
   def DualPoint(x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
         numpy.ndarray: Dual points corresponding to the input points.
      '''
      return x / np.linalg.norm(x, axis=1)[:, np.newaxis]**2

   def sol(self, x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
         numpy.ndarray or None: Solution values at the input points if is_unit_circle is True and c_value is 0.0, otherwise None.
      '''
      return pde2d_1.Phi(x[...,2:] - x[...,:2]) - \
               pde2d_1.Phi(np.linalg.norm(x[...,:2], axis=1)[:, np.newaxis] * \
               (x[...,2:] - pde2d_1.DualPoint(x[...,:2]))) \
               if self._is_unit_circle and self._c_value == 0.0 else None
   
   def pde(self, x, u_grad, rhs):
      '''
      x: numpy.ndarray.
         Input array of points.
      u_grad: numpy.ndarray.
         Gradient of the solution.
      rhs: function.
         Right-hand side function.
      
      Returns:
         numpy.ndarray: PDE values at the input points.
      '''
      return self._c_value * u_grad[0].squeeze() + u_grad[3] + u_grad[5] + rhs(x)
   
   def weak_form(self, U, V):
      '''
      U: fenics.FunctionSpace.
         Function space for the solution.
      V: fenics.FunctionSpace.
         Function space for the test function.
      
      Returns:
         fenics.Form or None: Weak form of the PDE. Returns None if FEniCS is not installed.
      '''
      try:
         import fenics
      except ImportError:
         print("FEniCS not installed!")
         return None
      u = fenics.TrialFunction(U)
      v = fenics.TestFunction(V)
      return fenics.dot(fenics.grad(u), fenics.grad(v)) * fenics.dx \
            - self._c_value * u * v * fenics.dx

class pde2d_3():
   '''
   2D shifted Laplacian - Delta u - c(1+x[0]^2+x[1]^2)u = f
   '''
   def __init__(self, c_value = 0.0, is_unit_circle = False):
      '''
      c_value: float, optional.
         The coefficient c in the shifted Laplacian. Default is 0.0.
      is_unit_circle: bool, optional.
         Indicates if the domain is a unit circle. Default is False.
      '''
      super().__init__()
      self._c_value = c_value
      self._is_unit_circle = is_unit_circle

   @staticmethod
   def nunknown():
      '''
      Returns:
         int: Number of unknowns.
      '''
      return 1
   
   @staticmethod
   def Phi(x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
         numpy.ndarray: Values of the fundamental solution of the Laplacian at the input points.
      '''
      eps_phi = 1e-6 if x.dtype == np.float32 else 1e-12
      return -1/(2*np.pi) * np.log(np.linalg.norm(x, axis=1) + eps_phi)

   @staticmethod
   def DualPoint(x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
         numpy.ndarray: Dual points corresponding to the input points.
      '''
      return x / np.linalg.norm(x, axis=1)[:, np.newaxis]**2

   def sol(self, x):
      '''
      x: numpy.ndarray.
         Input array of points.
      
      Returns:
         numpy.ndarray or None: Solution values at the input points if is_unit_circle is True and c_value is 0.0, otherwise None.
      '''
      return pde2d_1.Phi(x[...,2:] - x[...,:2]) - \
               pde2d_1.Phi(np.linalg.norm(x[...,:2], axis=1)[:, np.newaxis] * \
               (x[...,2:] - pde2d_1.DualPoint(x[...,:2]))) \
               if self._is_unit_circle and self._c_value == 0.0 else None
   
   def pde(self, x, u_grad, rhs):
      '''
      x: numpy.ndarray.
         Input array of points.
      u_grad: numpy.ndarray.
         Gradient of the solution.
      rhs: function.
         Right-hand side function.
      
      Returns:
         numpy.ndarray: PDE values at the input points.
      '''
      return self._c_value * (1.0 + x[:,0]**2 + x[:,1]**2) * u_grad[0].squeeze() + u_grad[3] + u_grad[5] + rhs(x)
   
   def weak_form(self, U, V):
      '''
      U: fenics.FunctionSpace.
         Function space for the solution.
      V: fenics.FunctionSpace.
         Function space for the test function.
      
      Returns:
         fenics.Form or None: Weak form of the PDE. Returns None if FEniCS is not installed.
      '''
      try:
         import fenics
      except ImportError:
         print("FEniCS not installed!")
         return None
      u = fenics.TrialFunction(U)
      v = fenics.TestFunction(V)
      return fenics.dot(fenics.grad(u), fenics.grad(v)) * fenics.dx \
            - self._c_value * (1 + fenics.SpatialCoordinate(U.mesh())[0]**2 + fenics.SpatialCoordinate(U.mesh())[1]**2) * u * v * fenics.dx
