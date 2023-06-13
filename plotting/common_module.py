import numpy as np
import tensorflow as tf


# ===================================================================================================


class AdaptiveActivation(tf.keras.layers.Layer):
    
    """
      Define activation function with adaptive trainable slope.
    """
    def __init__(self,**kwargs):
        super(AdaptiveActivation, self).__init__()
        self.a = tf.Variable(0.5, dtype=tf.float64, trainable=True)
        self.n = tf.constant(2.0, dtype=tf.float64)
        
    def call(self, x):
            return tf.keras.activations.tanh(self.a*self.n*x)


# ===================================================================================================


def get_grid(parameters):
    '''
      Create grid in spherical, compactified and cylindrical coordinates.
      The key 'coordinates' of the parameters dictionary determines which coordinates are equally spaced.
      'n_1', 'n_2' determine the resolution.
    '''
    # Resolution
    n_1 = parameters['n_1']
    n_2 = parameters['n_2']


    if parameters['coordinates'] == 'spherical':
        
        # Spherical grid
        rmin = 1
        rmax = parameters['extent'] * parameters['R_lc']
        thetamin = 0
        thetamax = np.pi

        r, theta = np.meshgrid(np.linspace(rmin, rmax, n_1), np.linspace(thetamin, thetamax, n_2), indexing='ij')
        q, mu = 1 / r, np.cos (theta)
        rho, z = r*np.sin(theta), r*np.cos(theta)

    elif parameters['coordinates'] == 'compactified':
        
        # Compactified grid
        qmin = 0
        qmax = 1
        mumin = -1
        mumax = 1

        q, mu = np.meshgrid(np.linspace(qmin, qmax, n_1), np.linspace(mumin, mumax, n_2), indexing='ij')
        r, theta = 1 / q, np.arccos(mu)
        rho, z = np.sqrt(1 - mu ** 2) / q, mu / q

    elif parameters['coordinates'] == 'cylindrical':
        
        # Cylindrical grid
        rhomin = 0
        rhomax = parameters['extent'] * parameters['R_lc']
        zmin = - parameters['extent'] * parameters['R_lc']
        zmax = parameters['extent'] * parameters['R_lc']

        rho, z = np.meshgrid(np.linspace(rhomin, rhomax, n_1), np.linspace(zmin, zmax, n_2), indexing='ij')
        r, theta = np.sqrt(rho ** 2 + z ** 2), np.arctan2(rho, z)
        q, mu = 1 / np.sqrt(rho ** 2 + z ** 2), z / np.sqrt(rho ** 2 + z ** 2)

    else:

        raise ValueError ("coordinates must be one of 'spherical', 'compactified', 'cylindrical'.\n\
                            Current value --> {}".format(parameters['coordinates']))
    
    return r, theta, q, mu, rho, z


# ===================================================================================================


def generate_test (x1, x2, parameters):

   """
      Creates th input tensor of the PINN for a test in some grid. 'x1' and 'x2' form a 2D meshgrid and 'coordinates' specifies
      the type of the coordinates (possible choises are 'spherical', 'compactified', 'cylindrical', 'cartesian').
   """
   
   X = np.zeros([x1.size, 2])


   if parameters['coordinates'] == 'spherical': 
   
      q = 1 / x1
      mu = np.cos(x2)

   elif parameters['coordinates'] == 'compactified':

      q = x1
      mu = x2

   elif parameters['coordinates'] == 'cylindrical':

      q = 1 / np.sqrt (x1 ** 2 + x2 ** 2)
      mu = 1 / np.sqrt (x1 ** 2 / x2 ** 2 + 1)

   else:

      raise ValueError ("coordinates must be one of 'spherical', 'compactified', 'cylindrical'.\n\
                            Current value --> {}".format(parameters['coordinates']))

   
   X[:,0] = q.flatten()
   X[:,1] = mu.flatten()

   return tf.convert_to_tensor(X)


# ===================================================================================================


def f_b (X, P_c, r_c):

   q = X [:,0]
   mu = X [:,1]

   # fb = (1 - mu ** 2) * (P_c + q * ((1 - P_c) * tf.nn.relu(1 - 1 / (q * r_c)) ** 2 / (1 - 1 / r_c) ** 2))
   fb = (1 - mu ** 2) * (P_c + q * ((1 - P_c) * tf.nn.relu(1 - 1 / (q * r_c)) ** 3 / (1 - 1 / r_c) ** 3))

   return fb


# ===================================================================================================


def h_b (X, r_c):

   q = X [:,0]
   mu = X [:,1]

   # hb = (1 - mu ** 2) * (1 - q) * tf.sqrt(tf.nn.relu (1 - 1 / (q * r_c)) ** 2 + mu ** 2)
   hb = (1 - mu ** 2) * (1 - q) * tf.sqrt(tf.nn.relu (1 - 1 / (q * r_c)) ** 3 + mu ** 2)

   return hb


# ===================================================================================================


def P_parametrization (X, N_p, parameters):
   
   """
      Calculate P and P_c using the output of the network and the parametrisation
   """
   
   Np = N_p (X)
   r_c = parameters ['r_c']
   
   if parameters['Pc_mode'] == 'free':
      
      P_c = tf.reduce_mean (Np[:,1])
   
   elif parameters['Pc_mode'] == 'fixed':
     
      P_c = parameters['Pc_fixed']
   
   else:
      
      raise ValueError ("Pc_mode must be one of 'free', 'fixed'.\n\
                        Current value --> {}".parameters['Pc_mode'])

   P = f_b (X, P_c, r_c) + h_b (X, r_c) * Np[:,0]

   return P, P_c


# ===================================================================================================


def T_parametrization (P, P_c, N_t, parameters):
   
   # Reshape N_t output from (n_points,1) to (n_points, ) for proper broadcasting
   N_t = tf.reshape(N_t (P), N_t (P).shape[0])
   R_lc = parameters['R_lc']
   sep_width = parameters['sep_width']

   # T = (-2 * P / R_lc + P ** 2 * N_t) * tf.experimental.numpy.heaviside(P_c ** 2 - P ** 2, 0.5)
   T = P * N_t * tf.math.erfc((P - (P_c + 2 * sep_width)) / sep_width) / 2
    
   return T


# ===================================================================================================


def get_scalar_functions_automatic (X, N_p, N_t, parameters):
    
   # Graph for 1st derivatives of P
   with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt1:
      gt1.watch(X)

      # Graph for 2nd derivatives of P
      with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt2:
         gt2.watch(X)

         # Calculate P, Pc from the parametrisation
         P, Pc = P_parametrization (X, N_p, parameters)

      # Calculate 1st derivatives
      Pgrad = gt2.gradient(P, X)
      dP_dq = Pgrad[:,0]
      dP_dmu = Pgrad[:,1]

   # Calculate 2nd derivatives
   d2P_dq2 = gt1.gradient(dP_dq, X)[:,0]
   d2P_dmu2 = gt1.gradient(dP_dmu, X)[:,1]

   # Calculate mixed derivatives
   d2P_dqdmu = gt1.gradient(dP_dq, X)[:,1]
   d2P_dmudq = gt1.gradient(dP_dmu, X)[:,0]

   # Graph for derivative of T w.r.t.P
   with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt3:
      gt3.watch(P)

      # Calculate T from the parametrisation
      T = T_parametrization (P, Pc, N_t, parameters)
   
   # Calculate dT_dP
   T_prime = gt3.gradient(T, P)

   # Transform derivatives in the coordinate system
   q = X[:,0]
   mu = X[:,1]
   
   if parameters['coordinates'] == 'spherical':

      dP_dx1 = - q ** 2 * dP_dq
      dP_dx2 = - tf.sqrt(1 - mu **2) * dP_dmu

      d2P_dx12 = 2 * q ** 3 * dP_dq + q ** 4 * d2P_dq2
      d2P_dx22 = - mu * dP_dmu + (1 - mu ** 2) * d2P_dmu2

      d2P_dx1dx2 = q ** 2 * tf.sqrt(1 - mu ** 2) * d2P_dqdmu
      d2P_dx2dx1 = q ** 2 * tf.sqrt(1 - mu ** 2) * d2P_dmudq

   elif parameters['coordinates'] == 'compactified':

      dP_dx1 = dP_dq
      dP_dx2 = dP_dmu
      
      d2P_dx12 = d2P_dq2
      d2P_dx22 = d2P_dmu2

      d2P_dx1x2 = d2P_dqdmu
      d2P_dx2x1 = d2P_dqdmu

   elif parameters['coordinates'] == 'cylindrical':

      raise ValueError ("Tensorflow derivatives via automatic differentiation are not compatible with cylindrical coordinates.\n\
                        Use finite differences or different coordinates.")

   # Tranform to numpy meshgrid arrays
   P = P.numpy().reshape([parameters['n_1'], parameters['n_2']])
   T = T.numpy().reshape([parameters['n_1'], parameters['n_2']])
   T_prime = T_prime.numpy().reshape([parameters['n_1'], parameters['n_2']])
   Pc = float(Pc)
   dP_dx1 = dP_dx1.numpy().reshape([parameters['n_1'], parameters['n_2']])
   dP_dx2 = dP_dx2.numpy().reshape([parameters['n_1'], parameters['n_2']])
   d2P_dx12 = d2P_dx12.numpy().reshape([parameters['n_1'], parameters['n_2']])
   d2P_dx22 = d2P_dx22.numpy().reshape([parameters['n_1'], parameters['n_2']])
   d2P_dx1dx2 = d2P_dx1dx2.numpy().reshape([parameters['n_1'], parameters['n_2']])
   d2P_dx2dx1 = d2P_dx2dx1.numpy().reshape([parameters['n_1'], parameters['n_2']])

   return P, Pc, T, T_prime, dP_dx1, dP_dx2, d2P_dx12, d2P_dx22, d2P_dx1dx2, d2P_dx2dx1


# ===================================================================================================


def get_scalar_functions_fd (X, N_p, N_t, parameters):
   
   # Get P, Pc and T from the parametrisation
   P, Pc = P_parametrization (X, N_p, parameters)
   T = T_parametrization (P, Pc, N_t, parameters)

   # Transform to numpy meshgrid arrays
   P = P.numpy().reshape([parameters['n_1'], parameters['n_2']])
   T = T.numpy().reshape([parameters['n_1'], parameters['n_2']])
   T_prime = np.gradient(T.flatten(), P.flatten(), edge_order=2).reshape(T.shape)
   Pc = float(Pc)

   # Unpack coordinates to numpy meshgrid arrays
   if parameters['coordinates'] == 'spherical':

      x1 = 1 / X[:,0]
      x2 = tf.acos(X[:,1])

   elif parameters['coordinates'] == 'compactified':

      x1 = X[:,0]
      x2 = X[:,1]

   elif parameters['coordinates'] == 'cylindrical':

      x1 = tf.sqrt(1 - X[:,1] ** 2) / X[:,0]
      x2 = X[:,1] / X[:,0]

   # Initialise arrays
   dP_dx1, dP_dx2, d2P_dx12, d2P_dx22, d2P_dx1dx2, d2P_dx2dx1 = [np.zeros_like(P) for _ in range(6)]
   
   # Reshape
   x1 = x1.numpy().reshape([parameters['n_1'], parameters['n_2']])
   x2 = x2.numpy().reshape([parameters['n_1'], parameters['n_2']])

   # Step size
   dx1 = x1[1,1] - x1[0,1]
   dx2 = x2[1,1] - x2[1,0]

   # First derivative w.r.t. first coordinate x1
   dP_dx1[0,:] = (-3 * P[0,:] + 4 * P[1,:] - P[2,:]) / (2 * dx1)
   dP_dx1[1:-1,:] = (P[2:,:] - P[0:-2,:]) / (2 * dx1)
   dP_dx1[-1,:] = (3 * P[-1,:] - 4 * P[-2,:] + P[-3,:]) / (2 * dx1)

   # First derivative w.r.t. second coordinate x2
   dP_dx2[:,0] = (-3 * P[:,0] + 4 * P[:,1] - P[:,2]) / (2 * dx2)
   dP_dx2[:,1:-1] = (P[:,2:] - P[0:,:-2]) / (2 * dx2)
   dP_dx2[:,-1] = (3 * P[:,-1] - 4 * P[:,-2] + P[:,-3]) / (2 * dx2)

   # Second derivative w.r.t. first coordinate x1
   d2P_dx12[0,:] = (2 * P[0,:] - 5 * P[1,:] + 4 * P[2,:] - P[3,:]) / dx1 ** 2
   d2P_dx12[1:-1,:] = (P[0:-2,:] -2 * P[1:-1,:] + P[2:,:]) / dx1 ** 2
   d2P_dx12[-1,:] = (2 * P[-1,:] - 5 * P[-2,:] + 4 * P[-3,:] - P[-4,:]) / dx1 ** 2
   
   # Second derivative w.r.t. second coordinate x2
   d2P_dx22[:,0] = (2 * P[:,0] - 5 * P[:,1] + 4 * P[:,2] - P[:,3]) / dx2 ** 2
   d2P_dx22[:,1:-1] = (P[:,0:-2] -2 * P[:,1:-1] + P[:,2:]) / dx2 ** 2
   d2P_dx22[:,-1] = (2 * P[:,-1] - 5 * P[:,-2] + 4 * P[:,-3] - P[:,-4]) / dx2 ** 2

   # Second mixed derivative
   
   d2P_dx1dx2[1:-1, 1:-1] = (P[2:,2:] - P[0:-2,2:] - P[2:,0:-2] + P[0:-2,0:-2]) / (4 * dx1 * dx2)
   # d2P_dx2dx1[1:-1, 1:-1] = (P[2:,2:] - P[0:-2,2:] - P[2:,0:-2] + P[0:-2,0:-2]) / (4 * dx1 * dx2)

   return P, Pc, T, T_prime, dP_dx1, dP_dx2, d2P_dx12, d2P_dx22, d2P_dx1dx2, d2P_dx2dx1


# ===================================================================================================


def get_scalar_functions(X, N_p, N_t, parameters):
   
   if parameters['deriv_mode'] == 'automatic':

      P, Pc, T, T_prime, dP_dx1, dP_dx2, d2P_dx12, d2P_dx22, d2P_dx1dx2, d2P_dx2dx1 = get_scalar_functions_automatic (X, N_p, N_t, parameters)
      
   elif parameters['deriv_mode'] == 'fd':

      P, Pc, T, T_prime, dP_dx1, dP_dx2, d2P_dx12, d2P_dx22, d2P_dx1dx2, d2P_dx2dx1 = get_scalar_functions_fd (X, N_p, N_t, parameters)

   return P, Pc, T, T_prime, dP_dx1, dP_dx2, d2P_dx12, d2P_dx22, d2P_dx1dx2, d2P_dx2dx1


# ===================================================================================================


def magnetic_field (x1, x2, dP_dx1, dP_dx2, T, parameters):
    
   """
      Calculate the magnetic field from the stream functions 'P' and 'T' in the 'x1', 'x2' grid specified by 'coordinates'.
   """

   if parameters['coordinates'] == 'spherical':
      
      B1 = 1 / (x1 ** 2 * np.sin(x2)) * dP_dx2
      B2 = -1 / (x1 * np.sin(x2)) * dP_dx1
      B3 = 1 / (x1 * np.sin(x2)) * T

   elif parameters['coordinates'] == 'compactified':

      B1 = - x1 ** 2 * dP_dx2
      B2 = - x1 ** 3 / np.sqrt (1 - x2 ** 2) * dP_dx1
      B3 = x1 / np.sqrt (1 - x2 ** 2) * T

   elif parameters['coordinates'] == 'cylindrical':

      B1 = - 1 / x1 * dP_dx2
      B2 = 1 / x1 * dP_dx1
      B3 = 1 / x1 * T

   else:

      raise ValueError ("coordinates must be one of 'spherical', 'compactified', 'cylindrical'.\n\
                        Current value --> {}".format(parameters['coordinates']))
   
   return B1, B2, B3


# ===================================================================================================


def electric_field (x1, x2, dP_dx1, dP_dx2, parameters):
    
   """
      Calculate the electric field from the stream function 'P' in the 'x1', 'x2' grid specified by 'coordinates'.
   """

   if parameters['coordinates'] == 'spherical':
      
      E1 = - 1 / parameters['R_lc'] * dP_dx1
      E2 = - 1 / (parameters['R_lc'] * x1) * dP_dx2
      E3 = np.zeros_like (E1)

   elif parameters['coordinates'] == 'compactified':

      E1 = 1 / parameters['R_lc'] * x1 ** 2 * dP_dx1
      E2 = 1 / parameters['R_lc'] * x1 * np.sqrt (1 - x2 ** 2) * dP_dx2
      E3 = np.zeros_like (E1)

   elif parameters['coordinates'] == 'cylindrical':

      E1 = - 1 / parameters['R_lc'] * dP_dx1
      E2 = - 1 / parameters['R_lc'] * dP_dx2
      E3 = np.zeros_like (E1)
      
   else:

      raise ValueError ("coordinates must be one of 'spherical', 'compactified', 'cylindrical'.\n\
                        Current value --> {}".format(parameters['coordinates']))
   
   return E1, E2, E3


# ===================================================================================================


def dot (u1, u2, u3, v1, v2, v3):
    
   '''
      Calculate the dot product between two vectors u and v
   '''

   return u1 * v1 + u2 * v2 + u3 * v3


# ===================================================================================================


def magnitude (u1, u2, u3):
    
    '''
      Calculate the magnitude of a vector u.
    '''

    return np.sqrt(u1 ** 2 + u2 ** 2 + u3 ** 2)


# ===================================================================================================


def div_B (x1, x2, d2P_dx1dx2, d2P_dx2dx1, parameters):

   if parameters['coordinates'] == 'spherical':

      divB = 1 / (x1 ** 2 * np.sin (x2)) * (d2P_dx1dx2 - d2P_dx2dx1)

   elif parameters['coordinates'] == 'compactified':

      divB = x1 ** 4 * (d2P_dx1dx2 - d2P_dx2dx1)

   return divB


# ===================================================================================================


def div_E (x1, x2, dP_dx1, dP_dx2, d2P_dx12, d2P_dx22, parameters):

   if parameters['coordinates'] == 'spherical':

      divE = - 1 / parameters['R_lc'] * (2 / x1 * dP_dx1 + d2P_dx12 + 1 / x1 ** 2 * np.cos (x2) / np.sin (x2) * dP_dx2 + 1 / x1 ** 2 * d2P_dx22)

   elif parameters['coordinates'] == 'compactified':

      divE = - 1 / parameters['R_lc'] * (x1 ** 4 * d2P_dx12 - 2 * x1 ** 2 * x2 * dP_dx2 + x1 ** 2 (1 - x2 ** 2) * d2P_dx22)

   return divE


# ===================================================================================================


def get_fields (x1, x2, P, T, dP_dx1, dP_dx2, d2P_dx12, d2P_dx22, d2P_dx1dx2, d2P_dx2dx1, parameters):
   
   # Magnetic and electric field
   B_1, B_2, B_3 = magnetic_field (x1, x2, dP_dx1, dP_dx2, T, parameters)
   E_1, E_2, E_3 = electric_field (x1, x2, dP_dx1, dP_dx2, parameters)

   # Magnitude
   B_mag = magnitude (B_1, B_2, B_3)
   E_mag = magnitude (E_1, E_2, E_3)
   B_pol = magnitude (B_1, B_2, 0)

   # Scalar product
   E_dot_B = dot (B_1, B_2, B_3, E_1, E_2, E_3)

   # Divergence
   divB = div_B(x1, x2, d2P_dx1dx2, d2P_dx2dx1, parameters)
   divE = div_E(x1, x2, dP_dx1, dP_dx2, d2P_dx12, d2P_dx22, parameters)

   return B_1, B_2, B_3, E_1, E_2, E_3, B_mag, E_mag, B_pol, E_dot_B, divB, divE


# ===================================================================================================


def get_pinn (X, P, Pc, N_p, N_t, parameters):
   
   Np = N_p (X)[:,0].numpy().reshape([parameters['n_1'], parameters['n_2']])
   Nt = N_t (tf.convert_to_tensor(P.ravel())).numpy().reshape([parameters['n_1'], parameters['n_2']])

   fb = f_b (X, Pc, parameters['r_c']).numpy().reshape([parameters['n_1'], parameters['n_2']])
   hb = h_b (X, parameters['r_c']).numpy().reshape([parameters['n_1'], parameters['n_2']])

   return Np, Nt, fb, hb


# ===================================================================================================


def pulsar_equation (x1, x2, dP_dx1, dP_dx2, d2P_dx12, d2P_dx22, T, T_prime, parameters):
   
   # Calculate Pulsar equation depending on the coordinates

   if parameters['coordinates'] == 'spherical':
      
      beta = x1 * np.sin(x2) / parameters['R_lc']

      pulsar_eq = (1 - beta ** 2) * (d2P_dx12 + 1 / x1 ** 2 * d2P_dx22) - 2 / x1 * beta ** 2 * dP_dx1 \
                  - 1 / x1 ** 2 * np.cos (x2) / np.sin(x2) * (1 + beta ** 2) * dP_dx2 + T * T_prime
      
   elif parameters['coordinates'] == 'compactified':

      beta = np.sqrt (1 - x2 ** 2) / (x1 * parameters['R_lc'])

      pulsar_eq = x1 ** 2 * (1 - beta ** 2) * (x1 ** 2 * d2P_dx12 + (1 - x2 ** 2) * d2P_dx22) \
                  + 2 * x1 ** 2 * (x1 * dP_dx1 + x2 * beta ** 2 * dP_dx2) + T * T_prime
      
   elif parameters['coordinates'] == 'cylindrical':

      beta = x1 / parameters['R_lc']

      pulsar_eq = (1 - beta ** 2) * (d2P_dx12 - 1 / x1 * dP_dx1 + d2P_dx22) - 2 * beta * dP_dx1 + T * T_prime   
      
   else:

      raise ValueError ("coordinates must be one of 'spherical', 'compactified', 'cylindrical'.\n\
                        Current value --> {}".format(parameters['coordinates']))
      
   return pulsar_eq


# ===================================================================================================


