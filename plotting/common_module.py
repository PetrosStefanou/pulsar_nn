import numpy as np
import tensorflow as tf




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


def generate_test (x1, x2, coordinates):

   """
      Creates th input tensor of the PINN for a test in some grid. 'x1' and 'x2' form a 2D meshgrid and 'coordinates' specifies
      the type of the coordinates (possible choises are 'spherical', 'compactified', 'cylindrical', 'cartesian').
   """
   
   X = np.zeros([x1.size, 2])


   if coordinates == 'spherical': 
   
      q = 1 / x1
      mu = np.cos(x2)

   elif coordinates == 'compactified':

      q = x1
      mu = x2

   elif coordinates == 'cylindrical':

      q = 1 / np.sqrt (x1 ** 2 + x2 ** 2)
      mu = 1 / np.sqrt (x1 ** 2 / x2 ** 2 + 1)

   elif coordinates == 'cartesian':

      q = 1 / np.sqrt (x1 ** 2 + x2 ** 2)
      mu = 1 / np.sqrt (x1 ** 2 / x2 ** 2 + 1)

   else:

      raise ValueError ("Coordinates must be one of 'spherical', 'compactified', 'cylindrical', 'cartesian'.")

   
   X[:,0] = q.flatten()
   X[:,1] = mu.flatten()

   return tf.convert_to_tensor(X)


# ===================================================================================================


def f_b (X, P_c, r_c):

   q = X [:,0]
   mu = X [:,1]

   fb = (1 - mu ** 2) * (P_c + q * ((1 - P_c) * tf.nn.relu(1 - 1 / (q * r_c)) ** 2 / (1 - 1 / r_c) ** 2))

   return fb


# ===================================================================================================


def h_b (X, r_c):

   q = X [:,0]
   mu = X [:,1]

   hb = (1 - mu ** 2) * (1 - q) * tf.sqrt(tf.nn.relu (1 - 1 / (q * r_c)) ** 2 + mu ** 2)

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


def T_parametrization (P, P_c, N_t, R_lc):
   
   # Reshape N_t output from (n_points,1) to (n_points, ) for proper broadcasting
   N_t = tf.reshape(N_t (P), N_t (P).shape[0])

   T = (-2 * P / R_lc + P ** 2 * N_t) * tf.experimental.numpy.heaviside(P_c ** 2 - P ** 2, 0.5)
    
   return T


# ===================================================================================================


def get_scalar_functions(X, N_p, N_t, parameters):

   # Calculate P, Pc and T from the parametrisations

   P, Pc = P_parametrization (X, N_p, parameters)
   T = T_parametrization (P, Pc, N_t, parameters['R_lc'])

   # Convert to np.arrays and reshape
   P = P.numpy().reshape([parameters['n_1'], parameters['n_2']])
   T = T.numpy().reshape([parameters['n_1'], parameters['n_2']])
   Pc = float(Pc)

   # Calculate T'(P)
   T_prime = np.gradient(T.flatten(), P.flatten(), edge_order=2).reshape(T.shape)
   T_prime = np.gradient(T.flatten(), P.flatten(), edge_order=2).reshape(T.shape)

   return P, Pc, T, T_prime


# ===================================================================================================


def magnetic_field (x1, x2, P, T, coordinates):
    
   """
      Calculate the magnetic field from the stream functions 'P' and 'T' in the 'x1', 'x2' grid specified by 'coordinates'.
   """

   dPdx1 = np.gradient(P, x1[:,1], axis=0, edge_order=2)
   dPdx2 = np.gradient(P, x2[1,:], axis=1, edge_order=2)

   if coordinates == 'spherical':
      
      B1 = 1 / (x1 ** 2 * np.sin(x2)) * dPdx2
      B2 = -1 / (x1 * np.sin(x2)) * dPdx1
      B3 = 1 / (x1 * np.sin(x2)) * T

   elif coordinates == 'compactified':

      B1 =  - x1 ** 2 * dPdx2
      B2 = -x1 ** 3 / np.sqrt (1 - x2 ** 2) * dPdx1
      B3 = x1 / np.sqrt (1 - x2 ** 2) * T

   elif coordinates == 'cylindrical':

      B1 = - 1 / x1 * dPdx2
      B2 = 1 / x1 * dPdx1
      B3 = 1 / x1 * T

   else:

      raise ValueError ("coordinates must be one of 'spherical', 'compactified', 'cylindrical'.\n\
                        Current value --> {}".format(coordinates))
   
   return B1, B2, B3


# ===================================================================================================


def electric_field (x1, x2, P, R_lc, coordinates):
    
   """
      Calculate the electric field from the stream function 'P' in the 'x1', 'x2' grid specified by 'coordinates'.
   """

   dPdx1 = np.gradient(P, x1[:,1], axis=0, edge_order=2)
   dPdx2 = np.gradient(P, x2[1,:], axis=1, edge_order=2)

   if coordinates == 'spherical':
      
      E1 = - 1 / R_lc * dPdx1
      E2 = - 1 / (R_lc * x1) * dPdx2
      E3 = np.zeros_like (E1)

   elif coordinates == 'compactified':

      E1 = 1 / R_lc * x1 ** 2 * dPdx1
      E2 = 1 / R_lc * x1 * np.sqrt (1 - x2 ** 2) * dPdx2
      E3 = np.zeros_like (E1)

   elif coordinates == 'cylindrical':

      E1 = - 1 / R_lc * dPdx1
      E2 = - 1 / R_lc * dPdx2
      E3 = np.zeros_like (E1)
      
   else:

      raise ValueError ("coordinates must be one of 'spherical', 'compactified', 'cylindrical'.\n\
                        Current value --> {}".format(coordinates))
   
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


def div (x1, x2, u1, u2, coordinates):
    
   '''
      Calculate the divergence of a vector 'u' in the 'coordinates' system. Axisymmetry is supposed. 
   '''

   du1dx1 = np.gradient(u1, x1[:,1], axis=0, edge_order=2)
   du2dx2 = np.gradient(u2, x2[1,:], axis=1, edge_order=2)

   if coordinates == 'spherical':

      divergence =  1 / x1 ** 2 * (2 * x1 * u1 + x1 ** 2 * du1dx1) + 1 / (x1 * np.sin (x2)) * (np.cos (x2) * u2 + np.sin(x2) * du2dx2)

   elif coordinates == 'compactified':

      divergence = 2 * x1 * u1 + x1 ** 2 * du1dx1 + x1 / np.sqrt(1 - x2 ** 2) * (x2 * u2 + np.sqrt(1 - x2 ** 2) * du2dx2)

   elif coordinates == 'cylindrical':

      divergence = 1 / x1 * (u1 + x1 * du1dx1) + du2dx2

   else:

      raise ValueError ("coordinates must be one of 'spherical', 'compactified', 'cylindrical'.\n\
                        Current value --> {}".format(coordinates))

   return divergence


# ===================================================================================================


def get_fields (x1, x2, P, T, parameters):
   
   # Magnetic and electric field
   B_1, B_2, B_3 = magnetic_field (x1, x2, P, T, coordinates=parameters['coordinates'])
   E_1, E_2, E_3 = electric_field (x1, x2, P, parameters ['R_lc'], coordinates=parameters['coordinates'])

   # Magnitude
   B_mag = magnitude (B_1, B_2, B_3)
   E_mag = magnitude (E_1, E_2, E_3)
   B_pol = magnitude (B_1, B_2, 0)

   # Scalar product
   E_dot_B = dot (B_1, B_2, B_3, E_1, E_2, E_3)

   # Divergence
   div_B = div(x1, x2, B_1, B_2, coordinates=parameters['coordinates'])
   div_E = div(x1, x2, E_1, E_2, coordinates=parameters['coordinates'])

   return B_1, B_2, B_3, E_1, E_2, E_3, B_mag, E_mag, B_pol, E_dot_B, div_B, div_E


# ===================================================================================================

