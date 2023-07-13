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


def P_parametrization (X, N_p, parameters):
   
   """
      Calculate P and P_c using the output of the network and the parametrisation
   """
   
   Np = N_p (X)
   
   if parameters['Pc_mode'] == 'free':
      
      P_c = tf.abs(tf.reduce_mean (Np[:,1]))
   
   elif parameters['Pc_mode'] == 'fixed':
     
      P_c = parameters['Pc_fixed']
   
   else:
      
      raise ValueError ("Pc_mode must be one of 'free', 'fixed'.\n\
                        Current value --> {}".parameters['Pc_mode'])

   P = f_b (X, P_c, parameters) + h_b (X, parameters) * Np[:,0]

   return P, P_c


# ===================================================================================================


def f_b (X, Pc, parameters):

   """
      Function giving the boundary conditions in the parametrization of P.
   """

   q = X [:,0]
   mu = X [:,1]

   # fb = (1 - mu ** 2) * (Pc + q * ((1 - Pc) * tf.nn.relu(1 - 1 / (q * r_c)) ** 3 / (1 - 1 / r_c) ** 3))

   dipole = 1 - mu**2
   monopole = Pc * (1 - tf.abs(mu))

   f1 = q ** 2
   f2 = tf.exp(- q ** 2 / (2 * (0.5 / parameters['R_lc']) ** 2))
   
   fb = f1 * dipole + f2 * monopole

   return fb


# ===================================================================================================


def h_b (X, parameters):

   q = X [:,0]
   mu = X [:,1]

   # hb = (1 - mu ** 2) * (1 - q) * tf.sqrt(tf.nn.relu (1 - 1 / (q * r_c)) ** 3 + mu ** 2)
   hb = 4 * q * (1 - q) ** 2 * (1 - mu ** 2)

   return hb


# ===================================================================================================


def T_parametrization (P, Pc, N_t, parameters):
   
   # Reshape N_t output from (n_points,1) to (n_points, ) for proper broadcasting
   N_t = tf.reshape(N_t (P), P.shape)
   
   T = N_t * g_b (P, Pc, parameters)
    
   return T


# ===================================================================================================


def g_b (P, Pc, parameters):

   """
      Parametrisation function for T(P)
   """

   width = parameters['sep_width']

   gb = P ** 2 * (1 + tf.experimental.numpy.heaviside(P - (Pc - 3 * width), 1) * (tf.exp(- (P - (Pc - 3 * width)) ** 2 / (2 * width ** 2)) - 1))

   return gb


# ===================================================================================================


