"""
   Module containing functions used in the pinn_training.ipynb notebook
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from scipy.special import legendre
import time


def create_model(pinn_config, print_summary=False):
   """
      Create a neural network that accepts N_input input neurons with N_layer-1 hidden layers
      of N_neuron number of neurons per hidden layer.
   """
   
   # Size of input
   N_input = 2 + pinn_config['N_multipoles'] - 1

   # Input layer
   X_input = Input((N_input,), name='input_layer')

   # Hidden layers
   X = Dense(pinn_config['N_neurons'], activation="tanh", name='hidden_layer_1')(X_input)
   for i in range(2, pinn_config['N_layers']):
      X = Dense(pinn_config['N_neurons'], activation="tanh", name='hidden_layer_{}'.format(i))(X)

   # Output layer
   X = Dense(1, activation=None, name='output_layer')(X)
   model = Model(inputs=X_input, outputs=X, name='PINN_PDE_Solver')
   
   # Print summary to console
   if print_summary:
      print(model.summary())

   return model


"=================================================================================="


def generate_input (N_points, N_multipoles, qmin=0.0, qmax=1.0, mumin=-1.0, mumax=1.0, blmin=-1.0, blmax=1.0):
   """
      Generate N_points number of NN inputs where each input neuron has a random value in its
      corresponding range
   """
   
   # Random number generator
   rng = np.random.default_rng()

   # Initialise input array
   X = np.zeros([N_points, 2+N_multipoles-1])

   # Fill with random values for q, theta and bls
   X[:,0] = rng.uniform(qmin, qmax, N_points)
   X[:,1] = rng.uniform(mumin, mumax, N_points)
   X[:,2:] = rng.uniform(blmin, blmax, [N_points, N_multipoles-1])

   return tf.convert_to_tensor(X)


"=================================================================================="


def generate_test(Nr, Ntheta, b_l, rmin=1.0, rmax=10.0, thetamin=0.0, thetamax=np.pi):
   #  a1,a2,a3,a4,a5 = coefs
   
   X = np.zeros([Nr * Ntheta, len(b_l) + 2])
   
   r, theta = np.meshgrid(np.linspace(rmin, rmax, Nr), np.linspace(thetamin, thetamax, Ntheta), indexing='ij')
   q = 1 / r
   mu = np.cos(theta)
   
   X[:,0] = q.flatten()
   X[:,1] = mu.flatten()
   X[:,2:] = b_l

   return r, theta, tf.convert_to_tensor(X)


"=================================================================================="


def P_l_prime (l, mu):
   """
      Evaluate the derivative of the legendre polynomial of order l at points mu
   """
   Pl = legendre(l)
   Plprime = Pl.deriv()
   Plprime_coeffs = list(Plprime.coeffs)
   Plprime_eval = tf.math.polyval(Plprime_coeffs, mu)

   return Plprime_eval


"=================================================================================="


def f_boundary (X):
  
   """
      Function that satisfies the boundary conditions at the boundary
   """

   # Unpack input array
   q = X[:,0]
   mu = X[:,1]
   b_l = X[:,2:]
   
   # Initialise with dipole contribution (always equal to 1)
   fb = tf.ones_like(mu)

   # Add multipole contributions for l>1 (the c_l[:,0] element is the cuadrupole coefficient)
   for l in range (2, b_l.shape[1]+2):
      fb += b_l[:,l-2] / l * P_l_prime (l, mu)

   # Multiply by q*sin^2(theta)
   fb = q * (1 - mu ** 2) * fb

   return fb


"=================================================================================="


def h_boundary (X):

   """
      Function that is zero at the boundary
   """
   
   # Unpack input array
   q = X[:,0]
   mu = X[:,1]
   
   return q * (1.0 - q) * (1.0 - mu ** 2)


"=================================================================================="


def P_solution (X, N):
   
   """
      Calculate the solution from the parametrisation, using the PINN output (N),
      the boundary condition (f_boundary) and the function describing the boundary (h_boundary)
   """
   
   # Network's output N is reshaped from (N_points,1) to (N_points,) to be compatible
   # with f_boundary and h_boundary shapes.
   N_reshaped = tf.reshape(N(X), h_boundary(X).shape)
   
   return (f_boundary(X) + h_boundary(X) * N_reshaped)


"=================================================================================="


def gradshafranov(X, N):
   
   # Unpack input array
   q = X[:,0]
   mu = X[:,1]

   with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt1:
      gt1.watch(X)

      with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt2:
         gt2.watch(X)

         # Calculate P
         P = P_solution (X, N)

      # Calculate 1st derivatives
      Pgrad = gt2.gradient(P, X)
      P_q = Pgrad[:,0]
      P_mu = Pgrad[:,1]

   # Calculate 2nd derivatives
   P_qq = gt1.gradient(P_q, X)[:,0]
   P_mumu = gt1.gradient(P_mu, X)[:,1]

   # Calculate Grad-Shafranov operator
   gs = q ** 2 * (q * (q * P_qq + 2.0 * P_q) + (1.0 - mu ** 2) * P_mumu)

   return P, gs


"=================================================================================="


# @tf.function(jit_compile=True)
def train_step(X, N, loss_function, optimizer): 

   """
         Gets the approximate solution and the PDE and optimizes the network so that the PDE is satisfied 
   """
   
   with tf.GradientTape() as gt:
      
      # Get solution and Grad-Shafranov equation
      P, gs = gradshafranov (X, N)

      # Calculate loss of the PDE
      loss = loss_function (gs, tf.zeros_like(gs))
   
   # Calculate derivatives of the loss w.r.t. the network's parameters 
   parameter_gradients = gt.gradient(loss, N.trainable_variables)

   # Optimize
   optimizer.apply_gradients(zip(parameter_gradients, N.trainable_variables))

   return P, gs, loss


"=================================================================================="


# @tf.function(jit_compile=True)
def evaluation_step (X_test, N, gs_metrics, P_metrics, gs_exact, P_exact):
   
   # Evaluate result for the test example
   P_test, gs_test = gradshafranov(X_test, N)

   # Reset and update the state of metrics for gs
   for i in range(len(gs_metrics)):
      gs_metrics[i].reset_states()
      gs_metrics[i].update_state(gs_test, gs_exact)

   # Reset and update the state of metrics for P
   for i in range(len(P_metrics)):
      P_metrics[i].reset_states()
      P_metrics[i].update_state(P_test, P_exact)


"=================================================================================="


def train_model (pinn_config, N
                , X_test, gs_metrics, P_metrics, gs_exact, P_exact
                , summary_writer, template
                , train_step):
   
   # Initialize timers
   time_eval = 0
   time_total = 0

   # Unpack metrics
   P_abs_loss, P_rel_loss, P_square_loss = P_metrics 
   test_loss = gs_metrics[0]
   
   # Get loss and optimizer from configuration dictionary
   loss_function = eval(pinn_config['loss_function'])
   optimizer = eval(pinn_config['optimizer'])

   # Train the model
   for e in range(pinn_config['N_epochs']):
      # Start timer
      start = time.time()
      
      # Renew the training points every N_renew epochs
      if e % pinn_config['N_renew'] == 0:
         X = generate_input (pinn_config['N_points'], pinn_config['N_multipoles'])

      # Make a training step
      
      P, gs, loss = train_step (X, N, loss_function, optimizer)

      # Update timers
      time_epoch = time.time() - start
      time_eval += time_epoch
      time_total += time_epoch

      # Calculate and print metrics every N_metrics epochs
      if e % pinn_config['N_eval'] == 0 or e == pinn_config['N_epochs'] - 1:
            
         # Evaluate the test example and calculate relevant metrics
         evaluation_step(X_test, N, gs_metrics, P_metrics, gs_exact, P_exact)

         # Print info to console
         print (template.format(e, loss.numpy(), test_loss.result().numpy()
                              , P_abs_loss.result().numpy(), time_eval)) 
         
         # Log for tensorboard
         with summary_writer.as_default():

            tf.summary.scalar('Train loss', loss.numpy(), step=e)
            tf.summary.scalar('Test loss', test_loss.result().numpy(), step=e)
            tf.summary.scalar('P abs loss', P_abs_loss.result().numpy(), step=e)
            tf.summary.scalar('P rel loss', P_rel_loss.result().numpy(), step=e)
            tf.summary.scalar('P square loss', P_square_loss.result().numpy(), step=e)
            tf.summary.scalar('Time_per_{}_epochs'.format(pinn_config['N_eval']), time_eval, step=e)

         # Reset Nmetrics timer
         time_eval = 0 


   # Print total training time
   print('='*50)
   print('Total time (s):', time_total)

   return loss.numpy(), test_loss.result().numpy() \
          , P_abs_loss.result().numpy(), P_rel_loss.result().numpy(), P_square_loss.result().numpy() \
          , time_total