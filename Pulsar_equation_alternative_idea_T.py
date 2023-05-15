# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:26:41 2023

@author: Usuario
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Add, Activation
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow import float32, concat, convert_to_tensor,float64
from tensorflow.keras.optimizers.legacy import Adam,Adadelta,SGD, Adamax
from matplotlib.colors import LogNorm 
import matplotlib.cm as cm
import matplotlib
import random

tf.keras.backend.set_floatx('float64')
# Print tensorflow version
print('TensorFLow version: ', tf.__version__)

# Check if GPU is used
print(tf.config.list_physical_devices('GPU'))


#Network computing P and Pc
Xp_input = Input((2,))
X = Dense(80,activation="tanh")(Xp_input)
X = Dense(80,activation="tanh")(X)
X = Dense(80,activation="tanh")(X)
X = Dense(2,activation=None)(X)
Np = Model(inputs=Xp_input, outputs=X)

#Network that computes T from P
Xt_input = Input((1,))
X = Dense(40,activation="tanh")(Xt_input)
X = Dense(40,activation="tanh")(X)
X = Dense(1,activation=None)(X)
Nt = Model(inputs=Xt_input, outputs=X)


Rlc = 15 #Light cylinder radius in units of the pulsar radius
rc = 15 #Radius of the ypoint
Nint = 1000 #Inputs for the interior
Nlc = 100 #Inputs at LC, to enforce TT' = 2Bz

def T_michel(P,Pc,Rlc=Rlc): #Michel Monopole solution for T(P)
    return -P*(2-P/Pc)/Rlc

def generate_inputs(Nint,Nlc): #Input generation
    q = np.random.rand(Nint)
    theta = np.pi*np.random.rand(Nint)
    mu = np.cos(theta)
    qlc = np.random.rand(Nlc)/Rlc
    mulc = np.random.choice([-1,1],size=Nlc)*np.sqrt(1-qlc**2*Rlc**2)
    return convert_to_tensor(np.hstack((q[:,None],mu[:,None])),float64),\
           convert_to_tensor(np.hstack((qlc[:,None],mulc[:,None])),float64) 

def P_t(X,Np): #P computation
    q = X[:,0,None]
    mu = X[:,1,None]
    Nout = Np(X)
    Pc = Nout[:,1]
    rho = tf.math.sqrt(1-mu**2)/q
    fb = tf.math.reduce_mean(Pc) + q*((1-tf.math.reduce_mean(Pc))/((rc**2-(1-mu**2))**2))*\
        tf.nn.relu(rc**2-rho**2)**2
    h = (1-q)*(tf.nn.relu(1-rho/rc)**2 + mu**2)
    P = (1-mu**2)*(fb+h*Nout[:,0,None])
    return P,Pc
    
def sign(x,b=0.5): #Sign function implemented with the heaviside function
    return 2*tf.experimental.numpy.heaviside(x,b)-1 #b=0.5 sets the sign to zero at x=0. ç

def soft_heav(x,mean,width): #A soft heaviside function. There are many variants, I used the erfc function. Width that Contopoulos put was 0.1Psi_open = 0.1*Rlc*Pc
    return tf.math.erfc((x-mean)/width)/2

def T_t(X,P,Pc,Nt):
    Pcmean = tf.math.reduce_mean(Pc)
    mu = X[:,1,None]
    return (-2*P/Rlc + P**2*Nt(P))*tf.experimental.numpy.heaviside(Pcmean**2-P**2,0.5)*sign(mu) #I think it is not necessary the sign, because TT' does not change sign

#To compute the residuals
def PDE(Np,Nt,X):
   
   # Unpack input array
   q = X[:,0]
   mu = X[:,1]

   with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt1:
      gt1.watch(X)

      with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt2:
         gt2.watch(X)

         # Calculate P
         P,Pc = P_t(X,Np)
      
      Pgrad = gt2.gradient(P, X)
      P_q = Pgrad[:,0]
      P_mu = Pgrad[:,1]
      
   P_qq = gt1.gradient(P_q, X)[:,0]
   P_mumu = gt1.gradient(P_mu, X)[:,1]
   
   with tf.GradientTape() as gtP:
       gtP.watch(P)
       T = T_t(X,P,Pc,Nt)
       
   dTdP = gtP.gradient(T,P)[:,0]
   br2 = (1-mu**2)/(q**2*Rlc**2)
   gs = q**2*(2*((1-mu**2)/Rlc**2 + q**2*(1-br2))*P_q/q) + q**4*(1-br2)*P_qq + \
       (1-mu**2)*q**2*(2*mu*P_mu/(q**2*Rlc**2) + (1-br2)*P_mumu)
   eqP = gs + T[:,0]*dTdP
   return P,Pc,eqP/tf.math.sqrt(tf.math.abs(mu)),T,dTdP,P_q,P_mu

lr = keras.optimizers.schedules.ExponentialDecay(2e-2,10000,0.98)
# optimizer = Adam(lr,0.99,0.999) #The optimizer
optimizer = Adamax(lr)

loss_function = keras.losses.MeanSquaredError()

def loss(X,eqP,Pc):
    zeros = tf.zeros(X.shape[0],float64)
    lossPDE = loss_function(eqP,zeros)
    lossPc = tf.math.reduce_std(Pc)
    return (lossPDE + 0.1*lossPc)/1.1

def losslc(T,dTdP,Pq,Pmu,Xlc):
    return loss_function(T[:,0]*dTdP/(Xlc[:,0]**2*tf.math.sqrt(tf.math.abs(Xlc[:,1]))),\
                         -2*(Xlc[:,0]*Pq+Xlc[:,1]*Pmu)/tf.math.sqrt(tf.math.abs(Xlc[:,1])))

def grads(Np,Nt,X,Xlc): #Gradients wrt the trainable parameters
    with tf.GradientTape(persistent=True) as tape2:
        P,Pc,eqP,T = PDE(Np,Nt,X)[:4]
        Tlc,dTdPlc,Pqlc,Pmulc = PDE(Np,Nt,Xlc)[3:]
        loss_value = 0.9*loss(X,eqP,Pc)+0.1*losslc(Tlc,dTdPlc,Pqlc,Pmulc,Xlc)
    gradientsP = tape2.gradient(loss_value,Np.trainable_variables)
    gradientsT = tape2.gradient(loss_value,Nt.trainable_variables)
    #Clip gradients wrt parameters if norm exceeds a thresold. 
    gradientsP, _ = tf.clip_by_global_norm(gradientsP, 0.1)
    gradientsT, _ = tf.clip_by_global_norm(gradientsT, 0.1)
    return gradientsP,gradientsT,P,T,Pc,loss_value

@tf.function(jit_compile=True)
def training(Np,Nt,X,optimizer): #Training step function
    gradientsP,gradientsT,P,T,Pc,loss_value = grads(Np,Nt,X,Xlc)
    optimizer.apply_gradients(zip(gradientsP,Np.trainable_variables))
    optimizer.apply_gradients(zip(gradientsT,Nt.trainable_variables))
    print('test tf.function')
    return P,T,Pc,loss_value,tf.norm(gradientsP[0])

Nepochs=20000
loss_list = np.zeros(round(int(Nepochs/100)+1,1))
template = 'Epoch {}, loss: {}'
X,Xlc = generate_inputs(Nint,Nlc)
epochs = []
loss_vs_epoch = []
open('data/loss_vs_epochs.txt', "w").close()

for i in range(Nepochs):
    if (i+1)%2000 == 0:
        X,Xlc = generate_inputs(Nint,Nlc)
        P,Pc,eqP,T = PDE(Np,Nt,X)[:4]
        Tlc,dTdPlc,Pqlc,Pmulc = PDE(Np,Nt,Xlc)[3:]
        loss_value = (loss(X,eqP,Pc)+losslc(Tlc,dTdPlc,Pqlc,Pmulc,Xlc))/2
        #X = random_permutation(X)
    P,T,Pc,loss_value,gradientsP = training(Np,Nt,X,optimizer)
    epochs.append(i)
    loss_vs_epoch.append(loss_value)
    if (i+1)%100 == 0 or i == 0:
        print(template.format(i+1,loss_value)) 
        print(tf.math.reduce_mean(Pc).numpy(),tf.math.reduce_std(Pc).numpy())
        with open('data/loss_vs_epochs.txt', 'a') as f:
            np.savetxt(f,np.column_stack((np.array(epochs), np.array(loss_vs_epoch))))
        epochs = []
        loss_vs_epoch = []

def generate_test(Nr,Ntheta):
    r = np.linspace(1,Rlc,Nr)
    theta = np.linspace(0,np.pi,Ntheta)
    theta,radius = np.meshgrid(theta,r)
    mu = np.cos(theta)
    q = 1/radius
    X = np.hstack((q.flatten()[:,None],mu.flatten()[:,None]))
    return convert_to_tensor(X),radius,theta

Nr = 400
Ntheta = 401
Xtest,radius,theta = generate_test(Nr,Ntheta)

q = 1/radius
mu = np.cos(theta)


Ptest,Pc = P_t(Xtest,Np)
with tf.GradientTape() as gtP:
     gtP.watch(Ptest)
     Ttest = T_t(Xtest,Ptest,Pc,Nt)
     
dTdP = gtP.gradient(Ttest,Ptest)[:,0]
Ptest = Ptest.numpy().reshape(Nr,Ntheta)
Ttest = Ttest.numpy().reshape(Nr,Ntheta)
dTdP = dTdP.numpy().reshape(Nr,Ntheta)

Np.save("data/Np_Rlc15_rc15_q1_width_01/Weights")
Nt.save("data/Nt_Rlc15_rc15_q1_width_01/Weights")


# ticks = np.linspace(Ptest.min(),Ptest.max(),400)
# ticks_colorbar = np.linspace(Ptest.min(),Ptest.max(),5)

# Pc = tf.math.reduce_mean(Pc)

# matplotlib.rcParams.update({'font.size': 28})
# fig1 = plt.figure(figsize=(16,16))
# plt.gca().set_aspect('equal')
# plt.contour(radius*np.sin(theta),radius*np.cos(theta),Ptest,ticks,colors="k")
# plt.contour(radius*np.sin(theta),radius*np.cos(theta),Ptest,levels=[Pc],colors="r")
# plt.xlabel(r'$x/R$',fontsize=28)
# plt.ylabel(r'$y/R$',fontsize=28)

# plt.axvline(x=rc)
# plt.axvline(x=Rlc,color="orange")

# fig2 = plt.figure(figsize=(16,16))
# clev = np.arange(Ptest.min(),Ptest.max(),.01)
# Plot = plt.contourf(radius*np.sin(theta),radius*np.cos(theta),Ptest,clev,extend="both")
# fig2.colorbar(Plot,ticks=ticks_colorbar)



# Tmich = T_michel(Ptest,Pc)

# plt.figure(3,figsize=(16,16))
# plt.scatter(Ptest[Ptest<=Pc]*Rlc,-Ttest[Ptest<=Pc]*Rlc**2,s=2)
# #plt.scatter(Ptest[Ptest<=Pc]*Rlc,-Tmich[Ptest<=Pc]*Rlc**2,s=2)
# plt.xlabel(r'$\psi$')
# plt.ylabel(r'$S$')



# plt.figure(4,figsize=(16,16))
# plt.scatter(Ptest[Ptest<=Pc]*Rlc,Ttest[Ptest<=Pc]*dTdP[Ptest<=Pc]*Rlc**3,s=2)
# plt.xlabel(r'$\psi$')
# plt.ylabel(r'$S\frac{dS}{d\psi}$')

# plt.show()

