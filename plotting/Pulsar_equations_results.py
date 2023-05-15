# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:45:19 2023

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
from tensorflow.keras.optimizers import Adam,Adadelta,SGD
from matplotlib.colors import LogNorm 
import matplotlib.cm as cm
import matplotlib
import copy
from numpy.random import default_rng
import os
from matplotlib.patches import Circle

#Train in Float64 
tf.keras.backend.set_floatx('float64')

#Load the networks

Np = keras.models.load_model("data/Rlc15_rc15_q1_alternative_idea_Np_T_no_EovB6/Weights")
Nt = keras.models.load_model("data/Rlc15_rc15_q1_alternative_idea_Nt_T_no_EovB6/Weights")

#Physical parameters: LC radius (Rlc) and y-point radius (rc)

Rlc = 15
rc = 15
rmax = 3*Rlc/2 #Radius until wou want to compute the solution.

#Test points functions 
def generate_test(Nr,Ntheta):
    radius = np.linspace(1,Rlc,Nr)
    theta = np.linspace(1e-2,np.pi-1e-2,Ntheta)
    theta,radius = np.meshgrid(theta,radius)
    mu = np.cos(theta)
    q = 1/radius
    X = np.hstack((q.flatten()[:,None],mu.flatten()[:,None]))
    return convert_to_tensor(X),radius,theta

#Test points in cartesian coordinates
def generate_cart_test(Nx,Nz,Xmax,Zmax):
    x = np.linspace(1e-4,Xmax,Nx)
    z = np.linspace(-Zmax,Zmax,Nz)
    Z,X = np.meshgrid(z,x)
    r = np.hypot(X,Z)
    mu = Z/r
    q = 1/r
    Xtest = np.hstack((q.flatten()[:,None],mu.flatten()[:,None]))
    return convert_to_tensor(Xtest),X,Z
    

#Test points corresponding to spherical slices at a distance r to the star center
def generate_ang_slices(Nth,r):
    radius = r*np.ones(Nth)
    mu = np.linspace(-1,1,Nth)
    theta = np.arccos(mu)
    q = 1/radius
    X = np.hstack((q.flatten()[:,None],mu.flatten()[:,None]))
    return convert_to_tensor(X),theta

#Test points at theta = constant
def generate_rad_slices(Nr,th):
    radius = np.linspace(1,rmax,Nr)
    theta = th*np.ones(Nr)
    mu = np.cos(theta)
    q = 1/radius
    X = np.hstack((q.flatten()[:,None],mu.flatten()[:,None]))
    return convert_to_tensor(X),radius

#Test points at z = constant
def generate_z_slices(Nr,z):
    radius = np.linspace(z,rmax,Nr)
    theta = np.arccos(z/radius)
    mu = np.cos(theta)
    q = 1/radius
    X = np.hstack((q.flatten()[:,None],mu.flatten()[:,None]))
    return convert_to_tensor(X),radius,theta

#Test points at rho = constant
def generate_rho_slices(Nr,rho,zmax):
    z = np.linspace(1e-4,zmax,Nr)
    radius = np.sqrt(z**2+rho**2)
    mu = z/radius
    theta = np.arccos(mu)
    q = 1/radius
    X = np.hstack((q.flatten()[:,None],mu.flatten()[:,None]))
    return convert_to_tensor(X),radius,theta,z

#P and T calculation functions

def P_t(X,Np):
    q = X[:,0,None]
    mu = X[:,1,None]
    Nout = Np(X)
    Pc = Nout[:,1]
    rho = tf.math.sqrt(1-mu**2)/q
    #Xeq = tf.stack([q[:,0],tf.zeros(mu.shape[0],float64)],axis=1)
    #Xlc = tf.stack([tf.ones(mu.shape[0],float64)/rc,tf.zeros(mu.shape[0],float64)],axis=1)
    fb = tf.math.reduce_mean(Pc) + q*((1-tf.math.reduce_mean(Pc))/((rc**2-(1-mu**2))**2))*\
        tf.nn.relu(rc**2-rho**2)**2
    h = (1-q)*(tf.nn.relu(1-rho/rc)**2+mu**2)
    P = (1-mu**2)*(fb+h*Nout[:,0,None])
    return P,Pc
    
def sign(x,b=0.5):
    return 2*tf.experimental.numpy.heaviside(x,b)-1 #b=0.5 sets the sign to zero at x=0. 

def T_t(X,P,Pc,Nt):
    mu = X[:,1,None]
    T = (-2*P/Rlc + P**2*Nt(P))*\
         tf.experimental.numpy.heaviside(tf.math.reduce_mean(Pc)**2-P**2,1)*sign(mu)
    return T


#Residuals and derivatives computed with the PINN

def PDE(Np,Nt,X):
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
    derivatives = (T,dTdP,P_q,P_mu,P_qq,P_mumu)
    return eqP,derivatives

#Magnetic, electric and electric currenty density fields calculation
def fields(Np,Nt,X):
    T,dTdP,P_q,P_mu,P_qq,P_mumu = PDE(Np,Nt,X)[-1]
    q = X[:,0]
    mu = X[:,1]
    Br = -q**2*P_mu
    Bth = q**3*P_q/tf.math.sqrt(1-mu**2)
    Bphi = T[:,0]*q/tf.math.sqrt(1-mu**2)
    Er = q**2*P_q/Rlc
    Eth = q*tf.math.sqrt(1-mu**2)*P_mu/Rlc
    Jr = -q**2*dTdP*P_mu
    Jth = q**3*dTdP*P_q/tf.math.sqrt(1-mu**2)
    Jphi = q**3*(-2*q*P_q-q**2*P_qq-(1-mu**2)*P_mumu)/tf.math.sqrt(1-mu**2)
    Bmod = Br**2+Bth**2+Bphi**2
    Bpol = Bmod-Bphi**2
    Emod = Er**2 + Eth**2
    fields = (Br,Bth,Bphi,Er,Eth,Jr,Jth,Jphi,Bmod,Emod,Bpol)
    return fields


Nr = 500
Ntheta = 601
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

ticks = np.linspace(Ptest.min(),Ptest.max(),500)
ticks_colorbar = np.linspace(Ptest.min(),Ptest.max(),5)

Pc = tf.math.reduce_mean(Pc)

matplotlib.rcParams.update({'font.size': 28})
fig1 = plt.figure(figsize=(16,16))
plt.gca().set_aspect('equal')
plt.contour(radius*np.sin(theta),radius*np.cos(theta),Ptest,ticks,colors="k")
plt.contour(radius*np.sin(theta),radius*np.cos(theta),Ptest,levels=[Pc],colors="r")
plt.xlabel(r'$x/R$',fontsize=28)
plt.ylabel(r'$y/R$',fontsize=28)
plt.axvline(x=rc)
plt.axvline(x=Rlc,color="orange")
# plt.savefig("Pcontours_Rlc_15_rc_15_q1_subnetwork.jpg")


fig2 = plt.figure(figsize=(16,16))
clev = np.arange(Ptest.min(),Ptest.max(),.01)
Plot = plt.contourf(radius*np.sin(theta),radius*np.cos(theta),Ptest,clev,extend="both")
fig2.colorbar(Plot,ticks=ticks_colorbar)
# plt.savefig("Pcontourf_Rlc_15_rc_15_q1_subnetwork.jpg")



plt.figure(3,figsize=(16,16))
plt.scatter(Ptest[Ptest<=Pc]*Rlc,-Ttest[Ptest<=Pc]*Rlc**2,s=2)
#plt.scatter(Ptest[Ptest<=Pc]*Rlc,-Tmich[Ptest<=Pc]*Rlc**2,s=2)
plt.xlabel(r'$\psi$')
plt.ylabel(r'$S$')
# plt.savefig("S(psi).jpg")




plt.figure(4,figsize=(16,16))
plt.scatter(Ptest[Ptest<=Pc]*Rlc,Ttest[Ptest<=Pc]*dTdP[Ptest<=Pc]*Rlc**3,s=2)
plt.xlabel(r'$\psi$')
plt.ylabel(r'$S\frac{dS}{d\psi}$')
# plt.savefig("SS'(psi).jpg")

matplotlib.rcParams.update({'font.size': 28})

plt.show()



Nr = 300
Ntheta = 300
Xtest,radius,theta = generate_test(Nr,Ntheta)

eqTp = PDE(Np, Nt, Xtest)[0]
eqTp = eqTp.numpy().reshape(Nr,Ntheta)

ticks_colorbar = np.linspace(np.abs(eqTp).min(),np.abs(eqTp).max(),5)

fig3 = plt.figure(figsize=(20,20))
plt.gca().set_aspect('equal')
plt.title("Radial component residuals")
clev = np.arange(np.abs(eqTp).min(),np.abs(eqTp).max()+1e-5,1e-5)
Plot = plt.contourf(radius*np.sin(theta),radius*np.cos(theta),np.abs(eqTp),clev,cmap="copper")
fig3.colorbar(Plot,ticks=ticks_colorbar)
plt.xlabel(r'$x/R$',fontsize=28)
plt.ylabel(r'$z/R$',fontsize=28)
# plt.savefig("eqTp_rc15_Rlc_15_q1_Pulsar_equation_noEovB.jpg",bbox_inches="tight")



matplotlib.rcParams.update({'font.size': 28})
Nz = 1000
Xtest,radius,theta,z = generate_rho_slices(Nz, rho=Rlc,zmax=30)

derivatives = PDE(Np, Nt, Xtest)[1]
Ptest = P_t(Xtest,Np)[0]
T,dTdP,Pq,Pmu = derivatives[:4]
q = 1/radius
mu = np.cos(theta)
Bz = -q**2*(q*Pq.numpy()+mu*Pmu.numpy())

plt.figure(figsize=(20,20))
plt.plot(Ptest[:,0].numpy()*Rlc,T[:,0].numpy()*dTdP.numpy()/2)
plt.plot(Ptest[:,0].numpy()*Rlc,Bz)
plt.show()

'''
Nr = 500
Ntheta = 501
Xtest,radius,theta = generate_test(Nr,Ntheta)

q = 1/radius
mu = np.cos(theta)

Br,Bth,Bphi,Er,Eth,Jr,Jth,Jphi,Bmod,Emod = fields(Np,Nt,Xtest)


diff = Bmod-Emod
diff = diff.numpy().reshape(Nr,Ntheta)
ticks_colorbar = np.logspace(diff.min(),diff.max(),5)

fig3 = plt.figure(figsize=(20,20))
plt.gca().set_aspect('equal')
plt.title(r'$B^2 - E^2$')
clev = np.arange(diff.min(),diff.max()+1e-3,1e-3)
Plot = plt.contourf(radius*np.sin(theta),radius*np.cos(theta),diff,clev,cmap="copper",norm = LogNorm())
fig3.colorbar(Plot,ticks=ticks_colorbar)
plt.xlabel(r'$x/R$',fontsize=28)
plt.ylabel(r'$z/R$',fontsize=28)
plt.savefig("Bmod-Emod_Rlc_15_rc_15_q1_subnetwork.jpg")
plt.show()
'''

Nz = 1000
Xtest,radius,theta,z = generate_rho_slices(Nz, rho=0.8*Rlc,zmax=10)

Br,Bth,Bphi,Er,Eth,Jr,Jth,Jphi,Bmod,Emod,Bpol = fields(Np,Nt,Xtest)


diff = Bmod.numpy()-Emod.numpy()
difpol = Bpol.numpy()-Emod.numpy()

matplotlib.rcParams.update({'font.size': 28})
plt.figure(figsize=(16,16))
plt.title("Difference between E2 and B2 at " + r'$\rho =$'+" constant")
plt.plot(z,diff,label=r'$B^2 - E^2$')
plt.plot(z,difpol,label=r'$B_p^2 - E^2$')
plt.plot(z,np.abs(Bphi.numpy()),label=r'$-B_\phi$')
plt.xlabel(r'$z$'+' '+r'$(\rho=0.8R_{LC})$')
plt.yscale("log")
plt.legend(loc="best")
# plt.savefig("rho_slices_diff_E2_B2_rho_08Rlc.jpg",bbox_inches="tight")
plt.show()

'''
Nr = 500
Ntheta = 501
Xtest,radius,theta = generate_test(Nr,Ntheta)

q = 1/radius
mu = np.cos(theta)

Br,Bth,Bphi,Er,Eth,Jr,Jth,Jphi,Bmod,Emod = fields(Np,Nt,Xtest)


diff = Bmod-Emod
diff = diff.numpy().reshape(Nr,Ntheta)

ticks = np.linspace(diff.min(),diff.max(),100)
matplotlib.rcParams.update({'font.size': 28})
fig1 = plt.figure(figsize=(16,16))
plt.gca().set_aspect('equal')
plt.contour(radius*np.sin(theta),radius*np.cos(theta),diff,ticks,colors="k")
plt.xlabel(r'$x/R$',fontsize=28)
plt.ylabel(r'$y/R$',fontsize=28)
plt.axvline(x=rc)
plt.axvline(x=Rlc,color="orange")
plt.show()
'''






    
    
    




    
    
    
