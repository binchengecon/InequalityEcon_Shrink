# SCRIPT FOR SOLVING THE MERTON PROBLEM
#%%
# import needed packages

import DGM2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import csv
import pandas as pd

# Parameters 


parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--num_layers_FFNN", type=int, default=0)
parser.add_argument("--activation_FFNN", type=str, default = "tanh")
parser.add_argument("--num_layers_RNN", type=int, default=3)
parser.add_argument("--nodes_per_layer", type=int, default=50)
parser.add_argument("--sampling_stages", type=int, default=1000)
parser.add_argument("--steps_per_sample", type=int, default=10)
parser.add_argument("--nSim_interior", type=int, default=1000)
parser.add_argument("--nSim_boundary", type=int, default=64)
# args = parser.parse_args()
args, unknown = parser.parse_known_args()


# Sannikov problem parameters 
gamma = 0.5
r = 0.03
rho = 0.05
Var = 0.07
Corr = 0.9
the = -np.log(Corr)
sig2 = 2*the*Var
zmean = np.exp(Var/2)

# Solution parameters (domain on which to solve PDE)
X_low = np.array([-0.02, zmean*0.8])  # wealth lower bound
X_high = np.array([4, zmean*1.2])          # wealth upper bound

# neural network parameters
num_layers_FFNN = args.num_layers_FFNN
num_layers_RNN = args.num_layers_RNN
nodes_per_layer = args.nodes_per_layer
starting_learning_rate = 0.001
activation_FFNN = args.activation_FFNN
# Training parameters
sampling_stages  = args.sampling_stages   # number of times to resample new time-space domain points
steps_per_sample = args.steps_per_sample    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = args.nSim_interior
nSim_boundary = args.nSim_boundary

# multipliers for oversampling i.e. draw X from [X_low - X_oversample, X_high + X_oversample]
X_oversample = 0.5
t_oversample = 0.5

# Plot options
n_plot = 600  # Points on plot grid for each dimension

# Save options
saveOutput = False
savefolder = 'Moll_KFE/num_layers_FFNN_{}_activation_FFNN_{}_num_layers_RNN_{}_nodes_per_layer_{}/sampling_stages_{}_steps_per_sample_{}/nSim_interior_{}_nSim_boundary_{}'.format(num_layers_FFNN, activation_FFNN, num_layers_RNN, nodes_per_layer, sampling_stages, steps_per_sample, nSim_interior, nSim_boundary)
saveName   = 'MollProblem' 
saveFigure = True
figureName = 'MollProblem' 

# Analytical Solution

# market price of risk


def u(c):
    return c**(1-gamma)/(1-gamma)


def u_deriv(c):
    return c**(-gamma)


# @tf.function
def u_deriv_inv(c):
    return c**(-1/gamma)




# Sampling function - randomly sample time-space pairs

def sampler(nSim_interior, nSim_boundary):
    ''' Sample time-space points from the function's domain; points are sampled
        uniformly on the interior of the domain, at the initial/terminal time points
        and along the spatial boundary at different time points. 
    
    Args:
        nSim_interior: number of space points in the interior of the function's domain to sample 
        nSim_terminal: number of space points at terminal time to sample (terminal condition)
    ''' 
    
    # Sampler #1: domain interior    
    # t_interior = np.random.uniform(low=t_low - 0.5*(T-t_low), high=T, size=[nSim_interior, 1])
#    t_interior = np.random.uniform(low=t_low - t_oversample, high=T, size=[nSim_interior, 1])
    # X_interior = np.random.uniform(low=X_low - 0.5*(X_high-X_low), high=X_high + 0.5*(X_high-X_low), size=[nSim_interior, 1])
    # a_interior = np.random.uniform(low=X_low[0], high=X_high[0], size=[nSim_interior, 1])
    a_interior = np.random.uniform(low=0, high=1, size=[nSim_interior, 1])**2 * (X_high[0]-X_low[0]) + X_low[0]
    # z_interior = np.random.uniform(low=X_low[1], high=X_high[1], size=[nSim_interior, 1])
    z_interior = np.random.uniform(low=0, high=1, size=[nSim_interior, 1])**2 * (X_high[1]-X_low[1]) + X_low[1]
#    X_interior = np.random.uniform(low=X_low - X_oversample, high=X_high + X_oversample, size=[nSim_interior, 1])
#    X_interior = np.random.uniform(low=X_low * X_multiplier_low, high=X_high * X_multiplier_high, size=[nSim_interior, 1])

    # Sampler #2: spatial boundary
        # no spatial boundary condition for this problem
    a_NBC = np.random.uniform(
        low=X_low[0], high=X_high[0], size=[nSim_boundary, 1])
    z_NBC = X_low[1] + (X_high[1] - X_low[1]) * np.random.binomial(1, 0.5, size = (nSim_boundary,1))

    a_SC_upper = X_high[0] * np.ones((nSim_boundary,1))
    z_SC_upper = np.random.uniform(
        low=X_low[1], high=X_high[1], size=[nSim_boundary, 1])

    a_SC_lower = X_low[0] * np.ones((nSim_boundary, 1))
    # z_SC_lower = np.random.uniform(
    #     low=0, high=1, size=[nSim_boundary, 1])
    z_SC_lower = np.random.uniform(
        low=0, high=1, size=[nSim_boundary, 1])**2 * (X_high[1]-X_low[1]) + X_low[1]
    # Sampler #3: initial/terminal condition
    # t_terminal = T * np.ones((nSim_terminal, 1))
#    X_terminal = np.random.uniform(low=X_low - X_oversample, high=X_high + X_oversample, size = [nSim_terminal, 1])
    # X_terminal = np.random.uniform(low=X_low - 0.5*(X_high-X_low), high=X_high + 0.5*(X_high-X_low), size = [nSim_terminal, 1])
#    X_terminal = np.random.uniform(low=X_low * X_multiplier_low, high=X_high * X_multiplier_high, size = [nSim_terminal, 1])
    
    # return t_interior, X_interior, t_terminal, X_terminal
    return a_interior, z_interior, a_NBC, z_NBC, a_SC_upper, z_SC_upper, a_SC_lower, z_SC_lower
    # return X_interior.astype(np.float32), X_boundary_NBC.astype(np.float32), X_boundary_SC.astype(np.float32)

# Loss function for Merton Problem PDE


def loss(model, a_interior, z_interior, a_NBC, z_NBC, a_SC_upper, z_SC_upper, a_SC_lower, z_SC_lower):
    ''' Compute total loss for training.
    
    Args:
        model:      DGM model object
        t_interior: sampled time points in the interior of the function's domain
        X_interior: sampled space points in the interior of the function's domain
        t_terminal: sampled time points at terminal point (vector of terminal times)
        X_terminal: sampled space points at terminal time
    ''' 
    # length = X_interior.shape[0]
    # Loss term #1: PDE
    # compute function value and derivatives at current sampled points
    # print(X_interior.shape, X_interior[:, 0:1].shape, X_interior[:, 1:2].shape)
    
    V = model(tf.stack([a_interior[:,0], z_interior[:,0]], axis=1))[:,0:1]
    g = model(tf.stack([a_interior[:,0], z_interior[:,0]], axis=1))[:,1:2]
    # V = model(a_interior, z_interior)
    V_a = tf.gradients(V, a_interior)[0]
    V_aa = tf.gradients(V_a, a_interior)[0]
    V_z = tf.gradients(V, z_interior)[0]
    V_zz = tf.gradients(V_z, z_interior)[0]
     
    
    
    c = tf.where(V_a <= 0, tf.zeros_like(V), u_deriv_inv(V_a))

    u_c = u(c)
    
    diff_V = -rho*V+u_c+V_a * \
        (z_interior+r*a_interior-c)+(-the*tf.math.log(z_interior)+sig2/2)*z_interior*V_z + sig2*z_interior**2/2*V_zz

    concave_V = tf.maximum(V_aa, tf.zeros_like(V))
    # compute average L2-norm of differential operator
    L1 = tf.reduce_mean(tf.square(diff_V))  + tf.reduce_mean(tf.square(concave_V))
    
    save = z_interior+r*a_interior-c 
    KFE_a = tf.gradients(g*save, a_interior)[0]

    mu_z = (-the*tf.math.log(z_interior)+sig2/2)*z_interior
    
    KFE_z = tf.gradients(g*mu_z, z_interior)[0]

    sigma_z = sig2
    
    KFE_zz0 = tf.gradients(g*sigma_z, z_interior)[0]
    KFE_zz1 = tf.gradients(KFE_zz0, z_interior)[0]
    
    KFE = -KFE_a - KFE_z + 1/2*KFE_zz1

    L1_KFE = tf.reduce_mean(tf.square(KFE))
    L2_KFE = tf.square(tf.reduce_mean(g) - 1)

    L3_KFE = tf.square(tf.reduce_mean(g*a_interior))
    
    L_KFE = L1_KFE+  L2_KFE + L3_KFE

    # Loss term #2: boundary condition
        # no boundary condition for this problem
    fitted_boundary_NBC = model(tf.stack([a_NBC[:,0], z_NBC[:,0]], axis=1))
    # fitted_boundary_NBC = model(a_NBC, z_NBC)
    # fitted_boundary_NBC_a = tf.gradients(
    #     fitted_boundary_NBC, X_boundary_NBC[:,0:1])[0]

    fitted_boundary_NBC_z = tf.gradients(
        fitted_boundary_NBC, z_NBC)[0]

    L2 = tf.reduce_mean( tf.square(fitted_boundary_NBC_z ) )
    

    # Loss term #3: initial/terminal condition

    fitted_boundary_SC_lower = model(
        tf.stack([a_SC_lower[:,0], z_SC_lower[:,0]], axis=1))

    # fitted_boundary_SC_lower = model(a_SC_lower[:,0], z_SC_lower[:,0])
    fitted_boundary_SC_lower_a = tf.gradients(
        fitted_boundary_SC_lower, a_SC_lower)[0]
    opt_boundary_SC_lower_a = tf.minimum(fitted_boundary_SC_lower_a - u_deriv(
        z_SC_lower+r*a_SC_lower), tf.zeros_like(fitted_boundary_SC_lower))
    
    L3_lower = tf.reduce_mean(tf.square(opt_boundary_SC_lower_a))


    # L3_lower = tf.reduce_mean(tf.square(fitted_boundary_SC_lower_a - u_deriv(
    # z_SC_lower+r*a_SC_lower)))
    
    
    fitted_boundary_SC_upper = model(
        tf.stack([a_SC_upper[:, 0], z_SC_upper[:, 0]], axis=1))

    # fitted_boundary_SC_upper = model(a_SC_upper, z_SC_upper)
    fitted_boundary_SC_upper_a = tf.gradients(
        fitted_boundary_SC_upper, a_SC_upper)[0]
    opt_boundary_SC_upper_a = tf.maximum(fitted_boundary_SC_upper_a - u_deriv(
        z_SC_upper+r*a_SC_upper), tf.zeros_like(fitted_boundary_SC_upper))
    
    L3_upper = tf.reduce_mean(tf.square(opt_boundary_SC_upper_a))
    
    L3 = L3_lower + L3_upper
    
    
    return L1, L2, L3, L_KFE 
    # return L1, L2
    

# Set up network

# initialize DGM model (last input: space dimension = 1)
model = DGM2.DCGM3Net(X_low, X_high, nodes_per_layer, num_layers_FFNN,num_layers_RNN, 2, 2, activation_FFNN)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
# t_interior_tnsr = tf.placeholder(tf.float32, [None,1])
a_interior_tnsr = tf.placeholder(tf.float32, [None,1])
z_interior_tnsr = tf.placeholder(tf.float32, [None,1])
a_NBC_tnsr = tf.placeholder(tf.float32, [None,1])
z_NBC_tnsr = tf.placeholder(tf.float32, [None,1])
a_SC_upper_tnsr = tf.placeholder(tf.float32, [None,1])
z_SC_upper_tnsr = tf.placeholder(tf.float32, [None,1])
a_SC_lower_tnsr = tf.placeholder(tf.float32, [None, 1])
z_SC_lower_tnsr = tf.placeholder(tf.float32, [None, 1])


# loss 
L1_tnsr, L2_tnsr, L3_tnsr, LKFE_tnsr = loss(
    model, a_interior_tnsr, z_interior_tnsr, a_NBC_tnsr, z_NBC_tnsr,a_SC_upper_tnsr,z_SC_upper_tnsr,a_SC_lower_tnsr,z_SC_lower_tnsr)
loss_tnsr = L1_tnsr + L2_tnsr + L3_tnsr + LKFE_tnsr

# value function

V = model(tf.stack([a_interior_tnsr[:, 0], z_interior_tnsr[:, 0]], axis=1))[:,0:1]

g = model(tf.stack([a_interior_tnsr[:, 0], z_interior_tnsr[:, 0]], axis=1))[:,1:2]

V_a = tf.gradients(V, a_interior_tnsr)[0]
V_aa = tf.gradients(V_a, a_interior_tnsr)[0]
V_z = tf.gradients(V, z_interior_tnsr)[0]
V_zz = tf.gradients(V_a, z_interior_tnsr)[0]

# optimal control computed numerically from fitted value function 
def control_c(V):
    
    V_a = tf.gradients(V, a_interior_tnsr)[0]
    V_aa = tf.gradients(V_a, a_interior_tnsr)[0]
    
    c = tf.where(V_a <= 0, tf.zeros(tf.shape(V)), u_deriv_inv(V_a))
    
    return c



numerical_c = control_c(V)

# set optimizer - NOTE THIS IS DIFFERENT FROM OTHER APPLICATIONS!
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,100000, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

# initialize variables
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)

# Train network
# initialize loss per training
loss_list = []

# for each sampling stage
for i in range(sampling_stages):
    
    # sample uniformly from the required regions
    a_interior, z_interior, a_NBC, z_NBC, a_SC_upper, z_SC_upper, a_SC_lower, z_SC_lower = sampler(nSim_interior, nSim_boundary)
    
    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        loss,L1,L2,L3,LKFE,_ = sess.run([loss_tnsr, L1_tnsr, L2_tnsr, L3_tnsr, LKFE_tnsr, optimizer],
                                   feed_dict={a_interior_tnsr: a_interior, z_interior_tnsr: z_interior, a_NBC_tnsr: a_NBC, z_NBC_tnsr: z_NBC, a_SC_upper_tnsr: a_SC_upper, z_SC_upper_tnsr: z_SC_upper, a_SC_lower_tnsr: a_SC_lower, z_SC_lower_tnsr: z_SC_lower})
        loss_list.append(loss)
    
    print(loss, L1, L2, L3, LKFE, i)

# save outout
#%%
os.makedirs('./SavedNets/' +savefolder+ '/', exist_ok=True)
# if saveOutput:
saver = tf.train.Saver()
saver.save(sess, './SavedNets/' +savefolder+ '/' + saveName)
       
# Plot value function results
model.summary()
os.makedirs('./Figure/'+savefolder+'/',exist_ok=True)

# LaTeX rendering for text in plots
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

figwidth = 10

# figure options
plt.figure()
plt.figure(figsize = (12,10))


# vector of t and S values for plotting

aspace = np.linspace(-0.02, 4, n_plot)
zspace = np.linspace(zmean*0.8, zmean*1.2, n_plot)
A, Z = np.meshgrid(aspace, zspace)
Xgrid = np.vstack([A.flatten(), Z.flatten()]).T

# simulate process at current t 

fitted_V = sess.run([V], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
g = sess.run([g], feed_dict={a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
fitted_c = sess.run([numerical_c], feed_dict={a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
fitted_Va = sess.run([V_a], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
fitted_Vaa = sess.run([V_aa], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
fitted_Vz = sess.run([V_z], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
fitted_Vzz = sess.run([V_zz], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, fitted_V.reshape(n_plot, n_plot), cmap='viridis')
# ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
ax.set_zlabel('$v(a,z)$')
# ax.set_title('Deep Learning Solution')

plt.savefig( './Figure/' +savefolder+ '/' + saveName + '_Value.png',bbox_inches='tight')


Moll_V = pd.read_csv("./MollData/V.csv", header = None)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, fitted_V.reshape(n_plot, n_plot)-Moll_V, cmap='viridis')
# ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
# ax.set_zlabel('$v(a,z)$')
# ax.set_title('Deep Learning Solution')

plt.savefig( './Figure/' +savefolder+ '/' + saveName + '_ValueDiff.png',bbox_inches='tight')


# # Surface plot of solution u(t,x)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, fitted_c.reshape(n_plot, n_plot), cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
# ax.set_zlabel('$c (a,z)$')
# ax.set_title('Deep Learning Solution')


plt.savefig( './Figure/' +savefolder+ '/' + saveName + '_Consumption.png',bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, g.reshape(n_plot, n_plot), cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
# ax.set_zlabel('$c (a,z)$')
# ax.set_title('Deep Learning Solution')


plt.savefig( './Figure/' +savefolder+ '/' + saveName + '_g.png',bbox_inches='tight')



# # Surface plot of solution u(t,x)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, fitted_Va.reshape(n_plot, n_plot), cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
# ax.set_title('$\partial V / \partial a$')
# ax.set_title('Deep Learning Solution')
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_Va.png',bbox_inches='tight')

Moll_Va = pd.read_csv("./MollData/Va_Upwind.csv", header = None)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, fitted_Va.reshape(n_plot, n_plot)-Moll_Va, cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
# ax.set_title('$\partial V / \partial a$')
# ax.set_title('Deep Learning Solution')
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_VaDiffUpwind.png',bbox_inches='tight')


Moll_Vacenter = pd.read_csv("./MollData/Va_center.csv", header = None)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, fitted_Va.reshape(n_plot, n_plot)-Moll_Vacenter, cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
# ax.set_zlabel('$\partial V / \partial a$')
# ax.set_zlabel('Difference')
# ax.set_title('Deep Learning Solution')
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_VaDiffCenter.png',bbox_inches='tight')


# # Surface plot of solution u(t,x)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, fitted_Vaa.reshape(n_plot, n_plot), cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
# ax.set_zlabel('$V_aa$')
# ax.set_title('$\partial^2 V / \partial a^2$')
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_Vaa.png',bbox_inches='tight')


# # Surface plot of solution u(t,x)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, fitted_Vz.reshape(n_plot, n_plot), cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
# ax.set_title('$\partial V / \partial z$')
# ax.set_title('Deep Learning Solution')
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_Vz.png',bbox_inches='tight')


Moll_Vz = pd.read_csv("./MollData/Vz.csv", header = None)

Diff_Vz = fitted_Vz.reshape(n_plot, n_plot)-Moll_Vz

# pd.DataFrame(Diff_Vz).to_csv('./Figure/' +savefolder+ '/' + saveName +'_Diff_Vz.csv')    

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, Diff_Vz, cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
# ax.set_title('$\partial V / \partial z$')
# ax.set_title('Deep Learning Solution')
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_VzDiff.png',bbox_inches='tight')


# # Surface plot of solution u(t,x)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, fitted_Vzz.reshape(n_plot, n_plot), cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
# ax.set_zlabel('$V_aa$')
# ax.set_title('$\partial^2 V / \partial z^2$')
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_Vzz.png',bbox_inches='tight')


#
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
ax.semilogy(range(len(loss_list)), loss_list, 'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi^{n_{epoch}}$')
# if saveFigure:
plt.savefig( './Figure/' +savefolder+ '/' + saveName + '_Loss.png',bbox_inches='tight')
