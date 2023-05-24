# SCRIPT FOR SOLVING THE MERTON PROBLEM

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
from matplotlib import cm

plt.rcParams["savefig.bbox"] = "tight"

# Parameters 


parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--num_layers_FFNN", type=int)
parser.add_argument("--activation_FFNN", type=str, default = "tanh")
parser.add_argument("--num_layers_RNN", type=int)
parser.add_argument("--nodes_per_layer", type=int)
parser.add_argument("--sampling_stages", type=int)
parser.add_argument("--steps_per_sample", type=int)
parser.add_argument("--nSim_interior", type=int)
parser.add_argument("--nSim_boundary", type=int)
parser.add_argument("--LearningRate", type=float)
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--weight", type=int)
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
starting_learning_rate = args.LearningRate # 1e-3
activation_FFNN = args.activation_FFNN
# Training parameters
sampling_stages  = args.sampling_stages   # number of times to resample new time-space domain points
steps_per_sample = args.steps_per_sample    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = args.nSim_interior
nSim_boundary = args.nSim_boundary
idd = args.id

weight = args.weight
# multipliers for oversampling i.e. draw X from [X_low - X_oversample, X_high + X_oversample]
X_oversample = 0.5
t_oversample = 0.5

# Plot options
n_plot = 600  # Points on plot grid for each dimension

# Save options
saveOutput = False
savefolder = 'Moll_control_weight_samplelower/num_layers_FFNN_{}_activation_FFNN_{}_num_layers_RNN_{}_nodes_per_layer_{}/sampling_stages_{}_steps_per_sample_{}_starting_learning_rate_{}_weight_{}/nSim_interior_{}_nSim_boundary_{}/id_{}/'.format(num_layers_FFNN, activation_FFNN, num_layers_RNN, nodes_per_layer, sampling_stages, steps_per_sample, starting_learning_rate, weight, nSim_interior, nSim_boundary, idd)
saveName   = 'MollProblem' 
saveFigure = True
figureName = 'MollProblem' 

# Analytical Solution
os.makedirs('./Figure/'+savefolder+'/',exist_ok=True)

os.makedirs('./SavedNets/' +savefolder+ '/', exist_ok=True)
# market price of risk

Moll_Va = pd.read_csv("./MollData/Va_Upwind.csv", header = None)
print(Moll_Va.shape)
Moll_Va = np.array(Moll_Va)
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

    a_interior = np.random.uniform(low=0, high=1, size=[nSim_interior, 1])**2 * (X_high[0]-X_low[0]) + X_low[0] 
    z_interior = np.random.uniform(low=0, high=1, size=[nSim_interior, 1])**2 * (X_high[1]-X_low[1]) + X_low[1]


    a_lower = X_low[0] * np.ones((nSim_boundary, 1))
    z_lower = np.random.uniform(low=0, high=1, size=[nSim_boundary, 1])**2 * (X_high[1]-X_low[1]) + X_low[1]

    a_interior_lower = np.random.uniform(low=0, high=1, size=[nSim_boundary, 1]) * (1-X_low[0]) + X_low[0] 
    z_interior_lower = np.random.uniform(low=0, high=1, size=[nSim_boundary, 1]) * (X_high[1]-X_low[1]) + X_low[1]


    return a_interior, z_interior, a_lower, z_lower, a_interior_lower, z_interior_lower
    # return X_interior.astype(np.float32), X_boundary_NBC.astype(np.float32), X_boundary_SC.astype(np.float32)

# Loss function for Merton Problem PDE


def loss(model, a_interior, z_interior, a_lower, z_lower, a_interior_lower, z_interior_lower):
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
    
    V = model(tf.stack([a_interior[:,0], z_interior[:,0]], axis=1))

    G = (z_interior[:,0]-X_low[1])**2 * (z_interior[:,0]-X_high[1])**2 * V
    
    G_a = tf.gradients(G, a_interior)[0]
    G_aa = tf.gradients(G_a, a_interior)[0]
    G_z = tf.gradients(G, z_interior)[0]
    G_zz = tf.gradients(G_z, z_interior)[0]
    
    G_a = tf.maximum(1e-10 * tf.ones_like(G), G_a)
    
    c = u_deriv_inv(G_a)
    u_c = u(c) 
    
    
    diff_G = -rho*G+u_c+G_a * \
        (z_interior+r*a_interior-c)+(-the*tf.math.log(z_interior)+sig2/2)*z_interior*G_z + sig2*z_interior**2/2*G_zz

    concave_G = tf.maximum(G_aa, tf.zeros_like(G))

    L1 = tf.reduce_mean(tf.square(diff_G))  + tf.reduce_mean(tf.square(concave_G))

    
    
    V_lower = model(tf.stack([a_lower[:,0], z_lower[:,0]], axis=1))
    G_lower = (z_lower-X_low[1])**2 * (z_lower-X_high[1])**2 * V_lower

    G_a_lower = tf.gradients(G_lower, a_lower)[0]
    G_aa_lower = tf.gradients(G_a_lower, a_lower)[0]
    G_z_lower = tf.gradients(G_lower, z_lower)[0]
    G_zz_lower = tf.gradients(G_z_lower, z_lower)[0]
     
    G_a_lower = tf.maximum(1e-10 * tf.ones_like(G_lower), G_a_lower)

    c_lower = tf.minimum(u_deriv_inv(G_a_lower), z_lower + r*a_lower)
    # c_lower = tf.maximum(c_lower, (X_low[1]+r*X_low[0])*tf.ones_like(G_lower))
    
    u_c_lower = u(c_lower) 
    
    
    diff_G_lower = -rho*G_lower+u_c_lower+G_a_lower * \
        (z_lower+r*a_lower-c_lower)+(-the*tf.math.log(z_lower)+sig2/2)*z_lower*G_z_lower + sig2*z_lower**2/2*G_zz_lower


    L2 = tf.reduce_mean(tf.square(diff_G_lower)) 
    
    

    V_intlow = model(tf.stack([a_interior_lower[:,0], z_interior_lower[:,0]], axis=1))

    G_intlow = (z_interior_lower[:,0]-X_low[1])**2 * (z_interior_lower[:,0]-X_high[1])**2 * V_intlow
    
    G_a_intlow = tf.gradients(G_intlow, a_interior_lower)[0]
    G_aa_intlow = tf.gradients(G_a_intlow, a_interior_lower)[0]
    G_z_intlow = tf.gradients(G_intlow, z_interior_lower)[0]
    G_zz_intlow = tf.gradients(G_z_intlow, z_interior_lower)[0]
    
    G_a_intlow = tf.maximum(1e-10 * tf.ones_like(G_intlow), G_a_intlow)
    
    c_intlow = u_deriv_inv(G_a_intlow)
    u_c_intlow = u(c_intlow) 
    
    
    diff_G_intlow = -rho*G_intlow+u_c_intlow+G_a_intlow * \
        (z_interior_lower+r*a_interior_lower-c_intlow)+(-the*tf.math.log(z_interior_lower)+sig2/2)*z_interior_lower*G_z_intlow + sig2*z_interior_lower**2/2*G_zz_intlow

    L3 = tf.reduce_mean(tf.square(diff_G_intlow)) 


    
    return L1, L2, L3
    # return L1, L2
    

# Set up network

# initialize DGM model (last input: space dimension = 1)
model = DGM2.DCGM2Net(X_low, X_high, nodes_per_layer, num_layers_FFNN,num_layers_RNN, 2, activation_FFNN)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
# t_interior_tnsr = tf.placeholder(tf.float32, [None,1])
a_interior_tnsr = tf.placeholder(tf.float32, [None,1])
z_interior_tnsr = tf.placeholder(tf.float32, [None,1])
a_lower_tnsr = tf.placeholder(tf.float32, [None,1])
z_lower_tnsr = tf.placeholder(tf.float32, [None,1])
a_interior_lower_tnsr = tf.placeholder(tf.float32, [None,1])
z_interior_lower_tnsr = tf.placeholder(tf.float32, [None,1])

# loss 
L1_tnsr, L2_tnsr, L3_tnsr = loss(
    model, a_interior_tnsr, z_interior_tnsr, a_lower_tnsr, z_lower_tnsr, a_interior_lower_tnsr, z_interior_lower_tnsr)
loss_tnsr = L1_tnsr + L2_tnsr + weight * L3_tnsr

# value function

V = model(tf.stack([a_interior_tnsr[:, 0], z_interior_tnsr[:, 0]], axis=1))

G = (z_interior_tnsr-X_low[1])**2 * (z_interior_tnsr-X_high[1])**2 * V

G_a = tf.gradients(G, a_interior_tnsr)[0]
G_aa = tf.gradients(G_a, a_interior_tnsr)[0]
G_z = tf.gradients(G, z_interior_tnsr)[0]
G_zz = tf.gradients(G_z, z_interior_tnsr)[0]

# optimal control computed numerically from fitted value function 
def control_c(G,G_a,G_z,G_aa,G_zz):
    
        
    c = u_deriv_inv(G_a)
    
    return c




def Loss_G(G,G_a,G_z,G_aa,G_zz):
    
    c = u_deriv_inv(G_a)

    Loss = -rho*G+u(c)+G_a * \
        (z_interior_tnsr+r*a_interior_tnsr-c)+(-the*tf.math.log(z_interior_tnsr)+sig2/2)*z_interior_tnsr*G_z + sig2*z_interior_tnsr**2/2*G_zz
        
    loss = tf.math.log(tf.abs(Loss))
    
    return loss


numerical_c = control_c(G,G_a,G_z,G_aa,G_zz)
Loss_GG = Loss_G(G,G_a,G_z,G_aa,G_zz)

# set optimizer - NOTE THIS IS DIFFERENT FROM OTHER APPLICATIONS!
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,20000, 0.95, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr, global_step=global_step)

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

    a_interior, z_interior, a_lower, z_lower, a_interior_lower, z_interior_lower  = sampler(nSim_interior, nSim_boundary)
    
    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        loss,L1,L2,L3,_ = sess.run([loss_tnsr, L1_tnsr, L2_tnsr, L3_tnsr, optimizer],
                                   feed_dict={a_interior_tnsr: a_interior, z_interior_tnsr: z_interior, a_lower_tnsr: a_lower, z_lower_tnsr: z_lower, a_interior_lower_tnsr: a_interior_lower, z_interior_lower_tnsr: z_interior_lower})
        loss_list.append(loss)
    
        # print(loss, L1, L2, L3, L1_KFE,L2_KFE,L3_KFE, i)
        
    if loss>1e-5:
        if i%100==0:

            print("{:7d}, {:.8f}, {:.8f}, {:.8f}, {:.8f}".format(int(i), loss,L1,L2,L3))
    else:
        if i%10==0:

            print("{:7d}, {:.8f}, {:.8f}, {:.8f}, {:.8f}".format(int(i), loss,L1,L2,L3))        
            
    if i%4000==0:
        
        aspace = np.linspace(-0.02, 4, n_plot)
        zspace = np.linspace(zmean*0.8, zmean*1.2, n_plot)
        A, Z = np.meshgrid(aspace, zspace)
        Xgrid = np.vstack([A.flatten(), Z.flatten()]).T

        # simulate process at current t 


        fitted_Va = sess.run([G_a], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

        fitted_V = sess.run([G], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

        fitted_c = sess.run([numerical_c], feed_dict={a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

        fitted_Vaa = sess.run([G_aa], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
        fitted_Vz = sess.run([G_z], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
        fitted_Vzz = sess.run([G_zz], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

        Loss = sess.run([Loss_GG], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]


        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(A, Z, np.zeros_like(A), color='red' , rcount=10, ccount=10)
        surf = ax.plot_surface(A, Z, Loss.reshape(n_plot, n_plot), cmap='viridis')
        fig.colorbar(surf, ax=ax)
        ax.view_init(35, 35)
        ax.set_xlabel('$a$')
        ax.set_ylabel('$z$')
        ax.set_zlim(-10,5)

        # ax.set_zlabel('$v(a,z)$')
        # ax.set_title('Deep Learning Solution')

        plt.savefig( './Figure/' +savefolder+ '/' + saveName + '_LossV_{}.png'.format(i),bbox_inches='tight')




        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(A, Z, fitted_V.reshape(n_plot, n_plot), cmap='viridis')
        # ax.view_init(35, 35)
        ax.set_xlabel('$a$')
        ax.set_ylabel('$z$')
        # ax.set_zlabel('$v(a,z)$')
        # ax.set_title('Deep Learning Solution')

        plt.savefig( './Figure/' +savefolder+ '/' + saveName + '_Value_{}.png'.format(i),bbox_inches='tight')


        # # Surface plot of solution u(t,x)
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(A, Z, fitted_Va.reshape(n_plot, n_plot), cmap='viridis')
        ax.view_init(35, 35)
        ax.set_xlabel('$a$')
        ax.set_ylabel('$z$')
        ax.set_zlim(0.75,1.10)
        # ax.set_title('$\partial V / \partial a$')
        # ax.set_title('Deep Learning Solution')
        plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_Va_{}.png'.format(i),bbox_inches='tight')

        fig = plt.figure(figsize=(16, 9))
        plt.plot(Z[:, 0], fitted_Va.reshape(n_plot, n_plot)[:, 0],
                label='NN Solution', color='black')
        plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
                label='State Constraint', color='red')
        plt.plot(Z[:, 0], Moll_Va[:,0], label='FDM Solution', color='blue', linestyle=":")
        plt.xlabel('$z$')
        plt.legend()
        plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_VaSlice_{}.png'.format(i),bbox_inches='tight')
        plt.close('all')

# save outout

os.makedirs('./SavedNets/' +savefolder+ '/', exist_ok=True)
# if saveOutput:
saver = tf.train.Saver()
# saver.save(sess, './SavedNets/' +savefolder+ '/' + saveName)
       
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

fitted_Va = sess.run([G_a], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

fitted_V = sess.run([G], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

fitted_c = sess.run([numerical_c], feed_dict={a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

fitted_Vaa = sess.run([G_aa], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
fitted_Vz = sess.run([G_z], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
fitted_Vzz = sess.run([G_zz], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

Loss = sess.run([Loss_GG], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]


fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(A, Z, np.zeros_like(A), color='red' , rcount=10, ccount=10)
surf = ax.plot_surface(A, Z, Loss.reshape(n_plot, n_plot), cmap='viridis')
fig.colorbar(surf, ax=ax)
ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
ax.set_zlim(-10,5)
# ax.set_zlabel('$v(a,z)$')
# ax.set_title('Deep Learning Solution')

plt.savefig( './Figure/' +savefolder+ '/' + saveName + '_LossV.png',bbox_inches='tight')



fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, fitted_V.reshape(n_plot, n_plot), cmap='viridis')
# ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
# ax.set_zlabel('$v(a,z)$')
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


# # Surface plot of solution u(t,x)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, fitted_Va.reshape(n_plot, n_plot), cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
ax.set_zlim(0.75,1.10)
# ax.set_title('$\partial V / \partial a$')
# ax.set_title('Deep Learning Solution')
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_Va.png',bbox_inches='tight')


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
ax.set_zlabel('Difference')
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
ax.set_title('$\partial^2 V / \partial z^2$')
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_Vzz.png',bbox_inches='tight')


#
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
ax.semilogy(range(len(loss_list)), loss_list, 'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi^{n_{epoch}}$')
# if saveFigure:
plt.savefig( './Figure/' +savefolder+ '/' + saveName + '_Loss.png',bbox_inches='tight')



pd.DataFrame(fitted_Va.reshape(n_plot, n_plot)).to_csv('./Figure/' +savefolder+ '/' + saveName +'_Va.csv',header=False,index=False)    



fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], fitted_Va.reshape(n_plot, n_plot)[:, 0],
        label='NN Solution', color='black')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
        label='State Constraint', color='red')
plt.plot(Z[:, 0], Moll_Va[:,0],
        label='FDM Solution', color='blue', linestyle=":")
plt.xlabel('$z$')
# plt.ylim(0.75, 1.10)
plt.legend()
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_VaSlice.png', bbox_inches='tight')