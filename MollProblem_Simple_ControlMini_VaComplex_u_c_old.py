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
parser.add_argument("--shrinkstep", type=float)
parser.add_argument("--shrinkcoef", type=float)
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
shrinkstep = args.shrinkstep
shrinkcoef = args.shrinkcoef
activation_FFNN = args.activation_FFNN
# Training parameters
sampling_stages  = args.sampling_stages   # number of times to resample new time-space domain points
steps_per_sample = args.steps_per_sample    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = args.nSim_interior
nSim_boundary = args.nSim_boundary
idd = args.id

weight = args.weight
penalty = weight
# multipliers for oversampling i.e. draw X from [X_low - X_oversample, X_high + X_oversample]
X_oversample = 0.5
t_oversample = 0.5

# Plot options
n_test = 100  # Points on plot grid for each dimension

n_plot = 600
# Save options
saveOutput = False
savefolder = 'Moll_ControlMini_VaComplex_u_c_old/num_layers_FFNN_{}_activation_FFNN_{}_num_layers_RNN_{}_nodes_per_layer_{}/sampling_stages_{}_steps_per_sample_{}_starting_learning_rate_{}_shrinkstep_{}_shrinkcoef_{}_weight_{}/nSim_interior_{}_nSim_boundary_{}/id_{}/'.format(num_layers_FFNN, activation_FFNN, num_layers_RNN, nodes_per_layer, sampling_stages, steps_per_sample, starting_learning_rate, shrinkstep, shrinkcoef, weight, nSim_interior, nSim_boundary, idd)
saveName   = 'MollProblem' 
saveFigure = True
figureName = 'MollProblem' 

# Analytical Solution
os.makedirs('./Figure/'+savefolder+'/',exist_ok=True)

os.makedirs('./SavedNets/' +savefolder+ '/', exist_ok=True)
# market price of risk

Moll_Va = pd.read_csv("./MollData/Va_f_40000,600_Shrink.csv", header = None)
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

    X_interior = np.random.uniform(low=X_low, high=X_high, size=[nSim_interior, 2])

    # Sampler #2: underline a

    a_alower = X_low[0] * np.ones((nSim_interior, 1))
    z_alower0 = X_low[1] * np.ones((1,1))
    z_alower1 = np.random.uniform(low=0, high=0.25, size=[int(nSim_interior/2), 1]) * (X_high[1] - X_low[1]) + X_low[1]
    z_alower2 = np.random.uniform(low=0.25, high=1, size=[int(nSim_interior/2)-1, 1]) * (X_high[1] - X_low[1]) + X_low[1]
    z_alower = np.concatenate([z_alower0, z_alower1,z_alower2],axis=0)

    X_alower = np.concatenate([a_alower,z_alower],axis=1)


    a_zboundary = np.random.uniform(
        low=X_low[0], high=X_high[0], size=[nSim_boundary, 1])
    z_zboundary = X_low[1] + (X_high[1] - X_low[1]) * np.random.binomial(1, 0.5, size = (nSim_boundary,1))

    X_zboundary = np.concatenate([a_zboundary,z_zboundary],axis=1)


    return X_interior, X_alower, X_zboundary

def control_c(model, X):
    a = X[:,0:1]
    z = X[:,1:2]

    V = model(tf.stack([a[:,0],z[:,0]], axis=1))
    V_a = tf.gradients(V, a)[0]

    V_a = tf.maximum(1e-10 * tf.ones_like(V), V_a)

    c = u_deriv_inv(V_a)
    
    return c

def value_info(model, X):
    a = X[:,0:1]
    z = X[:,1:2]

    V = model(tf.stack([a[:,0],z[:,0]], axis=1))
    V_a = tf.gradients(V, a)[0]
    V_aa = tf.gradients(V_a, a)[0]
    V_z = tf.gradients(V, z)[0]
    V_zz = tf.gradients(V_z, z)[0]

    
    return V, V_a, V_aa, V_z, V_zz



def loss_differentialoperator(model, X):
    a = X[:,0:1]
    z = X[:,1:2]

    V = model(tf.stack([a[:,0],z[:,0]], axis=1))
    V_a = tf.gradients(V, a)[0]
    V_z = tf.gradients(V, z)[0]
    V_zz = tf.gradients(V_z, z)[0]

    V_a = tf.maximum(1e-10 * tf.ones_like(V), V_a)

    c = u_deriv_inv(V_a)
    u_c = u(c) 
    
    diff_V = -rho*V+u_c+V_a * \
        (z+r*a-c)+(-the*tf.math.log(z)+sig2/2)*z*V_z + sig2*z**2/2*V_zz   
        
    L = tf.reduce_mean(tf.square(diff_V))
    return diff_V, L

def loss_differentialoperator_alower(model, X):
    a = X[:,0:1]
    z = X[:,1:2]

    V = model(tf.stack([a[:,0],z[:,0]], axis=1))
    V_a = tf.gradients(V, a)[0]
    V_z = tf.gradients(V, z)[0]
    V_zz = tf.gradients(V_z, z)[0]

    V_a = tf.maximum(1e-10 * tf.ones_like(V), V_a)


    V_a_new = tf.maximum(u_deriv(z+ r* a), V_a)

    c_new = u_deriv_inv(V_a_new)
    u_c_new = u(c_new) 
    
    c_old = u_deriv_inv(V_a)
    u_c_old = u(c_old) 

    Drift = tf.maximum(z+r*a-c_old, tf.zeros_like(V))

    diff_V = -rho*V+u_c_old+V_a * \
        Drift+(-the*tf.math.log(z)+sig2/2)*z*V_z + sig2*z**2/2*V_zz

    L = tf.reduce_mean(tf.square(diff_V))
    return diff_V, L

def loss_zboundarys(model, X):
    a = X[:,0:1]
    z = X[:,1:2]

    V = model(tf.stack([a[:,0],z[:,0]], axis=1))
    V_z = tf.gradients(V, z)[0]

    
    L = tf.reduce_mean( tf.square(V_z ) )
    return L

def loss_concave(model, X):
    a = X[:,0:1]
    z = X[:,1:2]

    V = model(tf.stack([a[:,0],z[:,0]], axis=1))
    V_a = tf.gradients(V, a)[0]
    V_aa = tf.gradients(V_a, a)[0]
    concave_V = tf.maximum(V_aa, tf.zeros_like(V))

    L = tf.reduce_mean( tf.square(concave_V ) )
    return L


def loss_neumann_boundary_alower(model, X):
    a = X[:,0:1]
    z = X[:,1:2]

    V = model(tf.stack([a[:,0],z[:,0]], axis=1))
    V_a = tf.gradients(V, a)[0]
    V_a = tf.maximum(1e-10 * tf.ones_like(V), V_a)

    Diff = tf.minimum(V_a-u_deriv(z+r*a),tf.zeros_like(V))

    L = tf.reduce_mean(tf.square(Diff))

    return L

def loss(model, X_interior, X_alower, X_zboundary):
    ''' Compute total loss for training.
    
    Args:
        model:      DGM model object

    ''' 

    Loss_V_interior, L1 = loss_differentialoperator(model, X_interior)

    Loss_V_alower, L2 = loss_differentialoperator_alower(model, X_alower)
    
    L3 = loss_zboundarys(model, X_zboundary) + loss_concave(model, X_interior)

    L4 = tf.cast(0,tf.float32)

    return L1, L2, L3, L4, Loss_V_interior, Loss_V_alower
    

# Set up network

# initialize DGM model (last input: space dimension = 1)
model = DGM2.DCGM3Net(X_low, X_high, nodes_per_layer, num_layers_FFNN,num_layers_RNN, 2, activation_FFNN)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
# t_interior_tnsr = tf.placeholder(tf.float32, [None,1])
X_interior_tnsr = tf.placeholder(tf.float32, [None,2])
X_alower_tnsr = tf.placeholder(tf.float32, [None,2])
X_zboundary_tnsr = tf.placeholder(tf.float32, [None,2])


# loss 
L1_tnsr, L2_tnsr, L3_tnsr, L4_tnsr, Loss_V_interior_tnsr, Loss_V_alower_tnsr = loss(
    model, X_interior_tnsr, X_alower_tnsr, X_zboundary_tnsr)
Loss_tnsr = L1_tnsr + L2_tnsr +  L3_tnsr + L4_tnsr

c_tnsr = control_c(model, X_interior_tnsr)
V_tnsr, V_a_tnsr, V_aa_tnsr, V_z_tnsr, V_zz_tnsr = value_info(model,X_interior_tnsr)

# set optimizer - NOTE THIS IS DIFFERENT FROM OTHER APPLICATIONS!
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step, shrinkstep, shrinkcoef, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Loss_tnsr, global_step=global_step)

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


    X_interior, X_alower, X_zboundary = sampler(nSim_interior, nSim_boundary)
    
    for _ in range(steps_per_sample):
        loss,L1,L2,L3,L4,_ = sess.run([Loss_tnsr, L1_tnsr, L2_tnsr, L3_tnsr, L4_tnsr, optimizer],
                                   feed_dict={X_interior_tnsr: X_interior, X_alower_tnsr: X_alower, X_zboundary_tnsr: X_zboundary})
        loss_list.append(loss)
    

    if loss>1e-5:
        if i%100==0:

            print("{:7d}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}".format(int(i), loss, L1, L2, L3, L4))
    else:
        if i%10==0:

            print("{:7d}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}".format(int(i), loss, L1, L2, L3, L4))
            
    if i%4000==0:
        
        aspace = np.linspace(-0.02, 4, n_plot)
        zspace = np.linspace(zmean*0.8, zmean*1.2, n_plot)
        A, Z = np.meshgrid(aspace, zspace)
        X_interior = np.vstack([A.flatten(), Z.flatten()]).T

        X_alower = np.vstack([A[A==-0.02].flatten(), Z[A==-0.02].flatten()]).T
        X_zboundary = np.vstack([A[(Z==zmean*0.8) | (Z==zmean*1.2)].flatten(), Z[(Z==zmean*0.8) | (Z==zmean*1.2)].flatten()]).T


        fitted_V, fitted_Va, fitted_Vaa, fitted_Vz, fitted_Vzz = sess.run([V_tnsr, V_a_tnsr, V_aa_tnsr, V_z_tnsr, V_zz_tnsr], feed_dict={X_interior_tnsr: X_interior})
        fitted_c = sess.run([c_tnsr], feed_dict={X_interior_tnsr: X_interior})[0]

        Loss_V_interior, Loss_V_alower = sess.run([Loss_V_interior_tnsr, Loss_V_alower_tnsr],
                                   feed_dict={X_interior_tnsr: X_interior, X_alower_tnsr: X_alower, X_zboundary_tnsr: X_zboundary})


        averageL1 = np.mean(np.abs(Loss_V_interior).reshape(n_plot, n_plot))
        averageL2 = np.mean(np.square(Loss_V_interior).reshape(n_plot, n_plot))


        Loss_V_interior_log10 = np.log(np.abs(Loss_V_interior))/np.log(10)

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(A, Z, np.zeros_like(A), color='red' , rcount=10, ccount=10)
        ax.plot_wireframe(A, Z, -5*np.ones_like(A), color='purple' , rcount=10, ccount=10)
        surf = ax.plot_surface(A, Z, Loss_V_interior_log10.reshape(n_plot, n_plot), cmap='viridis')
        fig.colorbar(surf, ax=ax)
        ax.view_init(35, 35)
        ax.set_xlabel('$a$')
        ax.set_ylabel('$z$')
        ax.set_zlim(-10,5)

        ax.set_title('Loss: L1Average={}, L2Average={}'.format(averageL1, averageL2))

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


        average_alower_L1 = np.mean(np.abs(Loss_V_alower))
        average_alower_L2 = np.mean(np.square(Loss_V_alower))

        A_alower, Z_alower = X_alower[:,0:1] , X_alower[:,1:2]
        
        # Loss_V_alower_log10 = np.log(np.abs(Loss_V_alower))/np.log(10)

        fig = plt.figure(figsize=(16, 9))
        # plt.plot(Z_alower[:, 0], Loss_V_alower_log10[:, 0],
        #         label='Log of Loss', color='black')
        plt.plot(Z_alower[:, 0], Loss_V_alower[:, 0],
                label='Loss', color='black')        
        plt.xlabel('$z$')
        plt.title('Loss: L1Average={}, L2Average={}'.format(average_alower_L1, average_alower_L2))
        
        plt.legend()
        plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_LossV_Slice_{}.png'.format(i),bbox_inches='tight')


        plt.close('all')



# save outout

os.makedirs('./SavedNets/' +savefolder+ '/', exist_ok=True)
# if saveOutput:
saver = tf.train.Saver()
# saver.save(sess, './SavedNets/' +savefolder+ '/' + saveName)
       
# Plot value function results
# model.summary()
os.makedirs('./Figure/'+savefolder+'/',exist_ok=True)



# vector of t and S values for plotting
aspace = np.linspace(-0.02, 4, n_plot)
zspace = np.linspace(zmean*0.8, zmean*1.2, n_plot)
A, Z = np.meshgrid(aspace, zspace)
X_interior = np.vstack([A.flatten(), Z.flatten()]).T


X_alower = np.vstack([A[A==-0.02].flatten(), Z[A==-0.02].flatten()]).T
X_zboundary = np.vstack([A[(Z==zmean*0.8) | (Z==zmean*1.2)].flatten(), Z[(Z==zmean*0.8) | (Z==zmean*1.2)].flatten()]).T


fitted_V, fitted_Va, fitted_Vaa, fitted_Vz, fitted_Vzz = sess.run([V_tnsr, V_a_tnsr, V_aa_tnsr, V_z_tnsr, V_zz_tnsr], feed_dict={X_interior_tnsr: X_interior})
fitted_c = sess.run([c_tnsr], feed_dict={X_interior_tnsr: X_interior})[0]

Loss_V_interior, Loss_V_alower = sess.run([Loss_V_interior_tnsr, Loss_V_alower_tnsr],
                            feed_dict={X_interior_tnsr: X_interior, X_alower_tnsr: X_alower, X_zboundary_tnsr: X_zboundary})

averageL1 = np.mean(np.abs(Loss_V_interior).reshape(n_plot, n_plot))
averageL2 = np.mean(np.square(Loss_V_interior).reshape(n_plot, n_plot))

Loss_V_interior_log10 = np.log(np.abs(Loss_V_interior))/np.log(10)


fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(A, Z, np.zeros_like(A), color='red' , rcount=10, ccount=10)
ax.plot_wireframe(A, Z, -5*np.ones_like(A), color='purple' , rcount=10, ccount=10)
surf = ax.plot_surface(A, Z, Loss_V_interior_log10.reshape(n_plot, n_plot), cmap='viridis')
fig.colorbar(surf, ax=ax)# ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
ax.set_zlim(-10,5)
# ax.set_zlabel('$v(a,z)$')
ax.set_title('Loss: L1Average={}, L2average={}'.format(averageL1, averageL2))

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
plt.plot(Z[:, 0], fitted_Va.reshape(n_plot, n_plot)[:, 0],
        label='NN Solution', color='black')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
        label='State Constraint', color='red')
plt.plot(Z[:, 0], Moll_Va[:,0],
        label='FDM Solution', color='blue', linestyle=":")
plt.xlabel('$z$')
# plt.ylim(0.75, 1.10)
plt.legend()
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_VaSlice.png', bbox_inches='tight')


average_alower_L1 = np.mean(np.abs(Loss_V_alower))
average_alower_L2 = np.mean(np.square(Loss_V_alower))

A_alower, Z_alower = X_alower[:,0:1] , X_alower[:,1:2]

# Loss_V_alower_log10 = np.log(np.abs(Loss_V_alower))/np.log(10)

fig = plt.figure(figsize=(16, 9))
# plt.plot(Z_alower[:, 0], Loss_V_alower_log10[:, 0],
#         label='Log of Loss', color='black')
plt.plot(Z_alower[:, 0], Loss_V_alower[:, 0],
        label='Loss', color='black')        
plt.xlabel('$z$')
plt.title('Loss: L1Average={}, L2Average={}'.format(average_alower_L1, average_alower_L2))

plt.legend()
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_LossV_Slice.png',bbox_inches='tight')


fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
ax.semilogy(range(len(loss_list)), loss_list, 'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi^{n_{epoch}}$')
# if saveFigure:
plt.savefig( './Figure/' +savefolder+ '/' + saveName + '_Loss.png',bbox_inches='tight')



pd.DataFrame(fitted_Va.reshape(n_plot, n_plot)).to_csv('./Figure/' +savefolder+ '/' + saveName +'_Va.csv',header=False,index=False)    
