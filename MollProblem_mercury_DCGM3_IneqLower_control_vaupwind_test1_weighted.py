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
penalty = weight
# multipliers for oversampling i.e. draw X from [X_low - X_oversample, X_high + X_oversample]
X_oversample = 0.5
t_oversample = 0.5

# Plot options
# n_plot = 60  # Points on plot grid for each dimension

# Save options
saveOutput = False
savefolder = 'Moll_control_newsampleNet2/num_layers_FFNN_{}_activation_FFNN_{}_num_layers_RNN_{}_nodes_per_layer_{}/sampling_stages_{}_steps_per_sample_{}_starting_learning_rate_{}_weight_{}/nSim_interior_{}_nSim_boundary_{}/id_{}/'.format(num_layers_FFNN, activation_FFNN, num_layers_RNN, nodes_per_layer, sampling_stages, steps_per_sample, starting_learning_rate, weight, nSim_interior, nSim_boundary, idd)
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

    a_interior = np.random.uniform(low=0, high=1, size=[nSim_interior, 1])**2 * (X_high[0]-(X_low[0] )) + X_low[0] 
    z_interior = np.random.uniform(low=0, high=1, size=[nSim_interior, 1])**2 * (X_high[1]-X_low[1]) + X_low[1]


    a_interior_lower = X_low[0] * np.ones((nSim_boundary, 1))
    z_interior_lower = np.random.uniform(low=0, high=1, size=[nSim_boundary, 1])**2 * (X_high[1]-X_low[1]) + X_low[1]


    # Sampler #2: spatial boundary

    a_NBC = np.random.uniform(
        low=X_low[0], high=X_high[0], size=[nSim_boundary, 1])
    z_NBC = X_low[1] + (X_high[1] - X_low[1]) * np.random.binomial(1, 0.5, size = (nSim_boundary,1))


    a_SC_upper = X_high[0] * np.ones((nSim_boundary,1))
    z_SC_upper = np.random.uniform(
        low=X_low[1], high=X_high[1], size=[nSim_boundary, 1])

    a_SC_lower = np.random.uniform(low=0, high=1, size=[nSim_boundary, 1])**2 * (1-(X_low[0] )) + X_low[0] 
    z_SC_lower = np.random.uniform(low=0, high=1, size=[nSim_boundary, 1])**2 * (X_high[1]-X_low[1]) + X_low[1]


    return a_interior, z_interior, a_interior_lower, z_interior_lower, a_NBC, z_NBC, a_SC_upper, z_SC_upper, a_SC_lower, z_SC_lower

# Loss function for Merton Problem PDE


def loss(model, a_interior, z_interior, a_interior_lower, z_interior_lower, a_NBC, z_NBC, a_SC_upper, z_SC_upper, a_SC_lower, z_SC_lower):
    ''' Compute total loss for training.
    
    Args:
        model:      DGM model object
        t_interior: sampled time points in the interior of the function's domain
        X_interior: sampled space points in the interior of the function's domain
        t_terminal: sampled time points at terminal point (vector of terminal times)
        X_terminal: sampled space points at terminal time
    ''' 

    
    V = model(tf.stack([a_interior[:,0], z_interior[:,0]], axis=1))
    V_a = tf.gradients(V, a_interior)[0]
    V_aa = tf.gradients(V_a, a_interior)[0]
    V_z = tf.gradients(V, z_interior)[0]
    V_zz = tf.gradients(V_z, z_interior)[0]
    V_a = tf.maximum(1e-10 * tf.ones_like(V), V_a)
    
    c = u_deriv_inv(V_a)
    u_c = u(c) 
    
    diff_V = -rho*V+u_c+V_a * \
        (z_interior+r*a_interior-c)+(-the*tf.math.log(z_interior)+sig2/2)*z_interior*V_z + sig2*z_interior**2/2*V_zz

    concave_V = tf.maximum(V_aa, tf.zeros_like(V))

    # weight = penalty*tf.square(z_interior-(zmean+0.05)) + 1
    # weight_sum = tf.reduce_sum(weight,axis=0,keepdims=True)
    
    # weighted = weight/weight_sum

    # L1 = tf.reduce_sum(tf.square(diff_V) * weighted)
    L1 = tf.reduce_mean(tf.square(diff_V))  
    
    

    
    
    V_lower = model(tf.stack([a_interior_lower[:,0], z_interior_lower[:,0]], axis=1))
    V_a_lower = tf.gradients(V_lower, a_interior_lower)[0]
    V_aa_lower = tf.gradients(V_a_lower, a_interior_lower)[0]
    V_z_lower = tf.gradients(V_lower, z_interior_lower)[0]
    V_zz_lower = tf.gradients(V_z_lower, z_interior_lower)[0]
    V_a_lower = tf.maximum(1e-10 * tf.ones_like(V_lower), V_a_lower)

    index1 = tf.cast(V_a_lower < u_deriv_inv(z_interior_lower+ r* a_interior_lower), tf.float32) 
    index2 = tf.cast(V_a_lower >= u_deriv_inv(z_interior_lower+ r* a_interior_lower), tf.float32)
    V_a_lower_new = u_deriv_inv(z_interior_lower+ r* a_interior_lower) * index1 + V_a_lower * index2
    
    c_lower = u_deriv_inv(V_a_lower_new)
    u_c_lower = u(c_lower) 
    
    
    diff_V_lower = -rho*V_lower+u_c_lower+V_a_lower * \
        (z_interior_lower+r*a_interior_lower-c_lower)+(-the*tf.math.log(z_interior_lower)+sig2/2)*z_interior_lower*V_z_lower + sig2*z_interior_lower**2/2*V_zz_lower



    # Loss term #2: boundary condition
    
    fitted_boundary_NBC = model(tf.stack([a_NBC[:,0], z_NBC[:,0]], axis=1))
    fitted_boundary_NBC_z = tf.gradients(
        fitted_boundary_NBC, z_NBC)[0]
    L2 = tf.reduce_mean( tf.square(fitted_boundary_NBC_z ) )
    
    L2 += tf.reduce_mean(tf.square(concave_V))

    # Loss term #3: initial/terminal condition

    V_SC_lower = model(tf.stack([a_SC_lower[:,0], z_SC_lower[:,0]], axis=1))
    
    V_SC_lower_a = tf.gradients(V_SC_lower, a_SC_lower)[0]
    V_SC_lower_aa = tf.gradients(V_SC_lower_a, a_SC_lower)[0]
    V_SC_lower_z = tf.gradients(V_SC_lower, z_SC_lower)[0]
    V_SC_lower_zz = tf.gradients(V_SC_lower_z, z_SC_lower)[0]
    V_SC_lower_a = tf.maximum(1e-10 * tf.ones_like(V_SC_lower), V_SC_lower_a)
    
    c_SC_lower = u_deriv_inv(V_SC_lower_a)
    u_c_SC_lower = u(c_SC_lower) 
    
    diff_V_SC_lower = -rho*V_SC_lower+u_c_SC_lower+V_SC_lower_a * \
        (z_SC_lower+r*a_SC_lower-c_SC_lower)+(-the*tf.math.log(z_SC_lower)+sig2/2)*z_SC_lower*V_SC_lower_z + sig2*z_SC_lower**2/2*V_SC_lower_zz


    L3_lower = tf.reduce_mean( tf.square(diff_V_SC_lower ) )

    
    fitted_boundary_SC_upper = model(
        tf.stack([a_SC_upper[:, 0], z_SC_upper[:, 0]], axis=1))

    fitted_boundary_SC_upper_a = tf.gradients(
        fitted_boundary_SC_upper, a_SC_upper)[0]
    opt_boundary_SC_upper_a = tf.maximum(fitted_boundary_SC_upper_a - u_deriv(
        z_SC_upper+r*a_SC_upper), tf.zeros_like(fitted_boundary_SC_upper))
    
    L3_upper = tf.reduce_mean(tf.square(opt_boundary_SC_upper_a))
    
    
    weight_lower = tf.square(z_interior_lower-1) + 1
    weight_sum_lower = tf.reduce_sum(weight_lower,axis=0,keepdims=True)
    
    weighted_lower = weight_lower/weight_sum_lower

    
    
    L3 = L3_lower + L3_upper +tf.reduce_sum( tf.square(diff_V_lower) * weighted_lower)
    
    return L1, L2, L3
    # return L1, L2
    

# Set up network

# initialize DGM model (last input: space dimension = 1)
model = DGM2.DCGM3Net(X_low, X_high, nodes_per_layer, num_layers_FFNN,num_layers_RNN, 2, activation_FFNN)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
# t_interior_tnsr = tf.placeholder(tf.float32, [None,1])
a_interior_tnsr = tf.placeholder(tf.float32, [None,1])
z_interior_tnsr = tf.placeholder(tf.float32, [None,1])
a_interior_lower_tnsr = tf.placeholder(tf.float32, [None,1])
z_interior_lower_tnsr = tf.placeholder(tf.float32, [None,1])
a_NBC_tnsr = tf.placeholder(tf.float32, [None,1])
z_NBC_tnsr = tf.placeholder(tf.float32, [None,1])
a_SC_upper_tnsr = tf.placeholder(tf.float32, [None,1])
z_SC_upper_tnsr = tf.placeholder(tf.float32, [None,1])
a_SC_lower_tnsr = tf.placeholder(tf.float32, [None, 1])
z_SC_lower_tnsr = tf.placeholder(tf.float32, [None, 1])


# loss 
L1_tnsr, L2_tnsr, L3_tnsr = loss(
    model, a_interior_tnsr, z_interior_tnsr, a_interior_lower_tnsr, z_interior_lower_tnsr, a_NBC_tnsr, z_NBC_tnsr,a_SC_upper_tnsr,z_SC_upper_tnsr,a_SC_lower_tnsr,z_SC_lower_tnsr)
loss_tnsr = L1_tnsr + L2_tnsr + weight * L3_tnsr

# value function

V = model(tf.stack([a_interior_tnsr[:, 0], z_interior_tnsr[:, 0]], axis=1))

V_a = tf.gradients(V, a_interior_tnsr)[0]
V_aa = tf.gradients(V_a, a_interior_tnsr)[0]
V_z = tf.gradients(V, z_interior_tnsr)[0]
V_zz = tf.gradients(V_a, z_interior_tnsr)[0]

# optimal control computed numerically from fitted value function 
def control_c(V):
    
    V_a = tf.gradients(V, a_interior_tnsr)[0]
        
    c = u_deriv_inv(V_a)
    
    return c



numerical_c = control_c(V)

def Loss_V(V):
    
    V_a = tf.gradients(V, a_interior_tnsr)[0]
    V_aa = tf.gradients(V_a, a_interior_tnsr)[0]
    V_z = tf.gradients(V, z_interior_tnsr)[0]
    V_zz = tf.gradients(V_a, z_interior_tnsr)[0]
    V_a = tf.maximum(1e-10 * tf.ones_like(V), V_a)
    c = u_deriv_inv(V_a)

    Loss = -rho*V+u(c)+V_a * \
        (z_interior_tnsr+r*a_interior_tnsr-c)+(-the*tf.math.log(z_interior_tnsr)+sig2/2)*z_interior_tnsr*V_z + sig2*z_interior_tnsr**2/2*V_zz
    
    loss = tf.abs(Loss)
    loss_log10 = tf.math.log(tf.abs(Loss))/tf.log(tf.constant(10, dtype=tf.float32))
    
    return loss_log10, loss

Loss_log10, Loss = Loss_V(V)

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

    n_plot=100

    a_interior, z_interior, a_interior_lower, z_interior_lower, a_NBC, z_NBC, a_SC_upper, z_SC_upper, a_SC_lower, z_SC_lower = sampler(nSim_interior, nSim_boundary)
    
    if i%4==0 and i>2000:
        aspace = np.linspace(-0.02, 1, n_plot)
        zspace = np.linspace(zmean*0.8, zmean*1.2, n_plot)
        A, Z = np.meshgrid(aspace, zspace, indexing = 'ij')
        Xgrid = np.vstack([A.flatten(), Z.flatten()]).T

        a_interior = Xgrid[:,0:1]
        z_interior = Xgrid[:,1:2]
        
    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        loss,L1,L2,L3,_ = sess.run([loss_tnsr, L1_tnsr, L2_tnsr, L3_tnsr, optimizer],
                                   feed_dict={a_interior_tnsr: a_interior, z_interior_tnsr: z_interior, a_interior_lower_tnsr: a_interior_lower, z_interior_lower_tnsr: z_interior_lower, a_NBC_tnsr: a_NBC, z_NBC_tnsr: z_NBC, a_SC_upper_tnsr: a_SC_upper, z_SC_upper_tnsr: z_SC_upper, a_SC_lower_tnsr: a_SC_lower, z_SC_lower_tnsr: z_SC_lower})
        loss_list.append(loss)
    
    fitted_Loss_log10, fitted_Loss = sess.run([Loss_log10, Loss], feed_dict={
                        a_interior_tnsr: a_interior, z_interior_tnsr: z_interior})
    averageL2 = np.mean(np.square(fitted_Loss))

    if loss>1e-5:
        if i%100==0:

            print("{:7d}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}".format(int(i), loss,L1,L2,L3, averageL2 ))
    else:
        if i%10==0:

            print("{:7d}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}".format(int(i), loss,L1,L2,L3, averageL2))        
            
    if i%4000==0:
        
        aspace = np.linspace(-0.02, 4, n_plot)
        zspace = np.linspace(zmean*0.8, zmean*1.2, n_plot)
        A, Z = np.meshgrid(aspace, zspace)
        Xgrid = np.vstack([A.flatten(), Z.flatten()]).T

        # simulate process at current t 


        fitted_Va = sess.run([V_a], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

        fitted_V = sess.run([V], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

        fitted_c = sess.run([numerical_c], feed_dict={a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

        fitted_Vaa = sess.run([V_aa], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
        fitted_Vz = sess.run([V_z], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
        fitted_Vzz = sess.run([V_zz], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

        fitted_Loss_log10, fitted_Loss = sess.run([Loss_log10, Loss], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})

        average = np.mean(fitted_Loss.reshape(n_plot, n_plot))
        averageL2 = np.mean(np.square(fitted_Loss).reshape(n_plot, n_plot))
        weight = penalty*(Z-(zmean+0.05))**2+1
        sum_weight = np.sum(weight)
        weighted_average = np.sum(fitted_Loss.reshape(n_plot, n_plot)*( weight/ sum_weight))
        weighted_averageL2 = np.sum(np.square(fitted_Loss).reshape(n_plot, n_plot)*( weight/ sum_weight))
        
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(A, Z, np.zeros_like(A), color='red' , rcount=10, ccount=10)
        surf = ax.plot_surface(A, Z, fitted_Loss_log10.reshape(n_plot, n_plot), cmap='viridis')
        ax.plot_wireframe(A, Z, -5*np.ones_like(A), color='purple' , rcount=10, ccount=10)
        fig.colorbar(surf, ax=ax)
        ax.view_init(35, 35)
        ax.set_xlabel('$a$')
        ax.set_ylabel('$z$')
        ax.set_zlim(-10,5)

        ax.set_title('Loss: L1Average={}, L1Weighted average={}, L2Average={}, L2Weighted average={}'.format(average, weighted_average, averageL2, weighted_averageL2))

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
        # plt.plot(Z[:, 0], Moll_Va[:,0], label='FDM Solution', color='blue', linestyle=":")
        plt.xlabel('$z$')
        plt.legend()
        plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_VaSlice_{}.png'.format(i),bbox_inches='tight')
        plt.close('all')


        aspace = np.linspace(-0.02, 1, n_plot)
        zspace = np.linspace(zmean*0.8, zmean*1.2, n_plot)
        A, Z = np.meshgrid(aspace, zspace)
        Xgrid = np.vstack([A.flatten(), Z.flatten()]).T

        # simulate process at current t 


        fitted_Va = sess.run([V_a], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

        fitted_V = sess.run([V], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

        fitted_c = sess.run([numerical_c], feed_dict={a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

        fitted_Vaa = sess.run([V_aa], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
        fitted_Vz = sess.run([V_z], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
        fitted_Vzz = sess.run([V_zz], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
        
        fitted_Loss_log10, fitted_Loss = sess.run([Loss_log10, Loss], feed_dict={
                            a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})

        average = np.mean(fitted_Loss.reshape(n_plot, n_plot))
        weight = penalty*(Z-1)**2+1
        sum_weight = np.sum(weight)
        weighted_average = np.sum(fitted_Loss.reshape(n_plot, n_plot)*( weight/ sum_weight))
        

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(A, Z, np.zeros_like(A), color='red' , rcount=10, ccount=10)
        ax.plot_wireframe(A, Z, -5*np.ones_like(A), color='purple' , rcount=10, ccount=10)
        surf = ax.plot_surface(A, Z, fitted_Loss_log10.reshape(n_plot, n_plot), cmap='viridis')
        fig.colorbar(surf, ax=ax)
        ax.view_init(35, 35)
        ax.set_xlabel('$a$')
        ax.set_ylabel('$z$')
        ax.set_zlim(-10,5)
        ax.set_title('Loss: L1Average={}, L1Weighted average={}'.format(average, weighted_average))

        plt.savefig( './Figure/' +savefolder+ '/' + saveName + '_LossV_Shrink_{}.png'.format(i),bbox_inches='tight')
        


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
n_plot=600
aspace = np.linspace(-0.02, 4, n_plot)
zspace = np.linspace(zmean*0.8, zmean*1.2, n_plot)
A, Z = np.meshgrid(aspace, zspace)
Xgrid = np.vstack([A.flatten(), Z.flatten()]).T

# simulate process at current t 

fitted_V = sess.run([V], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
fitted_c = sess.run([numerical_c], feed_dict={a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
fitted_Va = sess.run([V_a], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
fitted_Vaa = sess.run([V_aa], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
fitted_Vz = sess.run([V_z], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]
fitted_Vzz = sess.run([V_zz], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})[0]

fitted_Loss_log10, fitted_Loss = sess.run([Loss_log10, Loss], feed_dict={
                    a_interior_tnsr: Xgrid[:,0:1], z_interior_tnsr: Xgrid[:,1:2]})

average = np.mean(fitted_Loss.reshape(n_plot, n_plot))
weight = penalty*(Z-(zmean+0.05))**2+1
sum_weight = np.sum(weight)
weighted_average = np.sum(fitted_Loss.reshape(n_plot, n_plot)*( weight/ sum_weight))
        


fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(A, Z, np.zeros_like(A), color='red' , rcount=10, ccount=10)
surf = ax.plot_surface(A, Z, fitted_Loss_log10.reshape(n_plot, n_plot), cmap='viridis')
fig.colorbar(surf, ax=ax)# ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
ax.set_zlim(-10,5)
# ax.set_zlabel('$v(a,z)$')
ax.set_title('Loss: L1Average={}, L1Weighted average={}'.format(average, weighted_average))

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

Moll_Va = pd.read_csv("./MollData/Va_Upwind.csv", header = None)
Moll_Va = np.array(Moll_Va)

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
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
        label='State Constraint', color='red')
plt.plot(Z[:, 0], Moll_Va[:,0],
        label='FDM Solution', color='blue', linestyle=":")
plt.xlabel('$z$')
# plt.ylim(0.75, 1.10)
plt.legend()
plt.savefig('./Figure/' +savefolder+ '/' + saveName + '_VaSlice.png', bbox_inches='tight')