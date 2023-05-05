# SCRIPT FOR SOLVING THE MERTON PROBLEM

#%% import needed packages

import DGM2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse



#%% Parameters 

parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--xi", type=float)
args, unknown = parser.parse_known_args()

# Sannikov problem parameters 
r = 0.1
sigma = 1
xi = args.xi
# Solution parameters (domain on which to solve PDE)
X_low = 0.0  # wealth lower bound
X_high = 10          # wealth upper bound

# neural network parameters
num_layers = 3
nodes_per_layer = 50
starting_learning_rate = 0.001

# Training parameters
sampling_stages  = 10000   # number of times to resample new time-space domain points
steps_per_sample = 10    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = 1000
nSim_boundary = 100

# multipliers for oversampling i.e. draw X from [X_low - X_oversample, X_high + X_oversample]
X_oversample = 0.5
t_oversample = 0.5

# Plot options
n_plot = 10000  # Points on plot grid for each dimension

# Save options
saveOutput = False
savefolder = '/mercury_noretirement_robust_wider_xi={}/'.format(xi)
saveName   = 'SannikovProblem2'
saveFigure = False
figureName = 'SannikovProblem2_xi={}'.format(xi)

#%% Analytical Solution

# market price of risk

def u(c):
    return c**(1/2)


def u_deriv(c):
    return c**(-1/2)/2


# @tf.function
def u_deriv_inv(c):
    return c**(-2)/4


def F0(w):
    return -w**2


def hh(a):
    return 1/2*a**2 + 0.4 * a


def gamma(a):
    return a+0.4



#%% Sampling function - randomly sample time-space pairs

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
    X_interior = np.random.uniform(low=X_low, high=X_high, size=[nSim_interior, 1])
#    X_interior = np.random.uniform(low=X_low - X_oversample, high=X_high + X_oversample, size=[nSim_interior, 1])
#    X_interior = np.random.uniform(low=X_low * X_multiplier_low, high=X_high * X_multiplier_high, size=[nSim_interior, 1])

    # Sampler #2: spatial boundary
        # no spatial boundary condition for this problem
    
    X_boundary = X_low + (X_high - X_low) * np.random.binomial(1, 0.5, size = (nSim_boundary,1))
    # Sampler #3: initial/terminal condition
    # t_terminal = T * np.ones((nSim_terminal, 1))
#    X_terminal = np.random.uniform(low=X_low - X_oversample, high=X_high + X_oversample, size = [nSim_terminal, 1])
    # X_terminal = np.random.uniform(low=X_low - 0.5*(X_high-X_low), high=X_high + 0.5*(X_high-X_low), size = [nSim_terminal, 1])
#    X_terminal = np.random.uniform(low=X_low * X_multiplier_low, high=X_high * X_multiplier_high, size = [nSim_terminal, 1])
    
    # return t_interior, X_interior, t_terminal, X_terminal
    return X_interior, X_boundary
    # return X_interior.astype(np.float32), X_boundary.astype(np.float32)

#%% Loss function for Merton Problem PDE

def loss(model, X_interior, X_boundary):
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
    V = model(X_interior)
    V_x = tf.gradients(V, X_interior)[0]
    V_xx = tf.gradients(V_x, X_interior)[0]
    
    # a = tf.where(V_x+r*sigma**2*V_xx >= 0, tf.zeros((nSim_interior, 1)), -(1+0.4*V_x+sigma**2*r*0.4*V_xx)/(V_x+r*sigma**2*V_xx))


    # index3 = tf.cast( (V_x+r*sigma ** 2 * V_xx - V_x**2 * sigma**2/xi) > 0 , tf.float32)
    # index4 = tf.cast(tf.math.logical_and(
    #     V_x < 0, 1 + 0.4 * (V_x+r*sigma ** 2 * V_xx - V_x**2 * sigma**2/xi)<= 0) , tf.float32)
    
    # a = (-1/(V_x+r*sigma**2*V_xx) - 0.4) * \
    #     index1 + (tf.zeros_like(V)) * index2 + (-1/(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi) - 0.4)*index3 + (tf.zeros_like(V)) * index4

    # a = tf.where( V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi >= 0, tf.zeros_like(V), -1/(V_x+r*sigma**2*V_xx - V_x**2 * sigma**2/xi ) - 0.4)

    index1 = tf.cast(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi>=0, tf.float32)
    index2 = tf.cast(tf.math.logical_and(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi<0, 1 + 0.4*(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi) >0 ) , tf.float32)
    index3 = tf.cast(tf.math.logical_and(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi<0, 1 + 0.4*(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi) <0 ) , tf.float32)
    
    a = tf.zeros_like(V) * index1 + (-1/(V_x+r*sigma**2*V_xx - V_x**2 * sigma**2/xi ) - 0.4)*index2 + tf.zeros_like(V) * index3
    
    h = -sigma* V_x /xi * (a+0.4)
    # a = -(1+0.4*v_w+sigma**2*r*0.4*v_ww)/(v_w+r*sigma**2*v_ww)
    h_a = hh(a)
    c = tf.where(V_x < 0, (V_x/2)**2, tf.zeros((nSim_interior, 1)))
    u_c = u(c)
    gamma_a = gamma(a)
    
    diff_V = -r * V + r*(a-c + xi/2*h**2) + r * V_x * (X_interior-u_c+h_a + (a+0.4)*sigma*h) + V_xx/2*r**2*sigma**2*gamma_a**2

    # compute average L2-norm of differential operator
    L1 = tf.reduce_mean(tf.square(diff_V)) 
    
    # Loss term #2: boundary condition
        # no boundary condition for this problem
    
    fitted_boundary = model(X_boundary)
    target_boundary = F0(X_boundary)
    
    
    # L2 = tf.reduce_mean( tf.square(fitted_boundary - target_boundary) )
    L2 = tf.reduce_mean( tf.zeros_like(fitted_boundary - target_boundary) )
    

    # Loss term #3: initial/terminal condition
    # target_terminal = exponential_utility(X_terminal)
    # fitted_terminal = model(t_terminal, X_terminal)
    
    # L3 = tf.reduce_mean( tf.square(fitted_terminal - target_terminal) )

    return L1, L2
    

#%% Set up network

# initialize DGM model (last input: space dimension = 1)
model = DGM2.DGMNet(nodes_per_layer, num_layers, 1)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
# t_interior_tnsr = tf.placeholder(tf.float32, [None,1])
X_interior_tnsr = tf.placeholder(tf.float32, [None,1])
# t_terminal_tnsr = tf.placeholder(tf.float32, [None,1])
X_boundary_tnsr = tf.placeholder(tf.float32, [None,1])

# loss 
L1_tnsr, L2_tnsr = loss(model, X_interior_tnsr, X_boundary_tnsr)
loss_tnsr = L1_tnsr + L2_tnsr

# value function
V = model(X_interior_tnsr)

# optimal control computed numerically from fitted value function 
def control_a(V):
    
    V_x = tf.gradients(V, X_interior_tnsr)[0]
    V_xx = tf.gradients(V_x, X_interior_tnsr)[0]
    
    index1 = tf.cast(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi>=0, tf.float32)
    index2 = tf.cast(tf.math.logical_and(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi<0, 1 + 0.4*(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi) >0 ) , tf.float32)
    index3 = tf.cast(tf.math.logical_and(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi<0, 1 + 0.4*(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi) <0 ) , tf.float32)
    
    a = tf.zeros_like(V) * index1 + (-1/(V_x+r*sigma**2*V_xx - V_x**2 * sigma**2/xi ) - 0.4)*index2 + tf.zeros_like(V) * index3
    
    # h = -sigma* V_x /xi * (a+0.4)

    return a

def control_h(V):
    
    V_x = tf.gradients(V, X_interior_tnsr)[0]
    V_xx = tf.gradients(V_x, X_interior_tnsr)[0]
    
    index1 = tf.cast(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi>=0, tf.float32)
    index2 = tf.cast(tf.math.logical_and(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi<0, 1 + 0.4*(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi) >0 ) , tf.float32)
    index3 = tf.cast(tf.math.logical_and(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi<0, 1 + 0.4*(V_x+r*sigma **2 * V_xx - V_x**2 * sigma**2/xi) <0 ) , tf.float32)
    
    a = tf.zeros_like(V) * index1 + (-1/(V_x+r*sigma**2*V_xx - V_x**2 * sigma**2/xi ) - 0.4)*index2 + tf.zeros_like(V) * index3
    
    h = -sigma* V_x /xi * (a+0.4)

    return h


def control_c(V):
    # length = V.shape[0]
    V_x = tf.gradients(V, X_interior_tnsr)[0]
    V_xx = tf.gradients(V_x, X_interior_tnsr)[0]
    
    c = tf.where(V_x < 0, (V_x/2)**2, tf.zeros(tf.shape(V)))
    return c

numerical_a = control_a(V)
numerical_c = control_c(V)
numerical_h = control_h(V)

# set optimizer - NOTE THIS IS DIFFERENT FROM OTHER APPLICATIONS!
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,100000, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

# initialize variables
init_op = tf.global_variables_initializer()

# open session
sess = tf.Session()
sess.run(init_op)

#%% Train network
# initialize loss per training
loss_list = []

# for each sampling stage
for i in range(sampling_stages):
    
    # sample uniformly from the required regions
    X_interior, X_boundary = sampler(nSim_interior, nSim_boundary)
    
    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        loss,L1,L2,_ = sess.run([loss_tnsr, L1_tnsr, L2_tnsr, optimizer],
                                feed_dict = {X_interior_tnsr:X_interior, X_boundary_tnsr:X_boundary})
        loss_list.append(loss)
    
    print(loss, L1, L2, i)
os.makedirs('./SavedNets/' + savefolder,exist_ok=True)

# save outout
# if saveOutput:
#     saver = tf.train.Saver()
#     saver.save(sess, './SavedNets/' + savefolder + saveName)
       
#%% Plot value function results

saver = tf.train.Saver()

os.makedirs('./SavedNets/' + savefolder,exist_ok=True)

saver.save(sess, './SavedNets/' + savefolder + saveName)
# saver.restore(sess, './SavedNets' + savefolder + saveName)
# saver.restore(sess, "C:/Users/33678/InequalityEcon/SavedNets/robust_xi=5/SannikovProblem2")
    
# LaTeX rendering for text in plots
plt.style.use('classic')
plt.rcParams["savefig.bbox"] = "tight"
# plt.rcParams["figure.figsize"] = (10,8)
# plt.rcParams["figure.dpi"] = 500
plt.rcParams["font.size"] = 16
# plt.rcParams["legend.frameon"] = True
# plt.rcParams["lines.linewidth"] = 5

figwidth = 10

# figure options
plt.figure()
plt.figure(figsize = (12,10))

# time values at which to examine density
# valueTimes = [t_low, T/3, 2*T/3, T]
fig, axs = plt.subplot_mosaic(

[["left column", "right top"],
["left column", "right mid"],
["left column", "right h"],
 ["left column", "right down"]], figsize=(4 * figwidth, 26)

)

# vector of t and S values for plotting
X_plot = np.linspace(X_low, X_high, n_plot)
X_plot = X_plot.reshape(-1,1)


# simulate process at current t 

fitted_V = sess.run([V], feed_dict= {X_interior_tnsr:X_plot})[0]
fitted_a = sess.run([numerical_a], feed_dict= {X_interior_tnsr:X_plot})[0]
fitted_c = sess.run([numerical_c], feed_dict= {X_interior_tnsr:X_plot})[0]
fitted_h = sess.run([numerical_h], feed_dict= {X_interior_tnsr:X_plot})[0]
B_W = r*(X_plot-fitted_c**(1/2)+fitted_a**2/2+2 *
         fitted_a/5+(fitted_a+0.4)*sigma*fitted_h)
B_W_True = r*(X_plot-fitted_c**(1/2)+fitted_a**2/2+2 *
         fitted_a/5)
fitted_drift = B_W
fitted_drift_true = B_W_True

axs["left column"].plot(X_plot, fitted_V, color = 'red')
axs["left column"].set_ylim(-1,0.5)
axs["left column"].plot(X_plot, F0(X_plot), color = 'black')
axs["left column"].set_title("Profit $F(W)$")
axs["left column"].grid(linestyle=':')

axs["right top"].plot(X_plot, fitted_a, color = 'red')
axs["right top"].set_ylim(0,0.8)
axs["right top"].set_title("Effort $\\alpha(W)$")
axs["right top"].grid(linestyle=':')

axs["right mid"].plot(X_plot, fitted_c, color = 'red')
axs["right mid"].set_ylim(0, 0.4)
axs["right mid"].set_title("Consumption $\\pi(W)$")
axs["right mid"].grid(linestyle=':')

# axs["left down"].plot(X_plot, fitted_drift_true, color = 'red')
# # axs["right down"].set_ylim(0, 0.1)
# axs["left down"].set_title("True Drift of $W$")
# axs["left down"].grid(linestyle=':')

axs["right h"].plot(X_plot, fitted_h, color = 'red')
axs["right h"].set_ylim(0,0.6)
axs["right h"].set_title("Distortion $h(W)$")
axs["right h"].grid(linestyle=':')

axs["right down"].plot(X_plot, fitted_drift, color = 'red')
axs["right down"].set_ylim(0, 0.1)
axs["right down"].set_title("Distorted Drift of $W$")
axs["right down"].grid(linestyle=':')

os.makedirs('./Figure/' +savefolder,exist_ok=True)

plt.savefig('./Figure/' +savefolder+  figureName + '_All.pdf')

Fitted_matrix = np.zeros((n_plot,6))
Fitted_matrix[:,:1] = X_plot
Fitted_matrix[:,1:2] = fitted_V
Fitted_matrix[:,2:3] = fitted_a
Fitted_matrix[:,3:4] = fitted_c
Fitted_matrix[:,4:5] = fitted_h
Fitted_matrix[:,5:6] = fitted_drift
pd.DataFrame(Fitted_matrix).to_csv('./Figure/' +savefolder+ figureName +'_All.csv',header=False,index=False)    

if saveFigure:
    plt.savefig('./Figure/' +savefolder+ '/' + saveName  + '_All.pdf')



#%%
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
ax.semilogy(range(len(loss_list)), loss_list, 'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi^{n_{epoch}}$')

plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_LossList.pdf')

if saveFigure:
    plt.savefig(figureName + '_LossList.pdf')   