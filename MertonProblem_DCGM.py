# SCRIPT FOR SOLVING THE MERTON PROBLEM

#%% import needed packages
import DGM
import DGM2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.integrate  import solve_ivp
import pandas as pd

parser = argparse.ArgumentParser(description="slope")
parser.add_argument("--LearningRate", type=float)
parser.add_argument("--num_layers_FFNN", type=int)
parser.add_argument("--activation_FFNN", type=str, default = "tanh")
parser.add_argument("--num_layers_RNN", type=int)
parser.add_argument("--nodes_per_layer", type=int)
parser.add_argument("--sampling_stages", type=int)
parser.add_argument("--steps_per_sample", type=int)
parser.add_argument("--nSim_interior", type=int)
parser.add_argument("--nSim_boundary", type=int)
# args = parser.parse_args()
args, unknown = parser.parse_known_args()

# LaTeX rendering for text in plots
plt.style.use('classic')
plt.rcParams["savefig.bbox"] = "tight"
# plt.rcParams["figure.figsize"] = (10,8)
# plt.rcParams["figure.dpi"] = 500
plt.rcParams["font.size"] = 20
# plt.rcParams["legend.frameon"] = True
plt.rcParams["lines.linewidth"] = 2.5

#%% Parameters 

# Sannikov problem parameters 
r = 0.05      # risk-free rate
mu = 0.2      # asset drift
sigma = 0.25  # asset volatility
rho = 0.05

kappa = 1/rho * (np.log(rho)  + r/rho + (mu-r)**2/(2 * rho * sigma**2)-1)

eps = 1e-5


# Solution parameters (domain on which to solve PDE)
X_low = 0.2  # wealth lower bound
X_high = 4           # wealth upper bound


# neural network parameters

num_layers_FFNN = args.num_layers_FFNN
num_layers_RNN = args.num_layers_RNN
nodes_per_layer = args.nodes_per_layer
starting_learning_rate = args.LearningRate

activation_FFNN = args.activation_FFNN
# Training parameters
sampling_stages  = args.sampling_stages   # number of times to resample new time-space domain points
steps_per_sample = args.steps_per_sample    # number of SGD steps to take before re-sampling

nSim_interior = args.nSim_interior
nSim_boundary = args.nSim_boundary


# multipliers for oversampling i.e. draw X from [X_low - X_oversample, X_high + X_oversample]
X_oversample = 0.5
t_oversample = 0.5

# Plot options
n_plot = 600  # Points on plot grid for each dimension

# sloperange = np.linspace(args.lowerslope, args.upperslope, args.num)
# idd = args.id
# backup = args.backup

# Save options
saveOutput = False

savefolder = 'Merton/num_layers_FFNN_{}_activation_FFNN_{}_num_layers_RNN_{}_nodes_per_layer_{}/sampling_stages_{}_steps_per_sample_{}_learningrate_{}/nSim_interior_{}_nSim_boundary_{}/'.format(num_layers_FFNN, activation_FFNN, num_layers_RNN, nodes_per_layer, sampling_stages, steps_per_sample, starting_learning_rate, nSim_interior, nSim_boundary)


saveName   = 'Merton'
saveFigure = False
figureName = saveName
#%% Analytical Solution

os.makedirs('./SavedNets/' + savefolder  ,exist_ok=True)
os.makedirs('./Figure/' + savefolder  ,exist_ok=True)
# market price of risk
def V_Solution(x):
    
    return 1/rho * np.log(x) + kappa

def c_Solution(x):
    
    return rho * x

def w_Solution(x):
    
    return (mu - r )/(sigma**2) * np.ones_like(x)

def Drift_Solution(x):
    
    return r*x + (mu-r)/sigma**2 * x *(mu-r) - rho*x


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
    
    X_boundary = X_low * np.ones((nSim_boundary,1))
    # Sampler #3: initial/terminal condition
    # t_terminal = T * np.ones((nSim_terminal, 1))
#    X_terminal = np.random.uniform(low=X_low - X_oversample, high=X_high + X_oversample, size = [nSim_terminal, 1])
    # X_terminal = np.random.uniform(low=X_low - 0.5*(X_high-X_low), high=X_high + 0.5*(X_high-X_low), size = [nSim_terminal, 1])
#    X_terminal = np.random.uniform(low=X_low * X_multiplier_low, high=X_high * X_multiplier_high, size = [nSim_terminal, 1])
    
    X_far = X_high * np.ones((nSim_boundary,1))
    # return t_interior, X_interior, t_terminal, X_terminal
    return X_interior, X_boundary, X_far
    # return X_interior.astype(np.float32), X_boundary.astype(np.float32)

#%% Loss function for Merton Problem PDE

def loss1(model, X_interior, X_boundary, X_far):
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

    
    diff_V =  V - 1/rho * tf.log(X_interior)

    
    L1 = tf.reduce_mean(tf.square(diff_V))  

    L2_3 = tf.reduce_mean(tf.zeros_like(diff_V))
    
    L2 = L2_3


    return L1, L2
    
    
def loss2(model, X_interior, X_boundary, X_far):
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
    
    V_x = tf.maximum(V_x,eps * tf.ones_like(V))
    c = tf.where(V_x > 0, 1/V_x , eps * tf.ones_like(V))
    u_c = tf.log(c)
    
    w = - (mu-r)/sigma**2 * V_x / (X_interior * V_xx)
    
    
    diff_V =  u_c + V_x * (r*X_interior + w * X_interior*(mu-r) - c) + V_xx/2 * sigma**2 * w**2 * X_interior**2   -rho * V

    concave_V = tf.maximum(V_xx, -10 * eps * tf.ones_like(V))
    
    L1 = tf.reduce_mean(tf.square(diff_V))  + tf.reduce_mean(tf.square(concave_V))

    L2_3 = tf.reduce_mean(tf.zeros_like(diff_V))
    
    L2 = L2_3


    return L1, L2
    

#%% Set up network

# initialize DGM model (last input: space dimension = 1)
# model = DGM2.DGMNet(nodes_per_layer, num_layers, 1)
model = DGM2.DCGM2Net(X_low, X_high, nodes_per_layer, num_layers_FFNN,num_layers_RNN, 1, activation_FFNN)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
# t_interior_tnsr = tf.placeholder(tf.float32, [None,1])
X_interior_tnsr = tf.placeholder(tf.float32, [None,1])
# t_terminal_tnsr = tf.placeholder(tf.float32, [None,1])
X_boundary_tnsr = tf.placeholder(tf.float32, [None,1])
X_far_tnsr = tf.placeholder(tf.float32, [None,1])

# loss 
L1_tnsr, L2_tnsr = loss1(model, X_interior_tnsr, X_boundary_tnsr, X_far_tnsr)
loss1_tnsr = L1_tnsr +  L2_tnsr

L1_tnsr, L2_tnsr = loss2(model, X_interior_tnsr, X_boundary_tnsr, X_far_tnsr)
loss2_tnsr = L1_tnsr +  L2_tnsr

# value function
V = model(X_interior_tnsr)
V_x = tf.gradients(V, X_interior_tnsr)[0]
V_xx = tf.gradients(V_x, X_interior_tnsr)[0]

    
# optimal control computed numerically from fitted value function 
def control_c(V,V_x,V_xx):
    
    # V_xx = tf.gradients(V_x, X_interior_tnsr)[0]
    
    c = tf.where(V_x > 0, 1/V_x , eps * tf.ones_like(V))

    
    return c

def control_w(V,V_x,V_xx):

    w = - (mu-r)/sigma**2 * V_x / (X_interior_tnsr * V_xx)

    return w

numerical_w = control_w(V,V_x,V_xx)
numerical_c = control_c(V,V_x,V_xx)

# set optimizer - NOTE THIS IS DIFFERENT FROM OTHER APPLICATIONS!
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,10000, 0.95, staircase=True)
optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss1_tnsr, global_step=global_step)
optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss2_tnsr, global_step=global_step)

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
    X_interior, X_boundary, X_far = sampler(nSim_interior, nSim_boundary)
    
    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        
        if i<5000:
            loss,L1,L2,_ = sess.run([loss1_tnsr, L1_tnsr, L2_tnsr, optimizer1],
                                    feed_dict = {X_interior_tnsr:X_interior, X_boundary_tnsr:X_boundary, X_far_tnsr:X_far})
            loss_list.append(loss)
        else:
            loss,L1,L2,_ = sess.run([loss2_tnsr, L1_tnsr, L2_tnsr, optimizer2],
                                    feed_dict = {X_interior_tnsr:X_interior, X_boundary_tnsr:X_boundary, X_far_tnsr:X_far})
            loss_list.append(loss)
            
    print(loss, L1, L2, i)

    if i%2000==0:


        # vector of t and S values for plotting
        X_plot = np.linspace(X_low, X_high, n_plot)
        X_plot = X_plot.reshape(-1,1)


        # simulate process at current t 

        fitted_V = sess.run([V], feed_dict= {X_interior_tnsr:X_plot})[0]
        fitted_w = sess.run([numerical_w], feed_dict= {X_interior_tnsr:X_plot})[0]
        fitted_c = sess.run([numerical_c], feed_dict= {X_interior_tnsr:X_plot})[0]
        fitted_drift = (r * X_plot + fitted_w*X_plot * (mu-r) - fitted_c)
         

        figwidth = 10
        fig, axs = plt.subplot_mosaic(
        [["left column", "right top"],
        ["left column", "right mid"],
        ["left column", "right down"]], figsize=(4 * figwidth, 2 * figwidth)
        )


        axs["left column"].plot(X_plot, fitted_V, color = 'red', label = 'Neural Network Solution')
        axs["left column"].plot(X_plot, V_Solution(X_plot).reshape(-1,1), color = 'blue', linestyle='--',label='Analytical Solution')
        # axs["left column"].set_ylim(-1,0.15)
        # axs["left column"].plot(X_plotF0, F0(X_plotF0), color = 'black')
        axs["left column"].set_xlim(X_low,X_high)
        axs["left column"].set_title("Value Function $V(x)$")
        axs["left column"].grid(linestyle=':')
        axs["left column"].legend()

        axs["right top"].plot(X_plot, fitted_w, color = 'red', label = 'Neural Network Solution')
        axs["right top"].plot(X_plot, w_Solution(X_plot).reshape(-1,1), color = 'blue', linestyle='--',label='Analytical Solution')
        # axs["right top"].set_ylim(0,1)
        axs["right top"].set_xlim(X_low,X_high)
        axs["right top"].set_title("Investment Ratio $\\omega(x)$")
        axs["right top"].grid(linestyle=':')

        axs["right mid"].plot(X_plot, fitted_c, color = 'red', label = 'Neural Network Solution')
        axs["right mid"].plot(X_plot, c_Solution(X_plot).reshape(-1,1), color = 'blue', linestyle='--',label='Analytical Solution')
        # axs["right mid"].set_ylim(0, 1)
        axs["right mid"].set_xlim(X_low,X_high)
        axs["right mid"].set_title("Consumption $\\pi(W)$")
        axs["right mid"].grid(linestyle=':')

        axs["right down"].plot(X_plot, fitted_drift, color = 'red', label = 'Neural Network Solution')
        axs["right down"].plot(X_plot, Drift_Solution(X_plot).reshape(-1,1), color = 'blue', linestyle='--',label='Analytical Solution')
        # axs["right down"].set_ylim(0, 0.1)
        axs["right down"].set_xlim(X_low,X_high)
        axs["right down"].set_title("Drift of $x$")
        axs["right down"].grid(linestyle=':')
        
        # plt.savefig(figureName + '_All.png')
        plt.savefig('./Figure/' +savefolder+ figureName + '_i_{}_All.png'.format(i))

        plt.close('all')        



        figwidth = 10
        fig, axs = plt.subplot_mosaic(
        [["left top", "right top"],
        ["left mid", "right mid"]], figsize=(3 * figwidth, 1.5 * figwidth)
        )


        axs["left top"].plot(X_plot, np.log(abs(fitted_V-V_Solution(X_plot).reshape(-1,1))), color = 'red')
        axs["left top"].set_title("Difference in Value Function $V(x)$")
        axs["left top"].set_xlim(X_low,X_high)
        axs["left top"].grid(linestyle=':')

        axs["left mid"].plot(X_plot, np.log(abs(fitted_w-w_Solution(X_plot).reshape(-1,1))), color = 'red')
        axs["left mid"].set_title("Difference in Effort $\\omega(x)$")
        axs["left mid"].grid(linestyle=':')
        axs["left mid"].set_xlim(X_low,X_high)

        axs["right top"].plot(X_plot, np.log(abs(fitted_c-c_Solution(X_plot).reshape(-1,1))), color = 'red')
        axs["right top"].set_title("Difference in Consumption $\\pi(x)$")
        axs["right top"].grid(linestyle=':')
        axs["right top"].set_xlim(X_low,X_high)

        axs["right mid"].plot(X_plot, np.log(abs(fitted_drift-Drift_Solution(X_plot).reshape(-1,1))), color = 'red')
        axs["right mid"].set_title("Difference in Drift of $x$")
        axs["right mid"].grid(linestyle=':')
        axs["right mid"].set_xlim(X_low,X_high)

        # plt.savefig(figureName + '_All.png')
        plt.savefig('./Figure/' +savefolder+ figureName + '_i_{}_Diff.png'.format(i))

        plt.close('all')        
        
# save outout
if saveOutput:
    saver = tf.train.Saver()
    saver.save(sess, './SavedNets/' + saveName)
       
#%% Plot value function results
saver = tf.train.Saver()


# saver.save(sess, './SavedNets/' + savefolder + saveName)

    


# interval = np.linspace(0.01,1,100000)
# interval = interval.reshape(-1,1)

# F = sess.run([G], feed_dict= {X_interior_tnsr:interval})[0]
# V = sess.run([G_x], feed_dict= {X_interior_tnsr:interval})[0]

# Fmin = tf.reduce_min(F)
# Vmin = tf.gather(V,tf.math.argmin(F))
# Wmin = tf.gather(interval,tf.math.argmin(F))
# tf.print(Fmin, Vmin, Wmin)


# simulate process at current t 

X_plot = np.linspace(X_low, X_high, n_plot)
X_plot = X_plot.reshape(-1,1)

fitted_V = sess.run([V], feed_dict= {X_interior_tnsr:X_plot})[0]
fitted_w = sess.run([numerical_w], feed_dict= {X_interior_tnsr:X_plot})[0]
fitted_c = sess.run([numerical_c], feed_dict= {X_interior_tnsr:X_plot})[0]
fitted_drift = (r * X_plot + fitted_w*X_plot * (mu-r) - fitted_c)
    

figwidth = 10
fig, axs = plt.subplot_mosaic(
[["left column", "right top"],
["left column", "right mid"],
["left column", "right down"]], figsize=(4 * figwidth, 2 * figwidth)
)


axs["left column"].plot(X_plot, fitted_V, color = 'red', label = 'Neural Network Solution')
axs["left column"].plot(X_plot, V_Solution(X_plot).reshape(-1,1), color = 'blue', linestyle='--',label='Analytical Solution')
# axs["left column"].set_ylim(-1,0.15)
# axs["left column"].plot(X_plotF0, F0(X_plotF0), color = 'black')
axs["left column"].set_xlim(X_low,X_high)
axs["left column"].set_title("Value Function $V(x)$")
axs["left column"].grid(linestyle=':')
axs["left column"].legend()

axs["right top"].plot(X_plot, fitted_w, color = 'red', label = 'Neural Network Solution')
axs["right top"].plot(X_plot, w_Solution(X_plot).reshape(-1,1), color = 'blue', linestyle='--',label='Analytical Solution')
# axs["right top"].set_ylim(0,1)
axs["right top"].set_xlim(X_low,X_high)
axs["right top"].set_title("Investment Ratio $\\omega(x)$")
axs["right top"].grid(linestyle=':')

axs["right mid"].plot(X_plot, fitted_c, color = 'red', label = 'Neural Network Solution')
axs["right mid"].plot(X_plot, c_Solution(X_plot).reshape(-1,1), color = 'blue', linestyle='--',label='Analytical Solution')
# axs["right mid"].set_ylim(0, 1)
axs["right mid"].set_xlim(X_low,X_high)
axs["right mid"].set_title("Consumption $\\pi(W)$")
axs["right mid"].grid(linestyle=':')

axs["right down"].plot(X_plot, fitted_drift, color = 'red', label = 'Neural Network Solution')
axs["right down"].plot(X_plot, Drift_Solution(X_plot).reshape(-1,1), color = 'blue', linestyle='--',label='Analytical Solution')
# axs["right down"].set_ylim(0, 0.1)
axs["right down"].set_xlim(X_low,X_high)
axs["right down"].set_title("Drift of $x$")
axs["right down"].grid(linestyle=':')

# plt.savefig(figureName + '_All.png')
plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_All.pdf')

plt.close('all')        



figwidth = 10
fig, axs = plt.subplot_mosaic(
[["left top", "right top"],
["left mid", "right mid"]], figsize=(3 * figwidth, 1.5 * figwidth)
)


axs["left top"].plot(X_plot, np.log(abs(fitted_V-V_Solution(X_plot).reshape(-1,1))), color = 'red')
axs["left top"].set_title("Difference in Value Function $V(x)$")
axs["left top"].set_xlim(X_low,X_high)
axs["left top"].grid(linestyle=':')

axs["left mid"].plot(X_plot, np.log(abs(fitted_w-w_Solution(X_plot).reshape(-1,1))), color = 'red')
axs["left mid"].set_title("Difference in Effort $\\omega(x)$")
axs["left mid"].grid(linestyle=':')
axs["left mid"].set_xlim(X_low,X_high)

axs["right top"].plot(X_plot, np.log(abs(fitted_c-c_Solution(X_plot).reshape(-1,1))), color = 'red')
axs["right top"].set_title("Difference in Consumption $\\pi(x)$")
axs["right top"].grid(linestyle=':')
axs["right top"].set_xlim(X_low,X_high)

axs["right mid"].plot(X_plot, np.log(abs(fitted_drift-Drift_Solution(X_plot).reshape(-1,1))), color = 'red')
axs["right mid"].set_title("Difference in Drift of $x$")
axs["right mid"].grid(linestyle=':')
axs["right mid"].set_xlim(X_low,X_high)


# plt.savefig(figureName + '_All.png')
plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_Diff.pdf')

plt.close('all')        
        

#%%
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.semilogy(range(len(loss_list)), loss_list, 'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi^{n_{epoch}}$')
plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_LossList.png')




Fitted_matrix = np.zeros((len(X_plot),16))
Fitted_matrix[:,:1] = X_plot
Fitted_matrix[:,1:2] = fitted_V
Fitted_matrix[:,2:3] = fitted_w
Fitted_matrix[:,3:4] = fitted_c
Fitted_matrix[:,5:6] = fitted_drift

Fitted_matrix[:,6:7] = V_Solution(X_plot).reshape(-1,1)
Fitted_matrix[:,7:8] = w_Solution(X_plot).reshape(-1,1)
Fitted_matrix[:,8:9] = c_Solution(X_plot).reshape(-1,1)
Fitted_matrix[:,10:11] = Drift_Solution(X_plot).reshape(-1,1)

Fitted_matrix[:,11:12] = np.log10(abs(fitted_V-V_Solution(X_plot).reshape(-1,1)))
Fitted_matrix[:,12:13] = np.log10(abs(fitted_w-w_Solution(X_plot).reshape(-1,1)))
Fitted_matrix[:,13:14] = np.log10(abs(fitted_c-c_Solution(X_plot).reshape(-1,1)))
Fitted_matrix[:,15:16] = np.log10(abs(fitted_drift-Drift_Solution(X_plot).reshape(-1,1)))


pd.DataFrame(Fitted_matrix).to_csv('./Figure/' +savefolder+ '/' +  figureName +'_All.csv',header=False,index=False)    


