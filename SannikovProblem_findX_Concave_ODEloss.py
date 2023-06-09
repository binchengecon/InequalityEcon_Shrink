# SCRIPT FOR SOLVING THE MERTON PROBLEM

#%% import needed packages

import DGM2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description="slope")
parser.add_argument("--id", type=int)
parser.add_argument("--backup", type=int)
parser.add_argument("--num", type=int)
parser.add_argument("--upperslope", type=float)
parser.add_argument("--lowerslope", type=float)
parser.add_argument("--learning", type=float)

# args = parser.parse_args()
args, unknown = parser.parse_known_args()

# LaTeX rendering for text in plots
plt.style.use('classic')
plt.rcParams["savefig.bbox"] = "tight"
# plt.rcParams["figure.figsize"] = (10,8)
# plt.rcParams["figure.dpi"] = 500
plt.rcParams["font.size"] = 16
# plt.rcParams["legend.frameon"] = True
# plt.rcParams["lines.linewidth"] = 5

#%% Parameters 

# Sannikov problem parameters 
r = 0.1
sigma = 1


# Solution parameters (domain on which to solve PDE)
X_low = 0.0  # wealth lower bound
X_high = 1           # wealth upper bound

# neural network parameters
num_layers = 3
nodes_per_layer = 50
starting_learning_rate = args.learning
# Training parameters
sampling_stages  = 40000   # number of times to resample new time-space domain points
steps_per_sample = 10    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = 500
nSim_boundary = 1

# multipliers for oversampling i.e. draw X from [X_low - X_oversample, X_high + X_oversample]
X_oversample = 0.5
t_oversample = 0.5

# Plot options
n_plot = 600  # Points on plot grid for each dimension

sloperange = np.linspace(args.lowerslope, args.upperslope, args.num)
idd = args.id
backup = args.backup

guess = sloperange[idd]

print("slope={}".format(guess))
# Save options
saveOutput = False

savefolder = 'baseline_ODEloss/Interal=[{},{}]/Slopeid_{}/learning_{}/backup_{}/'.format(args.lowerslope, args.upperslope, idd, starting_learning_rate, backup)
saveName   = 'Sannikov'
saveFigure = False
figureName = saveName
#%% Analytical Solution

os.makedirs('./SavedNets/' + savefolder  ,exist_ok=True)
os.makedirs('./Figure/' + savefolder  ,exist_ok=True)
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

def F0_W(w):
    return -2*w

def h(a):
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
    X_interior = np.random.uniform(low=X_low-0.1, high=X_high, size=[nSim_interior, 1])
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

def loss(model, X_interior, X_boundary, X_far):
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
    
    
    c = tf.where(V_x < 0, (V_x/2)**2, tf.zeros((nSim_interior, 1)))
    u_c = u(c)
    
    a_orig = 2/5 + 4/25*V_x + 2*V + 2*c - 2*V_x * X_interior+ 2*V_x * tf.sqrt(c)
    
    a = tf.where(a_orig >= 0,a_orig, tf.zeros((nSim_interior, 1)))
    # a = tf.maximum(tf.zeros_like(V),a)

    # a = -(1+0.4*v_w+sigma**2*r*0.4*v_ww)/(v_w+r*sigma**2*v_ww)
    h_a = h(a)

    gamma_a = gamma(a)
    
    
    Upper = V - a + c - V_x * (X_interior - u_c +h_a)
    Lower = r*gamma_a**2 * sigma**2/2
    
    # diff_V = Upper/Lower - V_xx
    diff_V = Upper - V_xx * Lower

    # concave_V = tf.maximum(V_xx, tf.zeros_like(V))

    L1 = tf.reduce_mean(tf.square(diff_V))  
    # L1 += tf.reduce_mean(tf.square(concave_V))
    
    # Loss term #2: boundary condition
        # no boundary condition for this problem
    
    fitted_boundary = model(X_boundary)
    
    fitted_boundary_W = tf.gradients(
        fitted_boundary, X_boundary)[0]
    
    target_boundary = F0(X_boundary)
    target_boundary_W = guess

    
    L2_0 = tf.reduce_mean( tf.square(fitted_boundary - target_boundary) )
    L2_1 = tf.reduce_mean( tf.square(fitted_boundary_W - target_boundary_W) )
    
    fitted_boundary_far = model(X_far)
    
    L2_3 = tf.reduce_mean(tf.square(tf.maximum(fitted_boundary_far,tf.zeros_like(fitted_boundary_far))))
    
    L2 = L2_0 + L2_1 + L2_3


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
X_far_tnsr = tf.placeholder(tf.float32, [None,1])

# loss 
L1_tnsr, L2_tnsr = loss(model, X_interior_tnsr, X_boundary_tnsr, X_far_tnsr)
loss_tnsr = L1_tnsr +  L2_tnsr

# value function
V = model(X_interior_tnsr)
V_x = tf.gradients(V, X_interior_tnsr)[0]

# optimal control computed numerically from fitted value function 
def control_a(V):
    
    V_x = tf.gradients(V, X_interior_tnsr)[0]
    # V_xx = tf.gradients(V_x, X_interior_tnsr)[0]
    
    c = tf.where(V_x < 0, (V_x/2)**2, tf.zeros_like(V))
    u_c = u(c)
    
    a_orig = 2/5 + 4/25*V_x + 2*V + 2*c - 2*V_x * X_interior_tnsr+ 2*V_x * tf.sqrt(c)
    
    a = tf.where(a_orig >= 0, a_orig, tf.zeros_like(V))
    
    return a

def control_c(V):
    # length = V.shape[0]
    V_x = tf.gradients(V, X_interior_tnsr)[0]
    # V_xx = tf.gradients(V_x, X_interior_tnsr)[0]
    
    c = tf.where(V_x < 0, (V_x/2)**2, tf.zeros_like(V))

    return c

numerical_a = control_a(V)
numerical_c = control_c(V)

# set optimizer - NOTE THIS IS DIFFERENT FROM OTHER APPLICATIONS!
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,15000, 0.95, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr, global_step=global_step)

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
        loss,L1,L2,_ = sess.run([loss_tnsr, L1_tnsr, L2_tnsr, optimizer],
                                feed_dict = {X_interior_tnsr:X_interior, X_boundary_tnsr:X_boundary, X_far_tnsr:X_far})
        loss_list.append(loss)
    
    print(loss, L1, L2, i)

    if i%4000==0:
        figwidth = 10

        # figure options

        # time values at which to examine density
        # valueTimes = [t_low, T/3, 2*T/3, T]
        fig, axs = plt.subplot_mosaic(

        [["left column", "right top"],
        ["left column", "right mid"],
        ["left column", "right down"]], figsize=(4 * figwidth, 2 * figwidth)

        )

        # vector of t and S values for plotting
        X_plot = np.linspace(X_low, X_high, n_plot)
        X_plot = X_plot.reshape(-1,1)


        # simulate process at current t 

        fitted_V = sess.run([V], feed_dict= {X_interior_tnsr:X_plot})[0]
        fitted_a = sess.run([numerical_a], feed_dict= {X_interior_tnsr:X_plot})[0]
        fitted_c = sess.run([numerical_c], feed_dict= {X_interior_tnsr:X_plot})[0]
        B_W = r*(X_plot-fitted_c**(1/2)+fitted_a**2/2+2*fitted_a/5)
        fitted_drift = B_W

        axs["left column"].plot(X_plot, fitted_V, color = 'red')
        axs["left column"].set_ylim(-1,0.15)
        axs["left column"].plot(X_plot, F0(X_plot), color = 'black')
        axs["left column"].set_title("Profit $F(W)$")
        axs["left column"].grid(linestyle=':')

        axs["right top"].plot(X_plot, fitted_a, color = 'red')
        axs["right top"].set_ylim(0,1)
        axs["right top"].set_title("Effort $\\alpha(W)$")
        axs["right top"].grid(linestyle=':')

        axs["right mid"].plot(X_plot, fitted_c, color = 'red')
        axs["right mid"].set_ylim(0, 1)
        axs["right mid"].set_title("Consumption $\\pi(W)$")
        axs["right mid"].grid(linestyle=':')

        axs["right down"].plot(X_plot, fitted_drift, color = 'red')
        axs["right down"].set_ylim(0, 0.1)
        axs["right down"].set_title("Drift of $W$")
        axs["right down"].grid(linestyle=':')
            
        # plt.savefig(figureName + '_All.png')
        plt.savefig('./Figure/' +savefolder+ figureName + '_i_{}_All.png'.format(i))

        plt.close('all')        

# save outout
if saveOutput:
    saver = tf.train.Saver()
    saver.save(sess, './SavedNets/' + saveName)
       
#%% Plot value function results
saver = tf.train.Saver()


# saver.save(sess, './SavedNets/' + savefolder + saveName)

    

a = 0.01 * np.ones((1,1))
b = 1 * np.ones((1,1))


error = 1
tol =1e-7
count =0
maxcount = 100
while error > tol and count < maxcount:
    
    
    Va = sess.run([V], feed_dict= {X_interior_tnsr:a})[0]
    Vb = sess.run([V], feed_dict= {X_interior_tnsr:b})[0]
    
    f0 = Va - F0(a)
    f1 = Vb - F0(b)
    
    if f0*f1<0:
        
        c = a+b
        c = c/2
        
        Vc = sess.run([V], feed_dict= {X_interior_tnsr:c})[0]

        fc = Vc - F0(c)
        
        if f0*fc >0:
            
            a = c
        elif f1*fc>0:
            
            b = c

        error = abs(fc)
    else: 
        print("Error")
        c = a+b
        c = c/2
        
    count = count+1


        
Vc = sess.run([V], feed_dict= {X_interior_tnsr:c})[0]
dVc = sess.run([V_x], feed_dict= {X_interior_tnsr:c})[0]


L2_0 = Vc-F0(c)

L2_1 = dVc - F0_W(c)

print(L2_0, L2_1, L2_0+ L2_1)





figwidth = 10

# figure options
plt.figure(figsize = (12,10))

# time values at which to examine density
# valueTimes = [t_low, T/3, 2*T/3, T]
fig, axs = plt.subplot_mosaic(

[["left column", "right top"],
["left column", "right mid"],
["left column", "right down"]], figsize=(4 * figwidth, 2 * figwidth)

)

# vector of t and S values for plotting
X_plot = np.linspace(X_low, X_high, n_plot)
X_plot = X_plot.reshape(-1,1)


# simulate process at current t 

fitted_V = sess.run([V], feed_dict= {X_interior_tnsr:X_plot})[0]
fitted_a = sess.run([numerical_a], feed_dict= {X_interior_tnsr:X_plot})[0]
fitted_c = sess.run([numerical_c], feed_dict= {X_interior_tnsr:X_plot})[0]
B_W = r*(X_plot-fitted_c**(1/2)+fitted_a**2/2+2*fitted_a/5)
fitted_drift = B_W

axs["left column"].plot(X_plot, fitted_V, color = 'red')
axs["left column"].set_ylim(-1,0.15)
axs["left column"].plot(X_plot, F0(X_plot), color = 'black')
axs["left column"].set_title("Profit $F(W)$")
axs["left column"].grid(linestyle=':')

axs["right top"].plot(X_plot, fitted_a, color = 'red')
axs["right top"].set_ylim(0,1)
axs["right top"].set_title("Effort $\\alpha(W)$")
axs["right top"].grid(linestyle=':')

axs["right mid"].plot(X_plot, fitted_c, color = 'red')
axs["right mid"].set_ylim(0, 1)
axs["right mid"].set_title("Consumption $\\pi(W)$")
axs["right mid"].grid(linestyle=':')

axs["right down"].plot(X_plot, fitted_drift, color = 'red')
axs["right down"].set_ylim(0, 0.1)
axs["right down"].set_title("Drift of $W$")
axs["right down"].grid(linestyle=':')
    
# plt.savefig(figureName + '_All.png')
plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_All_IP_{}_dFdiff_{}.png'.format(c,L2_1))


if saveFigure:
    plt.savefig(figureName + '_All.png')



#%%
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
ax.semilogy(range(len(loss_list)), loss_list, 'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi^{n_{epoch}}$')
plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_LossList.png')

# plt.savefig(figureName + '_LossList.png')

if saveFigure:
    plt.savefig(figureName + '_LossList.png')   