# SCRIPT FOR SOLVING THE MERTON PROBLEM

#%% import needed packages

import DGM2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.integrate  import solve_ivp


parser = argparse.ArgumentParser(description="slope")
parser.add_argument("--id", type=int)
parser.add_argument("--backup", type=int)
parser.add_argument("--num", type=int)
parser.add_argument("--upperslope", type=float)
parser.add_argument("--lowerslope", type=float)
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

# num_layers = 3
# nodes_per_layer = 50
# # Training parameters
# sampling_stages  = 10   # number of times to resample new time-space domain points
# steps_per_sample = 10    # number of SGD steps to take before re-sampling

# Sampling parameters
# nSim_interior = 500
# nSim_boundary = 1

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

savefolder = 'baseline_ODEEmbedded3_loss/num_layers_FFNN_{}_activation_FFNN_{}_num_layers_RNN_{}_nodes_per_layer_{}/sampling_stages_{}_steps_per_sample_{}_learningrate_{}/nSim_interior_{}_nSim_boundary_{}/Interal=[{},{}]/Slopeid_{}_Slope_{}/backup_{}/'.format(num_layers_FFNN, activation_FFNN, num_layers_RNN, nodes_per_layer, sampling_stages, steps_per_sample, starting_learning_rate, nSim_interior, nSim_boundary, args.lowerslope, args.upperslope, idd, guess, backup)


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


def ODE(W, y):
    F,V = y

    
    F_update = V
    
    if V<0:
        c=(V/2)**2
    else:
        c = 0
    
    a_orig = 2/5 + 4/25 *V + 2*F+2*c-2*V*W +2*V*np.sqrt(c)
    
    a = a_orig * (a_orig > 0) + 0*(a_orig <=0)
    

    upper = F - a + c - V*(W-np.sqrt(c)+a**2/2 + 2/5 * a)
    lower = r*(a+2/5)**2 * sigma**2/2
    
    V_update = upper/lower
    
    return [F_update, V_update]

sol = solve_ivp(ODE, t_span=(0, 1), y0=[
    0, guess], method="DOP853", max_step=0.001)

X_plot = sol.t[sol.y[0] > -sol.t**2]
ODE_F = sol.y[0][sol.y[0] > -sol.t**2]
ODE_V = sol.y[1][sol.y[0] > -sol.t**2]


c_orig = (ODE_V/2)**2 * (ODE_V < 0) 

c = c_orig

a_orig = 2/5 + 4/25 * ODE_V + 2*ODE_F + \
    2*c-2*ODE_V*X_plot + 2*ODE_V*np.sqrt(c)

a = a_orig * (a_orig > 0) + 0*(a_orig <= 0)


ODE_a = a
ODE_c = c
ODE_drift = r*(X_plot-ODE_c**(1/2)+ODE_a**2/2+2*ODE_a/5)

X_plot = X_plot.reshape(-1,1)

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
    
    G = 0 + guess * X_interior + V * X_interior**2
    
    # V_x = tf.gradients(V, X_interior)[0]
    # V_xx = tf.gradients(V_x, X_interior)[0]
    
    G_x = tf.gradients(G, X_interior)[0]
    G_xx = tf.gradients(G_x, X_interior)[0]
    
    
    c = tf.where(G_x < 0, (G_x/2)**2, tf.zeros((nSim_interior, 1)))
    u_c = u(c)
    
    a_orig = 2/5 + 4/25*G_x + 2*G + 2*c - 2*G_x * X_interior+ 2*G_x * tf.sqrt(c)
    
    a = tf.where(a_orig >= 0,a_orig, tf.zeros((nSim_interior, 1)))
    # a = tf.maximum(tf.zeros_like(V),a)

    # a = -(1+0.4*v_w+sigma**2*r*0.4*v_ww)/(v_w+r*sigma**2*v_ww)
    h_a = h(a)

    gamma_a = gamma(a)
    
    
    Upper = G - a + c - G_x * (X_interior - u_c +h_a)
    Lower = r*gamma_a**2 * sigma**2/2
    
    # diff_V = Upper/Lower - V_xx
    diff_V = Upper - G_xx * Lower

    # concave_V = tf.maximum(V_xx, tf.zeros_like(V))

    L1 = tf.reduce_mean(tf.square(diff_V))  
    # L1 += tf.reduce_mean(tf.square(concave_V))
    
    # Loss term #2: boundary condition
        # no boundary condition for this problem
    
    # fitted_boundary = model(X_boundary)
    
    # fitted_boundary_W = tf.gradients(
    #     fitted_boundary, X_boundary)[0]
    
    # target_boundary = F0(X_boundary)
    # target_boundary_W = guess

    
    # L2_0 = tf.reduce_mean( tf.square(fitted_boundary - target_boundary) )
    # L2_1 = tf.reduce_mean( tf.square(fitted_boundary_W - target_boundary_W) )
    
    fitted_boundary_far = model(X_far)
    
    G_fitted_boundary_far = 0 + guess * X_far + fitted_boundary_far * X_far**2


    L2_3 = tf.reduce_mean(tf.square(tf.maximum(G_fitted_boundary_far,tf.zeros_like(G_fitted_boundary_far))))
    
    # L2 = L2_0 + L2_1 + L2_3
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
L1_tnsr, L2_tnsr = loss(model, X_interior_tnsr, X_boundary_tnsr, X_far_tnsr)
loss_tnsr = L1_tnsr +  L2_tnsr

# value function
V = model(X_interior_tnsr)
# V_x = tf.gradients(V, X_interior_tnsr)[0]
G = 0 + guess * X_interior_tnsr + V * X_interior_tnsr**2

# V_x = tf.gradients(V, X_interior)[0]
# V_xx = tf.gradients(V_x, X_interior)[0]

G_x = tf.gradients(G, X_interior_tnsr)[0]
G_xx = tf.gradients(G_x, X_interior_tnsr)[0]
    
    
# optimal control computed numerically from fitted value function 
def control_a(G,G_x):
    
    # V_xx = tf.gradients(V_x, X_interior_tnsr)[0]
    
    c = tf.where(G_x < 0, (G_x/2)**2, tf.zeros_like(G))
    u_c = u(c)
    
    a_orig = 2/5 + 4/25*G_x + 2*G + 2*c - 2*G_x * X_interior_tnsr+ 2*G_x * tf.sqrt(c)
    
    a = tf.where(a_orig >= 0, a_orig, tf.zeros_like(V))
    
    return a

def control_c(G,G_x):
    # length = V.shape[0]
    # V_x = tf.gradients(V, X_interior_tnsr)[0]
    # V_xx = tf.gradients(V_x, X_interior_tnsr)[0]
    
    c = tf.where(G_x < 0, (G_x/2)**2, tf.zeros_like(V))

    return c

numerical_a = control_a(G,G_x)
numerical_c = control_c(G,G_x)

# set optimizer - NOTE THIS IS DIFFERENT FROM OTHER APPLICATIONS!
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,10000, 0.95, staircase=True)
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


        # vector of t and S values for plotting
        X_plotF0 = np.linspace(X_low, X_high, n_plot)
        X_plotF0 = X_plotF0.reshape(-1,1)


        # simulate process at current t 

        fitted_V = sess.run([G], feed_dict= {X_interior_tnsr:X_plot})[0]
        fitted_a = sess.run([numerical_a], feed_dict= {X_interior_tnsr:X_plot})[0]
        fitted_c = sess.run([numerical_c], feed_dict= {X_interior_tnsr:X_plot})[0]
        B_W = r*(X_plot-fitted_c**(1/2)+fitted_a**2/2+2*fitted_a/5)
        fitted_drift = B_W

        figwidth = 10
        fig, axs = plt.subplot_mosaic(
        [["left column", "right top"],
        ["left column", "right mid"],
        ["left column", "right down"]], figsize=(4 * figwidth, 2 * figwidth)
        )


        axs["left column"].plot(X_plot, fitted_V, color = 'red', label = 'NN Solution')
        axs["left column"].plot(X_plot, ODE_F, color = 'blue', linestyle='--',label='ODE Solution')
        axs["left column"].set_ylim(-1,0.15)
        axs["left column"].plot(X_plotF0, F0(X_plotF0), color = 'black')
        axs["left column"].set_xlim(X_low,X_high)
        axs["left column"].set_title("Profit $F(W)$")
        axs["left column"].grid(linestyle=':')
        axs["left column"].legend()

        axs["right top"].plot(X_plot, fitted_a, color = 'red', label = 'NN Solution')
        axs["right top"].plot(X_plot, ODE_a, color = 'blue', linestyle='--',label='ODE Solution')
        axs["right top"].set_ylim(0,1)
        axs["right top"].set_xlim(X_low,X_high)
        axs["right top"].set_title("Effort $\\alpha(W)$")
        axs["right top"].grid(linestyle=':')

        axs["right mid"].plot(X_plot, fitted_c, color = 'red', label = 'NN Solution')
        axs["right mid"].plot(X_plot, ODE_c, color = 'blue', linestyle='--',label='ODE Solution')
        axs["right mid"].set_ylim(0, 1)
        axs["right mid"].set_xlim(X_low,X_high)
        axs["right mid"].set_title("Consumption $\\pi(W)$")
        axs["right mid"].grid(linestyle=':')

        axs["right down"].plot(X_plot, fitted_drift, color = 'red', label = 'NN Solution')
        axs["right down"].plot(X_plot, ODE_drift, color = 'blue', linestyle='--',label='ODE Solution')
        axs["right down"].set_ylim(0, 0.1)
        axs["right down"].set_xlim(X_low,X_high)
        axs["right down"].set_title("Drift of $W$")
        axs["right down"].grid(linestyle=':')
        
        # plt.savefig(figureName + '_All.png')
        plt.savefig('./Figure/' +savefolder+ figureName + '_i_{}_All.png'.format(i))

        plt.close('all')        



        figwidth = 10
        fig, axs = plt.subplot_mosaic(
        [["left top", "right top"],
        ["left mid", "right mid"]], figsize=(3 * figwidth, 1.5 * figwidth)
        )


        axs["left top"].plot(X_plot, np.log(abs(fitted_V-ODE_F.reshape(-1,1))), color = 'red')
        axs["left top"].set_title("Difference in Profit $F(W)$")
        axs["left top"].set_xlim(X_low,X_high)
        axs["left top"].grid(linestyle=':')

        axs["left mid"].plot(X_plot, np.log(abs(fitted_a-ODE_a.reshape(-1,1))), color = 'red')
        axs["left mid"].set_title("Difference in Effort $\\alpha(W)$")
        axs["left mid"].grid(linestyle=':')
        axs["left mid"].set_xlim(X_low,X_high)

        axs["right top"].plot(X_plot, np.log(abs(fitted_c-ODE_c.reshape(-1,1))), color = 'red')
        axs["right top"].set_title("Difference in Consumption $\\pi(W)$")
        axs["right top"].grid(linestyle=':')
        axs["right top"].set_xlim(X_low,X_high)

        axs["right mid"].plot(X_plot, np.log(abs(fitted_drift-ODE_drift.reshape(-1,1))), color = 'red')
        axs["right mid"].set_title("Difference in Drift of $W$")
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

    

a = 0.01 * np.ones((1,1))
b = 1 * np.ones((1,1))


error = 1
tol =1e-7
count =0
maxcount = 100
while error > tol and count < maxcount:
    
    
    Va = sess.run([G], feed_dict= {X_interior_tnsr:a})[0]
    Vb = sess.run([G], feed_dict= {X_interior_tnsr:b})[0]
    
    f0 = Va - F0(a)
    f1 = Vb - F0(b)
    
    if f0*f1<0:
        
        c = a+b
        c = c/2
        
        Vc = sess.run([G], feed_dict= {X_interior_tnsr:c})[0]

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


        
Vc = sess.run([G], feed_dict= {X_interior_tnsr:c})[0]
dVc = sess.run([G_x], feed_dict= {X_interior_tnsr:c})[0]


L2_0 = Vc-F0(c)

L2_1 = dVc - F0_W(c)

print(L2_0, L2_1, L2_0+ L2_1)


# simulate process at current t 

fitted_V = sess.run([G], feed_dict= {X_interior_tnsr:X_plot})[0]
fitted_a = sess.run([numerical_a], feed_dict= {X_interior_tnsr:X_plot})[0]
fitted_c = sess.run([numerical_c], feed_dict= {X_interior_tnsr:X_plot})[0]
B_W = r*(X_plot-fitted_c**(1/2)+fitted_a**2/2+2*fitted_a/5)
fitted_drift = B_W

figwidth = 10
fig, axs = plt.subplot_mosaic(
[["left column", "right top"],
["left column", "right mid"],
["left column", "right down"]], figsize=(4 * figwidth, 2 * figwidth)
)


axs["left column"].plot(X_plot, fitted_V, color = 'red', label = 'NN Solution')
axs["left column"].plot(X_plot, ODE_F, color = 'blue', linestyle='--',label='ODE Solution')
axs["left column"].set_ylim(-1,0.15)
axs["left column"].plot(X_plotF0, F0(X_plotF0), color = 'black')
axs["left column"].set_xlim(X_low,X_high)
axs["left column"].set_title("Profit $F(W)$")
axs["left column"].grid(linestyle=':')
axs["left column"].legend()

axs["right top"].plot(X_plot, fitted_a, color = 'red', label = 'NN Solution')
axs["right top"].plot(X_plot, ODE_a, color = 'blue', linestyle='--',label='ODE Solution')
axs["right top"].set_ylim(0,1)
axs["right top"].set_xlim(X_low,X_high)
axs["right top"].set_title("Effort $\\alpha(W)$")
axs["right top"].grid(linestyle=':')

axs["right mid"].plot(X_plot, fitted_c, color = 'red', label = 'NN Solution')
axs["right mid"].plot(X_plot, ODE_c, color = 'blue', linestyle='--',label='ODE Solution')
axs["right mid"].set_ylim(0, 1)
axs["right mid"].set_xlim(X_low,X_high)
axs["right mid"].set_title("Consumption $\\pi(W)$")
axs["right mid"].grid(linestyle=':')

axs["right down"].plot(X_plot, fitted_drift, color = 'red', label = 'NN Solution')
axs["right down"].plot(X_plot, ODE_drift, color = 'blue', linestyle='--',label='ODE Solution')
axs["right down"].set_ylim(0, 0.1)
axs["right down"].set_xlim(X_low,X_high)
axs["right down"].set_title("Drift of $W$")
axs["right down"].grid(linestyle=':')

# plt.savefig(figureName + '_All.png')
plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_All_IP_{}_dFdiff_{}.png'.format(c,L2_1))

plt.close('all')        



figwidth = 10
fig, axs = plt.subplot_mosaic(
[["left top", "right top"],
["left mid", "right mid"]], figsize=(3 * figwidth, 1.5 * figwidth)
)


axs["left top"].plot(X_plot, np.log(abs(fitted_V-ODE_F.reshape(-1,1))), color = 'red')
axs["left top"].set_title("Difference in Profit $F(W)$")
axs["left top"].set_xlim(X_low,X_high)
axs["left top"].grid(linestyle=':')

axs["left mid"].plot(X_plot, np.log(abs(fitted_a-ODE_a.reshape(-1,1))), color = 'red')
axs["left mid"].set_title("Difference in Effort $\\alpha(W)$")
axs["left mid"].grid(linestyle=':')
axs["left mid"].set_xlim(X_low,X_high)

axs["right top"].plot(X_plot, np.log(abs(fitted_c-ODE_c.reshape(-1,1))), color = 'red')
axs["right top"].set_title("Difference in Consumption $\\pi(W)$")
axs["right top"].grid(linestyle=':')
axs["right top"].set_xlim(X_low,X_high)

axs["right mid"].plot(X_plot, np.log(abs(fitted_drift-ODE_drift.reshape(-1,1))), color = 'red')
axs["right mid"].set_title("Difference in Drift of $W$")
axs["right mid"].grid(linestyle=':')
axs["right mid"].set_xlim(X_low,X_high)

# plt.savefig(figureName + '_All.png')
plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_Diff_IP_{}_dFdiff_{}.png'.format(c,L2_1))

plt.close('all')        
        

#%%
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.semilogy(range(len(loss_list)), loss_list, 'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi^{n_{epoch}}$')
plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_LossList.png')

