# SCRIPT FOR SOLVING THE MERTON PROBLEM

#%% import needed packages

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
from scipy.integrate  import solve_ivp


parser = argparse.ArgumentParser(description="slope")
parser.add_argument("--slope",nargs='+', type=float)
parser.add_argument("--uncertainty", nargs='+', type=float)
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
plt.rcParams["lines.linewidth"] = 4

#%% Parameters 

# Sannikov problem parameters 
r = 0.1
sigma = 1

colors = ['red', 'green', 'blue', 'orange', 'purple']

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

# sloperange = np.linspace(args.lowerslope, args.upperslope, args.num)
# idd = args.id
# backup = args.backup

slopearr = args.slope
xiarr  = args.uncertainty

slopearr = np.array(slopearr)
xiarr = np.array(xiarr)

print("slope={}, xi={}".format(slopearr, xiarr))
# Save options
saveOutput = False

savefolder = 'baseline_ODEEmbedded4_robust/num_layers_FFNN_{}_activation_FFNN_{}_num_layers_RNN_{}_nodes_per_layer_{}/sampling_stages_{}_steps_per_sample_{}_learningrate_{}/nSim_interior_{}_nSim_boundary_{}/'.format(num_layers_FFNN, activation_FFNN, num_layers_RNN, nodes_per_layer, sampling_stages, steps_per_sample, starting_learning_rate, nSim_interior, nSim_boundary)


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

X_plotF0 = np.linspace(0,1,1000)


figwidth = 10
fig, axs = plt.subplot_mosaic(
[["left column", "right top"],
["left column", "right mid"],
["left column", "right dis"],
["left down", "right down"]], figsize=(25, 15), sharex=True
)


for id in range(len(slopearr)):

    guess = slopearr[id]
    xi = xiarr[id]
    
    Sannikov = pd.read_csv(
        './Figure/' +savefolder+ '/slope={}_xi_{}'.format(slopearr[id], xiarr[id])+ '/' +figureName + '_All.csv')
    Sannikov = np.array(Sannikov)

    X_plot = Sannikov[:,:1] 
    fitted_V = Sannikov[:,1:2] 
    fitted_a = Sannikov[:,2:3] 
    fitted_c = Sannikov[:,3:4] 
    fitted_h = Sannikov[:,4:5] 
    fitted_drift = Sannikov[:,5:6] 

    ODE_F = Sannikov[:,6:7] 
    ODE_a = Sannikov[:,7:8] 
    ODE_c = Sannikov[:,8:9] 
    ODE_h = Sannikov[:,9:10] 
    ODE_drift = Sannikov[:,10:11] 

    Diff_F = Sannikov[:,11:12] 
    Diff_a = Sannikov[:,12:13] 
    Diff_c = Sannikov[:,13:14]
    Diff_h = Sannikov[:,14:15] 
    Diff_drift = Sannikov[:,15:16] 

    color_one = colors[id % len(colors)]

    axs["left column"].plot(X_plot, fitted_V, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["left column"].plot(X_plot, ODE_F, color = 'black', linestyle='--')
    axs["left column"].set_ylim(-1,0.15)
    if id==len(slopearr)-1:
        axs["left column"].plot(X_plotF0, F0(X_plotF0), color = 'black', label="Retirement Profit")
    axs["left column"].set_xlim(X_low,X_high)
    axs["left column"].set_title("Profit $F(W)$")
    axs["left column"].grid(linestyle=':')
    axs["left column"].legend()

    axs["left down"].plot(X_plot, fitted_a+fitted_h, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["left down"].plot(X_plot, ODE_a+ODE_h, color = 'black', linestyle='--')
    axs["left down"].set_ylim(-1,1.5)
    axs["left down"].set_xlim(X_low,X_high)
    axs["left down"].set_title("Drift of Output $X$")
    axs["left down"].grid(linestyle=':')
    axs["left down"].set_xlabel("$W$")



    axs["right top"].plot(X_plot, fitted_a, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["right top"].plot(X_plot, ODE_a, color = 'black', linestyle='--')
    axs["right top"].set_ylim(0,1)
    axs["right top"].set_xlim(X_low,X_high)
    axs["right top"].set_title("Effort $\\alpha(W)$")
    axs["right top"].grid(linestyle=':')

    axs["right mid"].plot(X_plot, fitted_c, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["right mid"].plot(X_plot, ODE_c, color = 'black', linestyle='--')
    axs["right mid"].set_ylim(0, 1)
    axs["right mid"].set_xlim(X_low,X_high)
    axs["right mid"].set_title("Consumption $\\pi(W)$")
    axs["right mid"].grid(linestyle=':')


    axs["right dis"].plot(X_plot, fitted_h, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["right dis"].plot(X_plot, ODE_h, color = 'black', linestyle='--')
    # axs["right down"].set_ylim(0, 0.1)
    axs["right dis"].set_xlim(X_low,X_high)
    axs["right dis"].set_title("Distortion $h(W)$")
    axs["right dis"].grid(linestyle=':')



    axs["right down"].plot(X_plot, fitted_drift, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["right down"].plot(X_plot, ODE_drift, color = 'black', linestyle='--')
    # axs["right down"].set_ylim(0, 0.1)
    axs["right down"].set_xlim(X_low,X_high)
    axs["right down"].set_title("Drift of Continuation Payoff $W$")
    axs["right down"].grid(linestyle=':')
    axs["right down"].set_xlabel("$W$")

# axs["left column"].plot(X_plotF0, F0(X_plotF0), color = 'black', label="Retirement Profit")


plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_All_{}.pdf'.format(len(xiarr)))

plt.close('all')        




figwidth = 10
fig, axs = plt.subplot_mosaic(
[["left top", "right top"],
["left mid", "right mid"],
["down", "down"]], figsize=(15, 10), sharex=True
)


for id in range(len(slopearr)):

    guess = slopearr[id]
    xi = xiarr[id]
    
    color_one = colors[id % len(colors)]

    Sannikov = pd.read_csv(
        './Figure/' +savefolder+ '/slope={}_xi_{}'.format(slopearr[id], xiarr[id])+ '/' +figureName + '_All.csv')
    Sannikov = np.array(Sannikov)

    X_plot = Sannikov[:,:1] 
    fitted_V = Sannikov[:,1:2] 
    fitted_a = Sannikov[:,2:3] 
    fitted_c = Sannikov[:,3:4] 
    fitted_h = Sannikov[:,4:5] 
    fitted_drift = Sannikov[:,5:6] 

    ODE_F = Sannikov[:,6:7] 
    ODE_a = Sannikov[:,7:8] 
    ODE_c = Sannikov[:,8:9] 
    ODE_h = Sannikov[:,9:10] 
    ODE_drift = Sannikov[:,10:11] 

    Diff_F = Sannikov[:,11:12] 
    Diff_a = Sannikov[:,12:13] 
    Diff_c = Sannikov[:,13:14]
    Diff_h = Sannikov[:,14:15] 
    Diff_drift = Sannikov[:,15:16] 


    axs["left top"].plot(X_plot, Diff_F, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["left top"].set_title("Difference in Profit $F(W)$")
    axs["left top"].set_xlim(X_low,X_high)
    axs["left top"].grid(linestyle=':')
    # axs["left top"].set_xlabel("$W$")

    axs["left mid"].plot(X_plot, Diff_a, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["left mid"].set_title("Difference in Effort $\\alpha(W)$")
    axs["left mid"].grid(linestyle=':')
    axs["left mid"].set_xlim(X_low,X_high)

    axs["right top"].plot(X_plot, Diff_c, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["right top"].set_title("Difference in Consumption $\\pi(W)$")
    axs["right top"].grid(linestyle=':')
    axs["right top"].set_xlim(X_low,X_high)

    axs["right mid"].plot(X_plot, Diff_h, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["right mid"].set_title("Difference in $h(W)$")
    axs["right mid"].grid(linestyle=':')
    axs["right mid"].set_xlim(X_low,X_high)

    axs["down"].plot(X_plot, Diff_drift, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["down"].set_title("Difference in Drift of Continuation Payoff $W$")
    axs["down"].grid(linestyle=':')
    axs["down"].set_xlim(X_low,X_high)
    axs["down"].legend()
    axs["down"].set_xlabel("$W$")

plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_Diff_{}.pdf'.format(len(xiarr)))

plt.close('all')        
            





figwidth = 10
fig, axs = plt.subplot_mosaic(
[["left column", "right top"],
["left column", "right mid"],
["left column", "right dis"],
["left down", "right down"]], figsize=(25, 15), sharex=True
)



for id in range(len(slopearr)):

    guess = slopearr[id]
    xi = xiarr[id]
    
    Sannikov = pd.read_csv(
        './Figure/' +savefolder+ '/slope={}_xi_{}'.format(slopearr[id], xiarr[id])+ '/' +figureName + '_All.csv')
    Sannikov = np.array(Sannikov)

    X_plot = Sannikov[:,:1] 
    fitted_V = Sannikov[:,1:2] 
    fitted_a = Sannikov[:,2:3] 
    fitted_c = Sannikov[:,3:4] 
    fitted_h = Sannikov[:,4:5] 
    fitted_drift = Sannikov[:,5:6] 

    ODE_F = Sannikov[:,6:7] 
    ODE_a = Sannikov[:,7:8] 
    ODE_c = Sannikov[:,8:9] 
    ODE_h = Sannikov[:,9:10] 
    ODE_drift = Sannikov[:,10:11] 

    Diff_F = Sannikov[:,11:12] 
    Diff_a = Sannikov[:,12:13] 
    Diff_c = Sannikov[:,13:14]
    Diff_h = Sannikov[:,14:15] 
    Diff_drift = Sannikov[:,15:16] 

    color_one = colors[id % len(colors)]

    index = (fitted_V+X_plot**2)[X_plot>0].argmin()

    X_plot = X_plot[:index+2]

    # print("check, index = {}, minimum ={}, new value ={}".format(index,(fitted_V+X_plot**2)[X_plot>0][index], (fitted_V+X_plot**2)[index+1]))
    
    
    fitted_V = Sannikov[:,1:2][:index+2]
    fitted_a = Sannikov[:,2:3][:index+2]
    fitted_c = Sannikov[:,3:4][:index+2]
    fitted_h = Sannikov[:,4:5][:index+2]
    fitted_drift = Sannikov[:,5:6][:index+2] 

    ODE_F = Sannikov[:,6:7][:index+2]
    ODE_a = Sannikov[:,7:8][:index+2]
    ODE_c = Sannikov[:,8:9][:index+2]
    ODE_h = Sannikov[:,9:10][:index+2]
    ODE_drift = Sannikov[:,10:11][:index+2]

    Diff_F = Sannikov[:,11:12][:index+2]
    Diff_a = Sannikov[:,12:13][:index+2]
    Diff_c = Sannikov[:,13:14][:index+2]
    Diff_h = Sannikov[:,14:15][:index+2]
    Diff_drift = Sannikov[:,15:16][:index+2]



    axs["left column"].plot(X_plot, fitted_V, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["left column"].plot(X_plot, ODE_F, color = 'black', linestyle='--')
    axs["left column"].set_ylim(-1,0.15)
    if id==len(slopearr)-1:
        axs["left column"].plot(X_plotF0, F0(X_plotF0), color = 'black', label="Retirement Profit")
    axs["left column"].set_xlim(X_low,X_high)
    axs["left column"].set_title("Profit $F(W)$")
    axs["left column"].grid(linestyle=':')
    axs["left column"].legend()

    axs["left down"].plot(X_plot, fitted_a+fitted_h, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["left down"].plot(X_plot, ODE_a+ODE_h, color = 'black', linestyle='--')
    axs["left down"].set_ylim(-1,1.5)
    axs["left down"].set_xlim(X_low,X_high)
    axs["left down"].set_title("Drift of Output $X$")
    axs["left down"].grid(linestyle=':')
    axs["left down"].set_xlabel("$W$")


    axs["right top"].plot(X_plot, fitted_a, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["right top"].plot(X_plot, ODE_a, color = 'black', linestyle='--')
    axs["right top"].set_ylim(0,1)
    axs["right top"].set_xlim(X_low,X_high)
    axs["right top"].set_title("Effort $\\alpha(W)$")
    axs["right top"].grid(linestyle=':')

    axs["right mid"].plot(X_plot, fitted_c, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["right mid"].plot(X_plot, ODE_c, color = 'black', linestyle='--')
    axs["right mid"].set_ylim(0, 1)
    axs["right mid"].set_xlim(X_low,X_high)
    axs["right mid"].set_title("Consumption $\\pi(W)$")
    axs["right mid"].grid(linestyle=':')


    axs["right dis"].plot(X_plot, fitted_h, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["right dis"].plot(X_plot, ODE_h, color = 'black', linestyle='--')
    # axs["right down"].set_ylim(0, 0.1)
    axs["right dis"].set_xlim(X_low,X_high)
    axs["right dis"].set_title("Distortion $h(W)$")
    axs["right dis"].grid(linestyle=':')



    axs["right down"].plot(X_plot, fitted_drift, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["right down"].plot(X_plot, ODE_drift, color = 'black', linestyle='--')
    axs["right down"].set_ylim(-0.1, 0.1)
    axs["right down"].set_xlim(X_low,X_high)
    axs["right down"].set_title("Drift of Continuation Payoff $W$")
    axs["right down"].grid(linestyle=':')
    axs["right down"].set_xlabel("$W$")

# axs["left column"].plot(X_plotF0, F0(X_plotF0), color = 'black', label="Retirement Profit")


plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_All_Stop_{}.pdf'.format(len(xiarr)))

plt.close('all')        




figwidth = 10
fig, axs = plt.subplot_mosaic(
[["left top", "right top"],
["left mid", "right mid"],
["down", "down"]], figsize=(15, 10), sharex=True
)

# fig, axs = plt.subplot_mosaic(
# [["left column", "right top"],
# ["left column", "right mid"],
# ["left column", "right dis"],
# ["left column", "right down"]], figsize=(20, 10), sharex=True
# )

for id in range(len(slopearr)):

    guess = slopearr[id]
    xi = xiarr[id]
    
    color_one = colors[id % len(colors)]

    Sannikov = pd.read_csv(
        './Figure/' +savefolder+ '/slope={}_xi_{}'.format(slopearr[id], xiarr[id])+ '/' +figureName + '_All.csv')
    Sannikov = np.array(Sannikov)

    X_plot = Sannikov[:,:1] 
    fitted_V = Sannikov[:,1:2] 
    fitted_a = Sannikov[:,2:3] 
    fitted_c = Sannikov[:,3:4] 
    fitted_h = Sannikov[:,4:5] 
    fitted_drift = Sannikov[:,5:6] 

    ODE_F = Sannikov[:,6:7] 
    ODE_a = Sannikov[:,7:8] 
    ODE_c = Sannikov[:,8:9] 
    ODE_h = Sannikov[:,9:10] 
    ODE_drift = Sannikov[:,10:11] 

    Diff_F = Sannikov[:,11:12] 
    Diff_a = Sannikov[:,12:13] 
    Diff_c = Sannikov[:,13:14]
    Diff_h = Sannikov[:,14:15] 
    Diff_drift = Sannikov[:,15:16] 

    index = (fitted_V+X_plot**2)[X_plot>0].argmin()

    X_plot = X_plot[:index+2]

    # print("check, index = {}, minimum ={}, new value ={}".format(index,(fitted_V+X_plot**2)[X_plot>0][index], (fitted_V+X_plot**2)[index+1]))
    
    
    fitted_V = Sannikov[:,1:2][:index+2]
    fitted_a = Sannikov[:,2:3][:index+2]
    fitted_c = Sannikov[:,3:4][:index+2]
    fitted_h = Sannikov[:,4:5][:index+2]
    fitted_drift = Sannikov[:,5:6][:index+2] 

    ODE_F = Sannikov[:,6:7][:index+2]
    ODE_a = Sannikov[:,7:8][:index+2]
    ODE_c = Sannikov[:,8:9][:index+2]
    ODE_h = Sannikov[:,9:10][:index+2]
    ODE_drift = Sannikov[:,10:11][:index+2]

    Diff_F = Sannikov[:,11:12][:index+2]
    Diff_a = Sannikov[:,12:13][:index+2]
    Diff_c = Sannikov[:,13:14][:index+2]
    Diff_h = Sannikov[:,14:15][:index+2]
    Diff_drift = Sannikov[:,15:16][:index+2]


    axs["left top"].plot(X_plot, Diff_F, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["left top"].set_title("Difference in Profit $F(W)$")
    axs["left top"].set_xlim(X_low,X_high)
    axs["left top"].grid(linestyle=':')
    
    axs["left mid"].plot(X_plot, Diff_a, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["left mid"].set_title("Difference in Effort $\\alpha(W)$")
    axs["left mid"].grid(linestyle=':')
    axs["left mid"].set_xlim(X_low,X_high)

    axs["right top"].plot(X_plot, Diff_c, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["right top"].set_title("Difference in Consumption $\\pi(W)$")
    axs["right top"].grid(linestyle=':')
    axs["right top"].set_xlim(X_low,X_high)

    axs["right mid"].plot(X_plot, Diff_h, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["right mid"].set_title("Difference in $h(W)$")
    axs["right mid"].grid(linestyle=':')
    axs["right mid"].set_xlim(X_low,X_high)

    axs["down"].plot(X_plot, Diff_drift, color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    axs["down"].set_title("Difference in Drift of Continuation Payoff $W$")
    axs["down"].grid(linestyle=':')
    axs["down"].set_xlim(X_low,X_high)
    axs["down"].legend()
    axs["down"].set_xlabel("$W$")

plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_Diff_Stop_{}.pdf'.format(len(xiarr)))

plt.close('all')        
            


fig = plt.figure(figsize=(16, 9))


for id in range(len(slopearr)):

    guess = slopearr[id]
    xi = xiarr[id]
    color_one = colors[id % len(colors)]

    def ODE_dis(W, y):
        F, V = y

        F_update = V

        if V < 0:
            c = (V/2)**2
        else:
            c = 0

        a_orig = 2/5 + 4/25 * V + 2*F+2*c-2*V*W + 2*V*np.sqrt(c)

        a = a_orig * (a_orig > 0) + 0*(a_orig <= 0)

        h = -1/xi*V*(a+2/5)*sigma

        upper = F - a + c - V*(W-np.sqrt(c)+a**2/2 + 2/5 * a)
        upper += -xi/2*h**2 -V*(a+2/5)*sigma*h
        
        lower = r*(a+2/5)**2 * sigma**2/2

        V_update = upper/lower

        return [F_update, V_update]



    sol = solve_ivp(ODE_dis, t_span=(0, 1), y0=[
        0, guess], method="DOP853", max_step=0.0001)


    X_plot = sol.t
    ODE_F = sol.y[0]
    ODE_V = sol.y[1]



    ODE_X = X_plot[1:-1]
    ODE_0dF = ODE_F[1:-1]
    ODE_dF = (ODE_F[2:]-ODE_F[0:-2])/(2*(X_plot[2]-X_plot[1]))
    ODE_ddF = (ODE_F[2:]+ODE_F[0:-2]-2*ODE_F[1:-1])/(X_plot[2]-X_plot[1])**2



    c_orig = (ODE_dF/2)**2 * (ODE_dF < 0)

    c = c_orig

    a_orig = 2/5 + 4/25 * ODE_dF + 2*ODE_0dF + \
        2*c-2*ODE_dF*ODE_X + 2*ODE_dF*np.sqrt(c)

    a = a_orig * (a_orig > 0) + 0*(a_orig <= 0)


    h = -1/xi*ODE_dF*(a+2/5)*sigma


    upper = ODE_0dF - a + c - ODE_dF*(ODE_X-np.sqrt(c)+a**2/2 + 2/5 * a)
    upper += -xi/2*h**2 -ODE_dF*(a+2/5)*sigma*h

    lower = r*(a+2/5)**2 * sigma**2/2

    ODE_error = upper/lower - ODE_ddF

    plt.plot(ODE_X, np.log(abs(ODE_error))/np.log(10), color = color_one, label = "$\\xi={}$".format(xiarr[id]))
    plt.xlabel('$W$')
    plt.legend()

plt.savefig('./Figure/' +savefolder+ '/' + figureName + '_ODEerror_{}.pdf'.format(len(xiarr)))

plt.close('all')       