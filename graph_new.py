import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import csv
import pandas as pd



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

n_plota = 600  # Points on plot grid for each dimension
n_plotz = 600  # Points on plot grid for each dimension

def u_deriv(c):
    return c**(-gamma)



plt.style.use('classic')
plt.rcParams["savefig.bbox"] = "tight"
# plt.rcParams["figure.figsize"] = (10,8)
# plt.rcParams["figure.dpi"] = 500
plt.rcParams["font.size"] = 16
plt.rcParams['text.usetex']=True
# plt.rcParams["legend.frameon"] = True
# plt.rcParams["lines.linewidth"] = 5


aspace = np.linspace(-0.02, 4, 600)
zspace = np.linspace(zmean*0.8, zmean*1.2, 600)
A, Z = np.meshgrid(aspace, zspace)


Moll_Vaforward600 = pd.read_csv(
    "./MollData/Va_f_600,600_Shrink.csv", header=None)
Moll_Vaforward10000 = pd.read_csv("./MollData/Va_f_10000,600_Shrink.csv", header=None)
Moll_Vaforward40000 = pd.read_csv("./MollData/Va_f_40000,600_Shrink.csv", header=None)
Moll_Vaforward600 = np.array(Moll_Vaforward600)
Moll_Vaforward10000 = np.array(Moll_Vaforward10000)
Moll_Vaforward40000 = np.array(Moll_Vaforward40000)


fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], Moll_Vaforward600[:, 0],
         label=r'$\partial_a^{Forward} v( \underline a,z)$' + r'$, {Low \ Resolution}$', color='blue')
plt.plot(Z[:, 0], Moll_Vaforward10000[:, 0],
         label=r'$\partial_a^{Forward} v( \underline a,z)$' + r'$, {Medium \ Resolution}$', color='green')
plt.plot(Z[:, 0], Moll_Vaforward40000[:, 0],
         label=r'$\partial_a^{Forward} v( \underline a,z)$' + r'$, {High \ Resolution}$', color='red')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='black')
plt.xlabel('$z$')
plt.ylim(0.75, 1.10)
plt.legend()
plt.savefig('./MollData/Va_FB.pdf', bbox_inches='tight')


Moll_VaUpwind600 = pd.read_csv(
    "./MollData/VaUpwind_600,600_Shrink.csv", header=None)
Moll_VaUpwind10000 = pd.read_csv("./MollData/VaUpwind_10000,600_Shrink.csv", header=None)
Moll_VaUpwind40000 = pd.read_csv("./MollData/VaUpwind_40000,600_Shrink.csv", header=None)
Moll_VaUpwind600 = np.array(Moll_VaUpwind600)
Moll_VaUpwind10000 = np.array(Moll_VaUpwind10000)
Moll_VaUpwind40000 = np.array(Moll_VaUpwind40000)



fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], Moll_Vaforward600[:, 0],
         label=r'$\partial_a^{Upwind} v( \underline a,z)$' + r'$, {Low \ Resolution}$', color='blue')
plt.plot(Z[:, 0], Moll_Vaforward10000[:, 0],
         label=r'$\partial_a^{Upwind} v( \underline a,z)$' + r'$, {Medium \ Resolution}$', color='green')
plt.plot(Z[:, 0], Moll_Vaforward40000[:, 0],
         label=r'$\partial_a^{Upwind} v( \underline a,z)$' + r'$, {High \ Resolution}$', color='red')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='black')
plt.xlabel('$z$')
plt.ylim(0.75, 1.10)
plt.legend()
plt.savefig('./MollData/Va_Upwind.pdf', bbox_inches='tight')



Moll_VaNN_forward = pd.read_csv(
    "./MollData/MollProblem_Va.csv", header=None)
Moll_VaNN_forward = np.array(Moll_VaNN_forward)


fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], Moll_VaNN_forward[:, 0],
         label=r'$\partial_a^{NN} v( \underline a,z)$', color='red')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='black')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r a)$', color = 'red')
# plt.view_init(35, 35)
plt.xlabel('$z$')
plt.ylim(0.75, 1.10)
plt.legend()
# plt.show()
# ax.set_zlabel('$\partial V / \partial a$')
# ax.set_zlabel('Difference')
# ax.set_title('Deep Learning Solution')
plt.savefig('./MollData/Va_NN_forward.pdf', bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], Moll_VaNN_forward[:, 0],
         label=r'$\partial_a^{NN} v( \underline a,z)$', color='red')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], Moll_Vaforward40000[:, 0],
         label=r'$\partial_a^{Forward} v( \underline a,z)$' + r'$, {High \ Resolution}$', color='blue', linestyle='--')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='black')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r a)$', color = 'red')
# plt.view_init(35, 35)
plt.xlabel('$z$')
plt.ylim(0.75, 1.10)
plt.legend()
# plt.show()
# ax.set_zlabel('$\partial V / \partial a$')
# ax.set_zlabel('Difference')
# ax.set_title('Deep Learning Solution')
plt.savefig('./MollData/Va_NN_forward_high.pdf', bbox_inches='tight')
fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], Moll_VaNN_forward[:, 0],
         label=r'$\partial_a^{NN} v( \underline a,z)$', color='red')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], Moll_Vaforward10000[:, 0],
         label=r'$\partial_a^{Forward} v( \underline a,z)$' + r'$, {Medium \ Resolution}$', color='blue', linestyle='--')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='black')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r a)$', color = 'red')
# plt.view_init(35, 35)
plt.xlabel('$z$')
plt.ylim(0.75, 1.10)
plt.legend()
# plt.show()
# ax.set_zlabel('$\partial V / \partial a$')
# ax.set_zlabel('Difference')
# ax.set_title('Deep Learning Solution')
plt.savefig('./MollData/Va_NN_forward_medium.pdf', bbox_inches='tight')
fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], Moll_VaNN_forward[:, 0],
         label=r'$\partial_a^{NN} v( \underline a,z)$', color='red')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], Moll_Vaforward600[:, 0],
         label=r'$\partial_a^{Forward} v( \underline a,z)$' + r'$, {Low \ Resolution}$', color='blue', linestyle='--')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='black')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r a)$', color = 'red')
# plt.view_init(35, 35)
plt.xlabel('$z$')
plt.ylim(0.75, 1.10)
plt.legend()
# plt.show()
# ax.set_zlabel('$\partial V / \partial a$')
# ax.set_zlabel('Difference')
# ax.set_title('Deep Learning Solution')
plt.savefig('./MollData/Va_NN_forward_low.pdf', bbox_inches='tight')




fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], np.maximum(Moll_VaNN_forward[:, 0], u_deriv(Z[:, 0]+r*A[:, 0])),
         label=r'$\partial_a^{NN} v( \underline a,z)$', color='red')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='black')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r a)$', color = 'red')
# plt.view_init(35, 35)
plt.xlabel('$z$')
plt.ylim(0.75, 1.10)
plt.legend()
# plt.show()
# ax.set_zlabel('$\partial V / \partial a$')
# ax.set_zlabel('Difference')
# ax.set_title('Deep Learning Solution')
plt.savefig('./MollData/Va_NN_forward.pdf', bbox_inches='tight')

fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], np.maximum(Moll_VaNN_forward[:, 0], u_deriv(Z[:, 0]+r*A[:, 0])),
         label=r'$\partial_a^{NN} v( \underline a,z)$', color='red')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], Moll_VaUpwind40000[:, 0],
         label=r'$\partial_a^{Upwind} v( \underline a,z)$' + r'$, {High \ Resolution}$', color='blue', linestyle='--')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='black')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r a)$', color = 'red')
# plt.view_init(35, 35)
plt.xlabel('$z$')
plt.ylim(0.75, 1.10)
plt.legend()
# plt.show()
# ax.set_zlabel('$\partial V / \partial a$')
# ax.set_zlabel('Difference')
# ax.set_title('Deep Learning Solution')
plt.savefig('./MollData/Va_NN_upwind_high.pdf', bbox_inches='tight')
fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], np.maximum(Moll_VaNN_forward[:, 0], u_deriv(Z[:, 0]+r*A[:, 0])),
         label=r'$\partial_a^{NN} v( \underline a,z)$', color='red')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], Moll_VaUpwind10000[:, 0],
         label=r'$\partial_a^{Upwind} v( \underline a,z)$' + r'$, {Medium \ Resolution}$', color='blue', linestyle='--')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='black')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r a)$', color = 'red')
# plt.view_init(35, 35)
plt.xlabel('$z$')
plt.ylim(0.75, 1.10)
plt.legend()
# plt.show()
# ax.set_zlabel('$\partial V / \partial a$')
# ax.set_zlabel('Difference')
# ax.set_title('Deep Learning Solution')
plt.savefig('./MollData/Va_NN_upwind_medium.pdf', bbox_inches='tight')
fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], np.maximum(Moll_VaNN_forward[:, 0], u_deriv(Z[:, 0]+r*A[:, 0])),
         label=r'$\partial_a^{NN} v( \underline a,z)$', color='red')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], Moll_VaUpwind600[:, 0],
         label=r'$\partial_a^{Upwind} v( \underline a,z)$' + r'$, {Low \ Resolution}$', color='blue', linestyle='--')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='black')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r a)$', color = 'red')
# plt.view_init(35, 35)
plt.xlabel('$z$')
plt.ylim(0.75, 1.10)
plt.legend()
# plt.show()
# ax.set_zlabel('$\partial V / \partial a$')
# ax.set_zlabel('Difference')
# ax.set_title('Deep Learning Solution')
plt.savefig('./MollData/Va_NN_upwind_low.pdf', bbox_inches='tight')
