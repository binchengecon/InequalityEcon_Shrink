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



# Moll_Vacenter = pd.read_csv("./MollData/Va_center.csv", header = None)

# fig = plt.figure(figsize=(16, 9))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(A, Z, Moll_Vacenter, cmap='viridis')
# ax.view_init(35, 35)
# ax.set_xlabel('$a$')
# ax.set_ylabel('$z$')
# ax.set_zlim(0.75,1.10)
# # ax.set_zlabel('$\partial V / \partial a$')
# # ax.set_zlabel('Difference')
# # ax.set_title('Deep Learning Solution')
# plt.savefig('./MollData/Va_center_py.png',bbox_inches='tight')


Moll_Va = pd.read_csv("./MollData/Va_Upwind.csv", header = None)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(A, Z, Moll_Va, cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$a$')
ax.set_ylabel('$z$')
ax.set_zlim(0.75,1.10)
# ax.set_zlabel('$\partial V / \partial a$')
# ax.set_zlabel('Difference')
# ax.set_title('Deep Learning Solution')
plt.savefig('./MollData/VaUpwind_py.png',bbox_inches='tight')

# Moll_Vaa_center = pd.read_csv("./MollData/Vaa_center.csv", header = None)

# fig = plt.figure(figsize=(16, 9))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(A, Z, Moll_Vaa_center, cmap='viridis')
# ax.view_init(35, 35)
# ax.set_xlabel('$a$')
# ax.set_ylabel('$z$')
# # ax.set_zlim(0.75,1.10)
# # ax.set_zlabel('$\partial V / \partial a$')
# # ax.set_zlabel('Difference')
# # ax.set_title('Deep Learning Solution')
# plt.savefig('./MollData/Vaa_center_py.png',bbox_inches='tight')




# Moll_Vacenter = pd.read_csv("./MollData/Va_center.csv", header= None)
# Moll_Vacenter = np.array(Moll_Vacenter)
# Moll_Va = pd.read_csv("./MollData/Va_Upwind.csv", header = None)
# Moll_Va = np.array(Moll_Va)

# Moll_Va_NN = pd.read_csv("./MollData/MollProblem_Va.csv", header=None,index_col=None)
# Moll_Va_NN = np.array(Moll_Va_NN)
# Moll_Va_NN = Moll_Va_NN[1:,1:]
plt.style.use('classic')
plt.rcParams["savefig.bbox"] = "tight"
# plt.rcParams["figure.figsize"] = (10,8)
# plt.rcParams["figure.dpi"] = 500
plt.rcParams["font.size"] = 16
plt.rcParams['text.usetex']=True
# plt.rcParams["legend.frameon"] = True
# plt.rcParams["lines.linewidth"] = 5

# fig = plt.figure(figsize=(16, 9))
# plt.plot(Z[:, 0], Moll_Vacenter[:, 0], label=r'$\partial_a^{Forward} v( \underline a,z)$', color='black')
# # plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r \underline a )$', color='red')
# # plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
# #          label=r'$u^\prime(z + r a)$', color = 'red')
# # plt.view_init(35, 35)
# plt.xlabel('$z$')
# plt.ylim(0.75, 1.10)
# plt.legend()
# # plt.show()
# # ax.set_zlabel('$\partial V / \partial a$')
# # ax.set_zlabel('Difference')
# # ax.set_title('Deep Learning Solution')
# plt.savefig('./MollData/Va_analysis_FB2.pdf', bbox_inches='tight')

# fig = plt.figure(figsize=(16, 9))
# # plt.plot(Z[:,0], Moll_Vacenter[:,0],label='$\partial_a v(a,z)$: Forward')
# plt.plot(Z[:,0], Moll_Va[:,0],label=r'$\partial_a^{Upwind} v( \underline a,z)$', color='black')
# # plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]), label=r'$u^\prime(z + r \underline{a})$')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r \underline a )$', color='red')
# # plt.view_init(35, 35)
# plt.xlabel('$z$')
# plt.ylim(0.75, 1.10)
# plt.legend()
# # ax.set_zlabel('$\partial V / \partial a$')
# # ax.set_zlabel('Difference')
# # ax.set_title('Deep Learning Solution')
# plt.savefig('./MollData/Va_analysis_UB2.pdf', bbox_inches='tight')

plt.style.use('classic')
plt.rcParams["savefig.bbox"] = "tight"
# plt.rcParams["figure.figsize"] = (10,8)
# plt.rcParams["figure.dpi"] = 500
plt.rcParams["font.size"] = 16
plt.rcParams['text.usetex']=True
# plt.rcParams["legend.frameon"] = True
# plt.rcParams["lines.linewidth"] = 5

# fig = plt.figure(figsize=(16, 9))
# # plt.plot(Z[:,0], Moll_Vacenter[:,0],label='$\partial_a v(a,z)$: Forward')
# plt.plot(Z[:, 0], Moll_Va_NN.reshape(n_plot, n_plot)[:, 0],
#          label=r'$\partial_a^{NN} v( \underline a,z)$', color='black')
# # plt.plot(Z[:,0], Moll_Va[:,0],label=r'$\partial_a^{Upwind} v( \underline a,z)$', color='black')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r \underline a )$', color='red')
# # plt.view_init(35, 35)
# plt.xlabel('$z$')
# # plt.ylim(0.75, 1.10)
# plt.legend()

# plt.savefig('./MollData/Va_slice.pdf', bbox_inches='tight')


# Moll_Vaforward400 = pd.read_csv("./MollData/Va_f_400_40.csv", header= None)
# Moll_Vaforward4000 = pd.read_csv("./MollData/Va_f_4000_40.csv", header= None)
# Moll_Vaforward40000 = pd.read_csv("./MollData/Va_f_40000_40.csv", header= None)
# Moll_Vaforward400 = np.array(Moll_Vaforward400)
# Moll_Vaforward4000 = np.array(Moll_Vaforward4000)
# Moll_Vaforward40000 = np.array(Moll_Vaforward40000)

# aspace = np.linspace(-0.02, 4, 400)
# zspace = np.linspace(zmean*0.8, zmean*1.2, 40)
# A, Z = np.meshgrid(aspace, zspace)

# fig = plt.figure(figsize=(16, 9))
# plt.plot(Z[:, 0], Moll_Vaforward400[:, 0],
#          label=r'$\partial_a^{Forward} v( \underline a,z)$', color='black')
# # plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r \underline a )$', color='red')
# # plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
# #          label=r'$u^\prime(z + r a)$', color = 'red')
# # plt.view_init(35, 35)
# plt.xlabel('$z$')
# plt.ylim(0.75, 1.10)
# plt.legend()
# # plt.show()
# # ax.set_zlabel('$\partial V / \partial a$')
# # ax.set_zlabel('Difference')
# # ax.set_title('Deep Learning Solution')
# plt.savefig('./MollData/Va_analysis_FB2_400.pdf', bbox_inches='tight')


# aspace = np.linspace(-0.02, 4, 4000)
# zspace = np.linspace(zmean*0.8, zmean*1.2, 40)
# A, Z = np.meshgrid(aspace, zspace)

# fig = plt.figure(figsize=(16, 9))
# plt.plot(Z[:, 0], Moll_Vaforward4000[:, 0],
#          label=r'$\partial_a^{Forward} v( \underline a,z)$', color='black')
# # plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r \underline a )$', color='red')
# # plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
# #          label=r'$u^\prime(z + r a)$', color = 'red')
# # plt.view_init(35, 35)
# plt.xlabel('$z$')
# plt.ylim(0.75, 1.10)
# plt.legend()
# # plt.show()
# # ax.set_zlabel('$\partial V / \partial a$')
# # ax.set_zlabel('Difference')
# # ax.set_title('Deep Learning Solution')
# plt.savefig('./MollData/Va_analysis_FB2_4000.pdf', bbox_inches='tight')


# aspace = np.linspace(-0.02, 4, 40000)
# zspace = np.linspace(zmean*0.8, zmean*1.2, 40)
# A, Z = np.meshgrid(aspace, zspace)

# fig = plt.figure(figsize=(16, 9))
# plt.plot(Z[:, 0], Moll_Vaforward40000[:, 0],
#          label=r'$\partial_a^{Forward} v( \underline a,z)$', color='black')
# # plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
# plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
#          label=r'$u^\prime(z + r \underline a )$', color='red')
# # plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
# #          label=r'$u^\prime(z + r a)$', color = 'red')
# # plt.view_init(35, 35)
# plt.xlabel('$z$')
# plt.ylim(0.75, 1.10)
# plt.legend()
# # plt.show()
# # ax.set_zlabel('$\partial V / \partial a$')
# # ax.set_zlabel('Difference')
# # ax.set_title('Deep Learning Solution')
# plt.savefig('./MollData/Va_analysis_FB2_40000.pdf', bbox_inches='tight')


Moll_Vaforward600 = pd.read_csv(
    "./MollData/Va_f_600,600_Shrink.csv", header=None)
Moll_Vaforward10000 = pd.read_csv("./MollData/Va_f_10000,600_Shrink.csv", header=None)
Moll_Vaforward600 = np.array(Moll_Vaforward600)
Moll_Vaforward10000 = np.array(Moll_Vaforward10000)


# pd.DataFrame(Moll_Vaforward600[:,0]).to_csv("./MollData/Va_f_600,600_Shrink.csv",header=False,index=False)    
# pd.DataFrame(Moll_Vaforward10000[:,0]).to_csv("./MollData/Va_f_10000,600_Shrink.csv",header=False,index=False)    

aspace = np.linspace(-0.02, 4, 600)
zspace = np.linspace(zmean*0.8, zmean*1.2, 600)
A, Z = np.meshgrid(aspace, zspace)


fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], Moll_Vaforward600[:, 0],
         label=r'$\partial_a^{Forward} v( \underline a,z)$', color='black')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='red')
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
plt.savefig('./MollData/Va_FB_600,600.pdf', bbox_inches='tight')

aspace = np.linspace(-0.02, 4, 10000)
zspace = np.linspace(zmean*0.8, zmean*1.2, 600)
A, Z = np.meshgrid(aspace, zspace)

fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], Moll_Vaforward10000[:, 0],
         label=r'$\partial_a^{Forward} v( \underline a,z)$', color='black')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='red')
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
plt.savefig('./MollData/Va_FB_10000,600.pdf', bbox_inches='tight')


Moll_Vaforward600 = pd.read_csv("./MollData/VaUpwind_600,600_Shrink.csv", header=None)
Moll_Vaforward10000 = pd.read_csv("./MollData/VaUpwind_10000,600_Shrink.csv", header=None)
Moll_Vaforward600 = np.array(Moll_Vaforward600)
Moll_Vaforward10000 = np.array(Moll_Vaforward10000)
# pd.DataFrame(Moll_Vaforward600[:,0]).to_csv("./MollData/VaUpwind_600,600_Shrink.csv",header=False,index=False)    
# pd.DataFrame(Moll_Vaforward10000[:,0]).to_csv("./MollData/VaUpwind_10000,600_Shrink.csv",header=False,index=False)    

aspace = np.linspace(-0.02, 4, 600)
zspace = np.linspace(zmean*0.8, zmean*1.2, 600)
A, Z = np.meshgrid(aspace, zspace)


fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], Moll_Vaforward600[:, 0],
         label=r'$\partial_a^{Upwind} v( \underline a,z)$', color='black')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='red')
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
plt.savefig('./MollData/Va_Upwind_600,600.pdf', bbox_inches='tight')

aspace = np.linspace(-0.02, 4, 10000)
zspace = np.linspace(zmean*0.8, zmean*1.2, 600)
A, Z = np.meshgrid(aspace, zspace)

fig = plt.figure(figsize=(16, 9))
plt.plot(Z[:, 0], Moll_Vaforward10000[:, 0],
         label=r'$\partial_a^{Upwind} v( \underline a,z)$', color='black')
# plt.plot(Z[:,0], Moll_Va[:,0],label='$\partial_a v(a,z)$: Upwind')
plt.plot(Z[:, 0], u_deriv(Z[:, 0]+r*A[:, 0]),
         label=r'$u^\prime(z + r \underline a )$', color='red')
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
plt.savefig('./MollData/Va_Upwind_10000,600.pdf', bbox_inches='tight')
