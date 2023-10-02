from random import *
from math import *
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import copy
import os
import glob
import csv
import random as rd
import seaborn as sns
import matplotlib.cm as cm

matplotlib.use('Agg')

def index2tuple(pos,i):
    # Return the tuple corresponding to node index i
    return [i for i in pos.keys()][i]

IND = 0

# Definitions of the physical parameters

F0 = 4e-3
Pi = 1e-2
tau_n = F0/Pi
l0 = 1

D = 0.001

# Definitions of the simulation parameters
dt = 0.1
num = 30000
pas = 20 # For data save
digit = 4

# Structure
N = 130
d = 5

r_cut = 1.5
r0 = r_cut /2.5
E0 = 10

r_cut_int = 0.7
r0_int = r_cut_int /2.5
E0_int = 1

G = nx.Graph(directed=False)
G.add_node((0,0))
pos = {}

pos[(0,0)]= (0, 0)

i = 1
while i != N :

    theta_r = rd.uniform(0, 2*np.pi)
    R_r = (d + r_cut/4) * np.sqrt(rd.uniform(0, 1))

    pos[(i,0)]= (R_r * np.cos(theta_r), R_r * np.sin(theta_r))

    for k in range(i):
        OM = ((R_r * np.cos(theta_r) - pos[index2tuple(pos,k)][0])**2 + (R_r * np.sin(theta_r) - pos[index2tuple(pos,k)][1])**2)**(1/2)

        if OM < r_cut_int:
            i = i - 1
            break

    i = i + 1
    print(i)

X = []
Y = []
for i in range(N):
    X.append(pos[index2tuple(pos,i)][0])
    Y.append(pos[index2tuple(pos,i)][1])
X = np.array(X[0:], dtype=float)
Y = np.array(Y[0:], dtype=float)

n = np.random.uniform(0, 2*np.pi, N)

print('Nombre total de noeuds ='+str(N))
print('Nombre de liens ='+str(len([e for e in G.edges])))
print('')

# Pratical functions for graph use


def struct1():
    """Fonction définissant les structures/obstacles de l'environnement complexe"""
    n = 300  #nombre de point par mur
    R = d +r_cut

    xl = -d  #left walll
    xr = d  #right walll
    yu = d   #up walll
    yd = -d  #down walll

    ones = np.ones((n,))

    #murs
    """
    up_wall = [np.linspace(xl, xr, n), yu * ones]
    down_wall = [np.linspace(xl, xr, n), yd * ones]
    left_wall = [xl * ones, np.linspace(yd, yu, n)]
    right_wall = [xr * ones, np.linspace(yd, yu, n)]

    structure = [up_wall, down_wall, left_wall, right_wall]
    """
    circle = [R * np.cos(np.linspace(0, 2*np.pi, n)), R * np.sin(np.linspace(0, 2*np.pi, n))]
    structure = [circle]

    structure2 = []
    for i in range(2):
        pos = []
        for vec in structure:
            pos = np.concatenate((pos, vec[i]))

        structure2.append(pos)

    return structure, d, np.array(structure2)

def tuple2index(pos,tuple):
    # Return the index corresponding to node tuple
    return [i for i in pos.keys()].index(tuple)

def norm1(x1, x2, y1, y2):
    return ((x1 - x2)**2 + (y1-y2)**2)**(1/2)

def WCAForce(i,X,Y):
    """Fonction accélération, renvoie un vecteur (vx, vy, ax, ay) et prend
    en entré un vecteur (x, y, vx, vy) et le temps"""

    OM = (( X[i] - structure2[0])**2 + ( Y[i] - structure2[1])**2)**(1/2)
    THETA = np.arctan2(structure2[1] -  Y[i], structure2[0] - X[i])

    F = 6 * E0/r0 * (2*(OM/r0)**(-13) - (OM/r0)**(-7)) * (OM < r_cut)
    Fx = np.sum(F * np.cos(THETA))
    Fy = np.sum(F * np.sin(THETA))

    return Fx, Fy

def WCAForce_int(i,X,Y):
    """Fonction accélération, renvoie un vecteur (vx, vy, ax, ay) et prend
    en entré un vecteur (x, y, vx, vy) et le temps"""

    OM = (( X[i] - X)**2 + ( Y[i] - Y)**2)**(1/2)
    OM = OM[np.arange(len(OM))!=i]
    THETA = np.arctan2(Y -  Y[i], X - X[i])
    THETA = THETA[np.arange(len(THETA))!=i]

    F = 6 * E0_int/r0_int * (2*(OM/r0_int)**(-13) - (OM/r0_int)**(-7)) * (OM < r_cut_int)
    Fx = np.sum(F * np.cos(THETA))
    Fy = np.sum(F * np.sin(THETA))

    return Fx, Fy

def fun(t, y):
    # y is a 3-rows 1D vector, return a 3-rows 1D vector

    X_ = y[:N]
    Y_ = y[N:2*N]
    n_ = y[2*N:]

    rhs = np.zeros([3*N])

    for i in range(N):

        Fx, Fy = WCAForce(i,X_,Y_)
        WFx, WFy = WCAForce_int(i,X_,Y_)
        #print(WFx)

        F = sqrt((Fx + WFx)**2 + (Fy + WFy)**2)
        if Fx + WFx!= 0:
            theta = atan2(Fy + WFy,Fx + WFx)
        else:
            if Fy + WFy > 0:
                theta = pi/2
            else:
                theta = -pi/2

        rhs[i] = F0*cos(n_[i]) + F*cos(theta)
        rhs[N+i] = F0*sin(n_[i]) + F*sin(theta)
        rhs[2*N+i] = (F/tau_n)*sin(theta - n_[i])

    #print(t)

    return rhs

def name(i,digit):

    i = str(i)

    while len(i)<digit:
        i = '0'+i

    i = 'img/'+i+'.png'

    return(i)

########
# Main #
########

structure, d, structure2 = struct1()

Xt = np.zeros((num, N))
Yt = np.zeros((num, N))
nt = np.zeros((num, N))

Xt[0] = X
Yt[0] = Y
nt[0] = n

# Solve

extension="img/*.png"
for f in glob.glob(extension):
  os.remove(f)

for it in range(num-1):

    t_span = np.array([it, it+dt]) # only one time step
    y0 = np.concatenate([Xt[it,:], Yt[it,:], nt[it,:]], axis=0) # solve the system for one time step

    print(it/num)

    #sol = solve_ivp(fun, t_span, y0, method='RK45', t_eval=[t+dt], first_step = dt)
    sol = solve_ivp(fun, t_span, y0, method='RK45', t_eval=[it+dt])

    # Update variables in the "new" arrays

    Xt[it+1, :] = np.transpose(sol.y[:N,:])
    Yt[it+1, :] = np.transpose(sol.y[N:2*N,:])
    nt[it+1, :] = np.transpose(sol.y[2*N:,:]%(2*pi))

    nt[it+1] = nt[it+1] + sqrt(2*D*dt)*np.random.normal(loc=0.0, scale=1.0,
        size=N)

#print(sol.t)
#print(np.shape(sol.y))

print("Computation ended correctly")

for it in range(0,num,pas):

    print(it/num)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    """
    for s in structure:
        ax.plot(s[0], s[1], "-", color = "blue", markersize = 5)
    """
    """
    for i in range(len(structure[0][0])):
        circle_a = plt.Circle((structure[0][0][i], structure[0][1][i]), r_cut, ec = "red", fc = "none", zorder = -1)
        ax.add_patch(circle_a)
    """
    colormap = sns.color_palette("hls", as_cmap=True)

    circle_r = plt.Circle((0, 0), d + r_cut/4, ec = "blue", fc = "none", zorder = -1)
    ax.add_patch(circle_r)

    for i in range(N):
        circle1 = plt.Circle((Xt[it, i], Yt[it, i]), r_cut_int/2, ec = "orange", fc = "white", zorder = 1e5)
        ax.add_patch(circle1)

    plt.quiver(Xt[it,:N], Yt[it,:N], np.cos(nt[it,:N]), np.sin(nt[it,:N]), pivot="mid", scale = 28.0, color=colormap(nt[it,:]/(2*np.pi)), zorder=10**6, edgecolor='k', linewidth = 0.2, width=0.01, headlength=2.5, headaxislength=2.5,headwidth=2.5)

    xmin, xmax, ymin, ymax = ax.axis("square")

    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    #ax.xticks([])
    #ax.yticks([])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    name_pic = name(int(it/pas),digit)
    plt.savefig(name_pic, bbox_inches='tight', dpi=300)
    plt.close(fig)

# ffmpeg -i img/%05d.png -r 30 -pix_fmt yuv420p hexagon.mp4
# ffmpeg -r 30 -i img/%05d.png -vcodec libx264 -y -an active_square_test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" (if 'width not divisible by two' error)
