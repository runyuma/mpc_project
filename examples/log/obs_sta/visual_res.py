import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
SHOW_PATH = True
SHOW_ERROR = True
SHOW_STATE = True
SHOW_CONTROL = True
SHOW_TERMINAL = True
SHOW_LYAPUNOV = True

group = "group1/"
files = [group+"data_noter.npz",group+"data_ter.npz"]
labels = ["without Terminal cost","with Terminal cost",]
fig,ax0 = plt.subplots()
data = np.load(files[1])
x = data["path"][:, 0]
y = data["path"][:, 1]
ax0.plot(x, y, label="reference path")
ax0.set_aspect(1)
ax0.set_title("path")
ax0.set_xlabel("x")
ax0.set_ylabel("y")
x = data["robot_states"][:, 0]
y = data["robot_states"][:, 1]
ax0.plot(x, y, label="robot trajectory")
ax0.legend()


if SHOW_PATH:
    fig,ax1 = plt.subplots()
    data = np.load(files[0])
    x = data["path"][:, 0]
    y = data["path"][:, 1]
    ax1.plot(x, y, label="robot trajectory")

    obs_x = data["obstacle_states"][:,0]
    obs_y = data["obstacle_states"][:,1]
    ax1.scatter(obs_x,obs_y,marker="x",s=100,label="obstacle")
    for ind,file in enumerate(files):
        data = np.load(file)
        x = data["robot_states"][:, 0]
        y = data["robot_states"][:, 1]
        ax1.plot(x, y, label="vehicle path "+labels[ind])
    ax1.legend()
    ax1.set_aspect(1)
    ax1.set_title("Vehicle trajectory and reference path")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
if SHOW_ERROR:
    fig,ax2 = plt.subplots(2,1,figsize=(8,18))
    for ind,file in enumerate(files):
        data = np.load(file)
        ec = data["error"][:,0]
        eh = data["error"][:,2]
        ax2[0].plot(ec, label=labels[ind])
        ax2[1].plot(eh, label=labels[ind])
    for i in range(2):
        ax2[i].legend()

    ax2[0].set_title("Lateral error for different beta")
    ax2[0].set_ylabel("lateral error/M")
    ax2[0].set_ylim(-2,2)
    ax2[1].set_title("Heading error for different beta")
    ax2[1].set_ylabel("heading error/Rads")
    ax2[1].set_ylim(-1, 1)

if SHOW_CONTROL:
    fig,ax3 = plt.subplots(2,1,figsize=(8,6))
    for ind,file in enumerate(files):
        data = np.load(file)
        acc = data["inputs"][0]
        delta = data["inputs"][1]
        ax3[0].plot(acc, label=labels[ind])
        ax3[1].plot(delta, label=labels[ind])
    for i in range(2):
        ax3[i].legend()
    ax3[0].set_title("vehicle accerlation")
    ax3[1].set_title("steering angle")
    ax3[0].set_ylabel("acc/M/s^2")
    ax3[1].set_ylabel("delta/Rads")

if SHOW_LYAPUNOV:
    fig,ax4 = plt.subplots(3,1,figsize=(6,6))
    fig.tight_layout()
    data = np.load(files[-1])

    cost = data["costs"][0]
    ax4[0].plot(cost[1:], label=r'Cost function $V_N$')
    ax4[0].legend()
    ax4[0].set_title("Cost function")

    ter_cost = (data["costs"][1]-data["costs"][3])
    stage_cost = data["costs"][2]
    ax4[1].plot(ter_cost[1:], label=r'Decrease in terminal cost $V_f(x)-V_f(f(x,K(x)))$')
    ax4[1].plot(stage_cost[1:], label=r'Terminal stage cost $l(x,u)$')
    ax4[1].legend(prop = {'size':12})
    ax4[1].set_title("Decrease in Terminal cost and Terminal stage cost")


    ax4[2].plot(ter_cost[1:]/100, label=r'Decrease in terminal cost without scaling $\frac{V_f(x)-V_f(f(x,K(x)))}{\beta}$')
    ax4[2].plot(stage_cost[1:], label=r'Terminal stage cost $l(x,u)$')
    ax4[2].legend(prop = {'size':12})
    ax4[2].set_title("Decrease in Terminal cost without scaling and Terminal stage cost")


plt.show()
