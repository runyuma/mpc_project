import numpy as np
import matplotlib.pyplot as plt

SHOW_PATH = True
SHOW_ERROR = True
SHOW_STATE = True
SHOW_CONTROL = True
SHOW_TERMINAL = True
group = "group2/"
files = [group+"dataN4.npz",group+"dataN8.npz",group+"dataN12.npz"]
labels = ["N=4","N=8","N=12"]
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
    ax1.plot(x, y, label="contour path")
    for ind,file in enumerate(files):
        data = np.load(file)
        x = data["robot_states"][:, 0]
        y = data["robot_states"][:, 1]
        ax1.plot(x, y, label=labels[ind])
    ax1.legend()
    ax1.set_aspect(1)
    ax1.set_title("path")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
if SHOW_ERROR:
    fig,ax2 = plt.subplots(3,1,figsize=(8,18))
    for ind,file in enumerate(files):
        data = np.load(file)
        ec = data["error"][:,0]
        el = data["error"][:,1]
        eh = data["error"][:,2]
        ax2[0].plot(ec, label=labels[ind])
        ax2[1].plot(el, label=labels[ind])
        ax2[2].plot(eh, label=labels[ind])
    for i in range(3):
        ax2[i].legend()
        ax2[i].set_xlabel("timestep")
    ax2[0].set_ylabel("lateral error/M")
    ax2[0].set_ylim(-1,1)
    ax2[1].set_ylabel("longitudinal error/M")
    ax2[1].set_ylim(0, 2)
    ax2[2].set_ylabel("heading error/Rads")
    ax2[2].set_ylim(-1, 1)

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
plt.show()