import numpy as np
import matplotlib.pyplot as plt

SHOW_PATH = True
SHOW_ERROR = True
SHOW_STATE = True
SHOW_CONTROL = True
SHOW_TERMINAL = True

group = "group1/"
files = [group+"data0.002.npz",group+"data0.02.npz",group+"data0.2.npz"]
labels = ["Q = 0.002","Q = 0.02","Q = 0.2"]
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
    fig,ax2 = plt.subplots(2,1,figsize=(8,18))
    for ind,file in enumerate(files):
        data = np.load(file)
        ec = data["error"][:,0]
        eh = data["error"][:,2]
        ax2[0].plot(ec, label=labels[ind])
        ax2[1].plot(eh, label=labels[ind])
    for i in range(2):
        ax2[i].legend()

    ax2[0].set_title("Lateral error for different Q")
    ax2[0].set_ylabel("lateral error/M")
    ax2[0].set_ylim(-1,1)
    ax2[1].set_title("Heading error for different Q")
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

plt.show()