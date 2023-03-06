# to follow a straight line 'contour'
# mainly for debugging, a dummy example
import os
import sys

CURRENT_PATH = os.getcwd()
print(CURRENT_PATH)
sys.path.append(CURRENT_PATH)

from linear_mpcc.mpcc_solver import mpcc_solver
from linear_mpcc.contour import Contour
from linear_mpcc.bicycle_model import ROBOT_STATE
from linear_mpcc.config import Param
from linear_mpcc.plot import visualization
import numpy as np
import matplotlib.pyplot as plt


def set_params():
    param_dict = {"dt": 0.1,
                "N": 20,
                "Q": np.diag([2.0, 1.0]),
                "P": np.diag([2.0, 1.0]),
                "q": np.array([1.0]),
                "Ru": np.diag([0.1, 0.1]),
                "Rv": np.diag([0.1]),
                "C2": 2.5
                }

    param = Param(param_dict)
    return param

def visualize(robot_state_real,robot_ctrl_real,param):

    dt = param.dt

    fig1, axs1 = plt.subplots(2, 1, figsize=(5, 12))
    fig2, axs2 = plt.subplots(3, 1, figsize=(5, 12))

    # for controls
    # acceleration
    axs1[0].plot(robot_ctrl_real[0][:])
    axs1[0].set_title("Control sequence optimized by MPCC")
    axs1[0].set_xlabel('timestep') # later change to time (divide by dt)
    axs1[0].set_ylabel('acc (m/s^2)')
    # steering change
    axs1[1].plot(robot_ctrl_real[1][:])
    axs1[1].set_xlabel('timestep') # later change to time (divide by dt)
    axs1[1].set_ylabel('d_delta (rad/s)')

    # for states
    axs2[0].set_title("Robot state sequence history")
    axs2[0].plot(robot_state_real[0][:])
    axs2[0].set_xlabel('timestep') # later change to time (divide by dt)
    axs2[0].set_ylabel('x (m)')
    axs2[1].plot(robot_state_real[1][:])
    axs2[1].set_xlabel('timestep') # later change to time (divide by dt)
    axs2[1].set_ylabel('y (m)')
    axs2[2].plot(robot_state_real[2][:])
    axs2[2].set_ylabel('yaw (rad)')
    axs2[2].set_xlabel('timestep') # later change to time (divide by dt)

    plt.show()

def main():

    path = [[0,i*0.1] for i in range(100)] # x constantly 0, y from 0 to 10
    goalx,goaly = path[-1][0],path[-1][1]

    contour = Contour(path)

    robot_state = ROBOT_STATE(.2,0,np.pi/2,0,0)  
    param = set_params()

    robot_x_real,robot_y_real,robot_yaw_real = [],[],[] # real robot state history - x,y,yaw
    robot_acc_real,robot_ddelta_real = [],[] # real robot control history - acc, ddelta

    ctrl = np.zeros((1,2)) # control array as in state space model

    STEP = 0
    while True:
        # mpcc opt
        ctrl = mpcc_solver(robot_state, contour, param)
        # append ctrl history
        robot_acc_real.append(ctrl[0])
        robot_ddelta_real.append(ctrl[1])
        
        # append state history
        robot_x_real.append(robot_state.x)
        robot_y_real.append(robot_state.y)
        robot_yaw_real.append(robot_state.yaw)

        # state update (transition with nonlinear simulation)
        robot_state.state_update(acc=ctrl[0], deltadot=ctrl[1], param=param)

        visualization(robot_state, contour)

        terminal_err = np.linalg.norm([robot_state.x-goalx,robot_state.y-goaly])

        if terminal_err <= .5:
            print("Goal reached.")
            break
        
        STEP += 1
        if STEP == 100:
            print("Haven't reach goal after 100 MPCC steps. Breaking out...")
            break

    # postprocessing
    visualize([robot_x_real,robot_y_real,robot_yaw_real],[robot_acc_real,robot_ddelta_real],param)
    
if __name__ == '__main__':
    main()
