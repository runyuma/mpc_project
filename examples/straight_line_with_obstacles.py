# to follow a straight line 'contour'
# mainly for debugging, a dummy example
import os
import sys

CURRENT_PATH = os.getcwd()
sys.path.append(CURRENT_PATH)

from linear_mpcc.mpcc_solver import mpcc_solver
from linear_mpcc.contour import Contour
from linear_mpcc.bicycle_model import ROBOT_STATE
from linear_mpcc.config import Param
from linear_mpcc.plot import visualization,mpc_visualization
from linear_mpcc.dyn_obst import DynamicObstacles
import numpy as np
import matplotlib.pyplot as plt


def set_params():
    param_dict = {"dt": 0.5,
                "N": 5,
                "Q": np.diag([4.0, 20.0]),
                "P": 2 * np.diag([4.0, 20.0]),
                "q": np.array([10.0]),
                "Ru": np.diag([0.1, 1]),
                "Rv": np.diag([0.1]),
                "C2": 2.5,
                "max_vel":2.0,
                "max_acc": 2.0,
                "max_deltadot": 0.3,
                "max_delta": 0.3,
                "use_terminal_cost": True,
                "use_prev_optim_ctrl": True
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

    # linear path
    path = [[0,i*0.1] for i in range(400)] # x constantly 0, y from 0 to 10
    robot_state = ROBOT_STATE(.5, 0, np.pi / 2 + 0.2, 0.1, 0)

    goalx,goaly = path[-1][0],path[-1][1]
    contour = Contour(path)
    param = set_params()

    robot_x_real,robot_y_real,robot_yaw_real = [],[],[] # real robot state history - x,y,yaw
    robot_acc_real,robot_ddelta_real = [],[] # real robot control history - acc, ddelta

    ctrl = np.zeros((1,2)) # control array as in state space model
    prev_optim_ctrl = np.zeros((2,param.N)) # store the previous MPC's optimal control solution for next problem's state transitions
    prev_optim_theta = -contour.path_length * np.ones((1,param.N))   # at the start, assume it is the original value

    # obstacles
    obstacles = []
    obs1 = DynamicObstacles(xo=-5.,yo=8.,phi=0.,alpha=4.,beta=3.,dx=.3, dy=.2, dphi=2.)
    obstacles.append(obs1)

    STEP = 0
    while True:
        # mpcc opt
        ctrl,pred_states,theta = mpcc_solver(robot_state=robot_state, obstacles_state=obstacles, contour=contour, param=param,
                                            prev_optim_ctrl=prev_optim_ctrl, prev_optim_theta=prev_optim_theta)
        prev_optim_ctrl = ctrl
        prev_optim_theta = theta

        one_step_ctrl = (ctrl[:,0]).reshape(2,)

        # append ctrl history
        robot_acc_real.append(one_step_ctrl[0])
        robot_ddelta_real.append(one_step_ctrl[1])
        
        # append state history
        robot_x_real.append(robot_state.x)
        robot_y_real.append(robot_state.y)
        robot_yaw_real.append(robot_state.yaw)

        # state update (transition with nonlinear simulation)
        robot_state.state_update(acc=one_step_ctrl[0], deltadot=one_step_ctrl[1], param=param)

        for obs in obstacles:
            obs.state_update()

        visualization(robot_state, contour, obstacles)
        mpc_visualization(pred_states)
        plt.pause(0.01)

        terminal_err = np.linalg.norm([robot_state.x-goalx,robot_state.y-goaly])

        if terminal_err <= 0.5:
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
