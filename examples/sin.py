# to follow a straight line 'contour'
# mainly for debugging, a dummy example
import os
import sys

CURRENT_PATH = os.getcwd()
sys.path.append(CURRENT_PATH)

from linear_mpcc.mpcc_solver import mpcc_solver
from linear_mpcc.contour import Contour
from linear_mpcc.kinematic_bicycle_model  import ROBOT_STATE
# from linear_mpcc.bicycle_model import ROBOT_STATE
from linear_mpcc.config import Param
from linear_mpcc.plot import visualization,mpc_visualization
import numpy as np
import matplotlib.pyplot as plt
from linear_mpcc.obstacle import Obstacle
# file_path = "log/horizon/group2/dataN12.npz"
# file_path = "log/parameter/group1/data0.002.npz"
file_path = "log/terminal cost/group1/data100.npz"
def set_params():
    # for bycle model
    param_dict = {"dt": 0.5,
                "N": 8,
                "Q": np.diag([4.0, 20.0, 1]),
                "P": 2*np.diag([2.0, 20.0]),
                "q": np.array([[0.02]]) ,
                "R": np.diag([0.1, 1]),
                "Rdu": np.diag([0.1, 1]),
                "Rv": np.diag([0.1]),
                "beta": 1*100,
                "disc_offset": 0.5,
                "radius": 2,
                "C1":0.5,
                "C2": 2.5,
                "max_vel":2.0,
                "max_acc": 2.0,
                "max_deltadot": 0.15,
                "max_delta": 0.3,
                "use_terminal_cost": True,
                "use_terminal_contraint":True,
                "use_prev_optim_ctrl":True

                }

    param = Param(param_dict)
    return param


def main():

    # quater circle path
    resolution = 0.1
    path = [[20*np.cos(1.57-1.57*i/314),20*np.sin(1.57-1.57*i/314)] for i in range(314)]+[[40-20*np.cos(1.57*i/314),20*np.sin(-1.57*i/314)] for i in range(314*2)]+[[80-20*np.cos(1.57*i/314),20*np.sin(1.57*i/314)] for i in range(314*2)]
    obstacles = None
    # robot_state = ROBOT_STATE(0.2, 20, 0.0, 0.5, 0)
    robot_state = ROBOT_STATE(0.1, 20.0-0.4, 0.2, 0.5, 0)
    goalx,goaly = path[-1][0],path[-1][1]
    contour = Contour(path)
    param = set_params()

    robot_x_real,robot_y_real,robot_yaw_real = [],[],[] # real robot state history - x,y,yaw
    robot_acc_real,robot_ddelta_real = [],[] # real robot control history - acc, ddelta
    ctrl = np.zeros((1,2)) # control array as in state space model
    prev_optim_ctrl = np.zeros((3,param.N)) # store the previous MPC's optimal control solution for next problem's state transitions
    prev_optim_theta = -contour.path_length * np.ones((1,param.N))   # at the start, assume it is the original value

    cost =[]
    error = []
    robot_states = []
    terminal_cost = []
    terminal_stage_cost = []
    terminal_cost_next = []
    STEP = 0
    t = 0
    figc, axsc = plt.subplots(1, 1)
    if obstacles is not None:
        fig_, axs_ = plt.subplots(len(obstacles), 1, figsize=(5, 12))

    while True:
        # mpcc opt
        if obstacles is not None:
            plt.sca(axs_)

            plt.pause(0.0001)

        ctrl,pred_states,theta,v,e,log = mpcc_solver(robot_state=robot_state, contour=contour,obstacles=None, param=param,
            prev_optim_ctrl=prev_optim_ctrl, prev_optim_theta=prev_optim_theta)
        prev_optim_ctrl = np.vstack([ctrl,v])
        prev_optim_theta = theta

        one_step_ctrl = (ctrl[:,0]).reshape(2,)
        print("one_step_ctrl: ",one_step_ctrl)

        # append ctrl history
        robot_acc_real.append(one_step_ctrl[0])
        robot_ddelta_real.append(one_step_ctrl[1])
        
        # append state history
        robot_states.append([robot_state.x,robot_state.y,robot_state.yaw,robot_state.v,robot_state.delta])

        plt.sca(axsc)
        visualization(robot_state, contour,axsc,obstacles)
        mpc_visualization(pred_states,contour,theta,axsc)
        # state update (transition with nonlinear simulation)
        plt.pause(0.0001)

        robot_state.state_update(one_step_ctrl[0], one_step_ctrl[1], v[0][0], param=param)
        if obstacles is not None:
            for obs in obstacles:
                obs.update(param.dt)

        # print(contour.get_location(theta))
        if param.use_terminal_cost:
            cost.append(log['cost'])
            terminal_cost.append(log['terminal_cost'])
            terminal_cost_next.append(log['terminal_costnext'])
            terminal_stage_cost.append(log['terminal_stage_cost'])
        error.append(log["error"])


        terminal_err = np.linalg.norm([robot_state.x-goalx,robot_state.y-goaly])

        if terminal_err <= 0.5:
            print("Goal reached.")
            break
        
        STEP += 1
        if STEP == 250:
            print("Haven't reach goal after 100 MPCC steps. Breaking out...")
            break
        t = t + param.dt
        if obstacles is not None:
            plt.sca(axs_)
            plt.cla()
        plt.sca(axsc)
        plt.cla()

    fig1, axs1 = plt.subplots(2, 1, figsize=(5, 12))
    fig2, axs2 = plt.subplots(2, 1, figsize=(5, 12))
    fig3, axs3 = plt.subplots(3, 1, figsize=(5, 12))

    axs1[0].plot(robot_acc_real)
    axs1[0].set_title("Control sequence optimized by MPCC")
    axs1[0].set_xlabel('timestep') # later change to time (divide by dt)
    axs1[0].set_ylabel('acc (m/s^2)')
    # steering change
    axs1[1].plot(robot_ddelta_real)
    axs1[1].set_xlabel('timestep') # later change to time (divide by dt)
    axs1[1].set_ylabel('d delta (rad/s)')
    if param.use_terminal_cost:
        axs2[0].set_title("terminal cost")
        axs2[0].plot(terminal_cost,label='terminal cost')
        axs2[0].plot(terminal_stage_cost,label = 'terminal_stage_cost')
        axs2[0].legend()
        axs2[1].set_title("cost")
        axs2[1].plot(cost)

    axs3[0].set_title("lateral error")
    axs3[0].plot([i[0] for i in error],label='x')
    axs3[1].set_title("longitudinal error")
    axs3[1].plot([i[1] for i in error],label='y')
    axs3[2].set_title("heading error")
    axs3[2].plot([i[2] for i in error],label='yaw')

    costs = np.vstack([np.array(cost),np.array(terminal_cost),np.array(terminal_stage_cost),np.array(terminal_cost_next)])
    inputs = np.vstack([np.array(robot_acc_real),np.array(robot_ddelta_real)])
# record path costs error robot_states inputs
    np.savez(file_path,path = np.array(path),costs=costs,inputs=inputs,robot_states=np.array(robot_states),error=np.array(error))

    plt.show()
    
if __name__ == '__main__':
    main()
