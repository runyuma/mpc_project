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
import numpy as np

def set_params():
    param_dict = {"dt": 0.1,
                "N": 10,
                "Q": np.diag([2.0, 1.0]),
                "P": np.diag([2.0, 1.0]),
                "q": np.array([1.0]),
                "Ru": np.diag([0.1, 0.1]),
                "Rv": np.diag([0.1]),
                "C2": 2.5
                }

    param = Param(param_dict)
    return param

def main():
    path = [[0,i*0.1] for i in range(100)] # x constantly 0, y from 0 to 10
    goalx = path[-1]
    goaly = path[-1]

    contour = Contour(path)

    robot_state = ROBOT_STATE(2,0,np.pi,0,0)  
    param = set_params()

    robot_state_real = [] # real robot state

    ctrl = np.zeros((1,2)) # control array as in state space model

    while True:
        # mpcc opt
        ctrl = mpcc_solver(robot_state, contour, param)

        # state update
        robot_state_real.append(robot_state)
        robot_state.state_update(acc=ctrl[0], deltadot=ctrl[1], param=param)

        terminal_err = np.linalg.norm([robot_state.x-goalx,robot_state.y-goaly])

        if terminal_err <= 0.1:
            print("Goal reached.")
            break




if __name__ == '__main__':
    main()
