import numpy as np
import matplotlib.pyplot as plt
from linear_mpcc.mpcc import linear_mpc_control

def mpcc_solver(robot_state, contour, param, prev_optim_ctrl, prev_optim_v, use_prev_optim_ctrl=True):
    """
    solve the linear model predictive contour control problem
    :param robot_state: robot state
    :param theta: theta
    :param contour: contour
    :param param: parameters
    :return: control input
    """
    theta = contour.find_closest_point(robot_state)
    horizon = min(-theta, 5)
    contour.regression(theta, horizon)
    x,u,theta = linear_mpc_control(robot_state,theta,contour,param,prev_optim_ctrl, prev_optim_v,use_prev_optim_ctrl)
    
    return u,x,theta
