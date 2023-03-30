import numpy as np
import matplotlib.pyplot as plt
from linear_mpcc.mpcc import linear_mpc_control_kb,linear_mpc_control_b

def mpcc_solver(robot_state, contour,obstacles, param, prev_optim_ctrl, prev_optim_theta):
    """
    solve the linear model predictive contour control problem
    :param robot_state: robot state
    :param theta: theta
    :param contour: contour
    :param param: parameters
    :return: control input
    """
    theta = contour.find_closest_point(robot_state)
    horizon = min(-theta, 15)
    contour.regression(theta, horizon)
    x,u,theta,v,e,log = linear_mpc_control_kb(robot_state,theta,contour,obstacles,param,prev_optim_ctrl, prev_optim_theta)
    # x, u, theta,v,e,log = linear_mpc_control_b(robot_state, theta, contour, param, prev_optim_ctrl, prev_optim_theta)
    return u,x,theta,v,e,log
