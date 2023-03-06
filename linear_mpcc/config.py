import numpy as np


class Param:
    def __init__(self,param_dict):
        # mpcc parameters
        self.dt = param_dict["dt"]    # discretize time interval (sec)
        self.N = param_dict["N"]      # horizon of MPC
        self.Q = param_dict["Q"]      # as defined in cost function
        self.P = param_dict["P"]      # terminal cost
        self.q = param_dict["q"]      # as defined in cost function
        self.R = param_dict["Ru"]     # as defined in cost function (Ru)
        self.Rv = param_dict["Rv"]    # as defined in cost function (Rv)
        self.delta_max = param_dict["max_delta"]    # maximum steering angle (rad)
        self.delta_dot_max = param_dict["max_deltadot"]    # maximum steering angle change (rad/s)
        self.v_max = param_dict["max_vel"]    # maximum velocity (m/s)
        self.a_max = param_dict["max_acc"]    # maximum acceleration (m/s^2)

        # vehicle parameters
        self.C2 = param_dict["C2"]    # inter-axle distance (NOT reciprocal)
        