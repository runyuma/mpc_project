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
        # vehicle parameters
        self.C2 = param_dict["C2"]    # inter-axle distance (NOT reciprocal)

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