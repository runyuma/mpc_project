
import numpy as np
class Param:
    def __init__(self,param_dict):
        # mpcc parameters
        self.dt = param_dict["dt"]
        self.N = param_dict["N"]
        self.Q = param_dict["Q"]
        self.P = param_dict["P"]
        self.q = param_dict["q"]
        self.R = param_dict["R"]
        self.Rv = param_dict["Rv"]
        # vehicle parameters
        self.C2 = param_dict["C2"]#inter-axle distance
param_dict = {
    "dt": 0.1,
    "N": 10,
    "Q": np.diag([2.0, 1.0]),
    "P": np.diag([2.0, 1.0]),
    "q": np.array([1.0]),
    "R": np.diag([0.1, 0.1]),
    "Rv": np.diag([0.1]),
    "C2": 2.5}
param = Param(param_dict)