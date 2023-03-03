import math
import numpy as np
def calc_linear_discrete_model(v, phi, delta,param):
    """
    calc linear and discrete time dynamic model.
    :param v: speed: v_bar
    :param phi: angle of vehicle: phi_bar
    :param delta: steering angle: delta_bar
    :return: A, B, C
    """

    A = np.array([[1.0, 0.0, param.dt * math.cos(phi), - param.dt * v * math.sin(phi),0.],
                  [0.0, 1.0, param.dt * math.sin(phi), param.dt * v * math.cos(phi),0.],
                  [0.0, 0.0, 1.0, param.dt * delta * param.C2, param.dt * delta * param.C2],
                  [0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0]])

    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [0.0, 0.0],
                  [param.dt, 0.0],
                  [0.0, param.dt]])

    C = np.array([param.dt * v * math.sin(phi) * phi,
                  - param.dt * v * math.cos(phi) * phi,
                  - param.dt * v * delta * param.C2,
                  0.0,
                  0.0])

    return A, B, C