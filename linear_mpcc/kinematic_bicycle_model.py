import math
import numpy as np


def calc_linear_discrete_model(v, phi, delta, param):
    """
    Calculate linearized and discretized state space model for the (approximated) vehicle dynamics.
        
    -state definition:
        x_k = [   X_k   ] : x position under Cartesian frame
              |   Y_k   | : y position under Cartesian frame
              |  phi_k  | : pose orientation under Cartesian frame (yaw)
              |   v_k   | : vehicle velocity in the longitudinal direction (along the vehicle)
              [ delta_k ] : steerging angle

    -arguments:
        v     : speed: v_bar
        phi   : pose angle of vehicle: phi_bar
        delta : steering angle: delta_bar
    
    -returns:
        A, B, C : linearized transition matrices in state space model
    """
    A = np.array([[1.0, 0.0, - param.dt * v * math.sin(phi), param.dt * math.cos(phi), 0.],
                  [0.0, 1.0, param.dt * v * math.cos(phi), param.dt * math.sin(phi), 0.],
                  [0.0, 0.0, 1.0, param.dt * delta / param.C2, param.dt * v / param.C2],
                  [0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0]])

    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [0.0, 0.0],
                  [param.dt, 0.0],
                  [0.0, param.dt]])

    C = np.array([param.dt * v * math.sin(phi) * phi,
                  - param.dt * v * math.cos(phi) * phi,
                  - param.dt * v * delta / param.C2,
                  0.0,
                  0.0])

    return A, B, C

class ROBOT_STATE():
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0,delta = 0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.delta = delta
        #todo:delta

    def state_update(self, acc, deltadot,thetadot, param):
        """
        update the state of the robot (discretized, but nonlinear transition)

        -arguments:
            acc      : acceleration
            deltadot : steering angle
            param    : model parameters
        """
        dt = param.dt
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / param.C2 * np.tan(self.delta) * dt
        self.v += acc * dt
        self.delta += deltadot * dt