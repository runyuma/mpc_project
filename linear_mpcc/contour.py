import os
import sys

CURRENT_PATH = os.getcwd()
sys.path.append(CURRENT_PATH)

import numpy as np
from linear_mpcc.bicycle_model import ROBOT_STATE


class Contour:
    def __init__(self, path, resolution=0.1):
        # path: list of points n*2
        self.resolution = resolution
        self.path = path
        self.path_length = len(path)*resolution
        self.theta = -self.path_length

    def find_closest_point(self, robot_state):
        min_index = (self.theta+self.path_length)/self.resolution
        path = np.array(self.path)
        point = np.array([robot_state.x,robot_state.y])
        dist = np.linalg.norm(path-point,axis=1)
        index = np.argmin(dist)
        index = max(index,min_index)
        theta = index*self.resolution-self.path_length
        self.theta = theta
        return theta

    def loc(self,theta):
        index = int((theta+self.path_length)/self.resolution)
        return self.path[index]

    def loc_index(self,theta):
        index = int((theta+self.path_length)/self.resolution)
        return index

    def regression(self,theta,horizon):
        # got local parametric contour
        start_index = self.loc_index(theta)
        end_index = self.loc_index(theta+horizon)
        leng = int(horizon/self.resolution)
        s = [theta+i*self.resolution for i in range(leng)]
        x = np.array(self.path)[start_index:start_index+leng,0]
        y = np.array(self.path)[start_index:start_index+leng,1]
        self.xparam = np.polyfit(s,x,3)
        self.yparam = np.polyfit(s,y,3)
 

if __name__ == '__main__':
    path = [[0,i*0.1] for i in range(100)]
    robot_state = ROBOT_STATE(0,2,0,0,0)
    contour = Contour(path)
    theta = contour.find_closest_point(robot_state)
    contour.regression(-5,3)
    print(theta, contour.loc(theta))
