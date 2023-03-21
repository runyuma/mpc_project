import matplotlib.pyplot as plt
import numpy as np
from linear_mpcc.bicycle_model import ROBOT_STATE
from linear_mpcc.contour import Contour
from matplotlib import patches

def visualization(robot_state, contour, obstacles=None):
    plt.cla()
    x_min = np.min(np.array(contour.path)[:,0])
    x_max = np.max(np.array(contour.path)[:,0])
    y_min = np.min(np.array(contour.path)[:,1])
    y_max = np.max(np.array(contour.path)[:,1])
    bound = 5
    plt.xlim(x_min-bound,x_max+bound)
    plt.ylim(y_min-bound,y_max+bound)
    ax = plt.gca()
    ax.set_aspect(1)
    
    x = np.array(contour.path)[:,0]
    y = np.array(contour.path)[:,1]
    theta = contour.find_closest_point(robot_state)
    horizon = min(-theta, 5)
    # contour.regression(theta, horizon)
    a1, b1, c1, d1 = contour.xparam
    a2, b2, c2, d2 = contour.yparam
    thetas = np.linspace(theta, theta + horizon, 100)
    xs = a1 * thetas ** 3 + b1 * thetas ** 2 + c1 * thetas + d1
    ys = a2 * thetas ** 3 + b2 * thetas ** 2 + c2 * thetas + d2
    ref = contour.loc(theta)
    plt.plot(x,y,c='b',)
    plt.plot(xs,ys,c='yellow',linewidth=2)
    plt.scatter(ref[0],ref[1],c='r')
    draw_car(robot_state.x,robot_state.y,robot_state.yaw,robot_state.delta)
    if obstacles!=None:
        for obstacle in obstacles:
            draw_obstacle(ax, obstacle.xo, obstacle.yo, obstacle.alpha, obstacle.beta, obstacle.phi)

def draw_car(x,y,yaw,steer,color='black'):
    # draw car
    RF = 2  # [m] distance from center to vehicle front end of vehicle
    RB = 2  # [m] distance from center to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.5  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width

    car = np.array([[-RB, -RB, RF, RF, -RB],
                    [W / 2, -W / 2, -W / 2, W / 2, W / 2]])

    wheel = np.array([[-TR, -TR, TR, TR, -TR],
                      [TW / 4, -TW / 4, -TW / 4, TW / 4, TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
                     [np.sin(yaw), np.cos(yaw)]])

    Rot2 = np.array([[np.cos(steer), -np.sin(steer)],
                     [np.sin(steer), np.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[WB/2], [-WD / 2]])
    flWheel += np.array([[WB/2], [WD / 2]])
    rrWheel += np.array([[-WB/2], [-WD / 2]])
    rlWheel += np.array([[-WB/2], [WD / 2]])

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)

def draw_obstacle(ax,x,y,alpha,beta,phi):
    # phi = np.deg2rad(phi)
    obs = patches.Ellipse((x,y), alpha, beta,angle=phi)
    ax.add_patch(obs)

def mpc_visualization(states):
    # print("predstates",states[:,0:2])
    x = np.array(states)[0]
    y = np.array(states)[1]
    plt.plot(x,y,'go')
if __name__ == '__main__':
    path = [[i*0.1,20] for i in range(500)]
    # path = [[20*np.cos(1.57-1.57*i/314),20*np.sin(1.57-1.57*i/314)] for i in range(314)]
    robot_state = ROBOT_STATE(0.2,20,0,0,0)
    contour = Contour(path)
    visualization(robot_state, contour)
