import numpy as np
import matplotlib.pyplot as plt
class Obstacle():
    def __init__(self,pos,vel = np.array([0,0]),radius = 2):
        self.pos = pos
        self.vel = vel
        self.radius = radius
    def update(self,dt):
        self.pos = self.pos + self.vel*dt
    def get_velobstacle(self,robot_state,horizon,param,visual = None):
        disc_1 = np.array([robot_state.x+param.disc_offset*np.cos(robot_state.yaw),robot_state.y+param.disc_offset*np.sin(robot_state.yaw)])
        disc_2 = np.array([robot_state.x-param.disc_offset*np.cos(robot_state.yaw),robot_state.y-param.disc_offset*np.sin(robot_state.yaw)])
        center1 = self.pos - disc_1
        center2 = self.pos - disc_2
        angle1 = np.arctan2(center1[1],center1[0])
        angle2 = np.arctan2(center2[1],center2[0])

        anglediff = np.arcsin(np.sin(angle1)*np.cos(angle2)-np.cos(angle1)*np.sin(angle2))
        if anglediff>0:
            larger = 0
        else:
            larger = 1

        dangle1 = np.arcsin((self.radius+param.radius)/np.linalg.norm(center1))
        dangle2 = np.arcsin((self.radius+param.radius)/np.linalg.norm(center2))
        print(dangle1, dangle2)
        if larger == 0:
            lineangle1 = max(angle1 + dangle1,angle2 + dangle2)
            lineangle2 = min(angle2 - dangle2,angle1 - dangle1)
        else:
            lineangle1 = min(angle1 - dangle1,angle2 - dangle2)
            lineangle2 = max(angle2 + dangle2,angle1 + dangle1)

        #ax+by+c>0

        if larger == 1:
            n1 = np.array([np.cos(lineangle1-np.pi/2),np.sin(lineangle1-np.pi/2)])
            n2 = np.array([np.cos(lineangle2+np.pi/2),np.sin(lineangle2+np.pi/2)])
            c1 = -np.dot(n1,self.vel)
            c2 = -np.dot(n2,self.vel)
            line1 = np.array([np.cos(lineangle1-np.pi/2),np.sin(lineangle1-np.pi/2),c1])
            line2 = np.array([np.cos(lineangle2+np.pi/2),np.sin(lineangle2+np.pi/2),c2])
        else:
            n1 = np.array([np.cos(lineangle1+np.pi/2),np.sin(lineangle1+np.pi/2)])
            n2 = np.array([np.cos(lineangle2-np.pi/2),np.sin(lineangle2-np.pi/2)])
            c1 = -np.dot(n1,self.vel)
            c2 = -np.dot(n2,self.vel)
            line1 = np.array([np.cos(lineangle1+np.pi/2),np.sin(lineangle1+np.pi/2),c1])
            line2 = np.array([np.cos(lineangle2-np.pi/2),np.sin(lineangle2-np.pi/2),c2])
        smallcenter1 = center1/horizon
        smallcenter2 = center2/horizon
        center1+=self.vel
        center2+=self.vel
        smallcenter1+=self.vel
        smallcenter2+=self.vel
        samllradius = (self.radius+param.radius)/horizon
        if abs(anglediff)<np.pi/8:
            if np.linalg.norm(center1)<np.linalg.norm(center2):
                line3 = np.array([-np.cos(angle1),-np.sin(angle1),-np.dot(-np.array([np.cos(angle1),np.sin(angle1)]),(smallcenter1+samllradius*np.array([-np.cos(angle1),-np.sin(angle1)])))])
            else:
                line3 = np.array([-np.cos(angle2),-np.sin(angle2),-np.dot(-np.array([np.cos(angle2),np.sin(angle2)]),(smallcenter2+samllradius*np.array([-np.cos(angle2),-np.sin(angle2)])))])

        else:
            _angle = np.arctan2(center1[1]-center2[1],center1[0]-center2[0])
            k = np.array([np.cos(_angle+np.pi/2),np.sin(_angle+np.pi/2)])
            if (-smallcenter1*k)[0]<0:
                k=-k
            line3 = np.array([k[0],k[1],-np.dot(k,((smallcenter1+smallcenter2)/2+samllradius*k))])
        v = np.array([np.cos(robot_state.yaw+0.5*robot_state.delta)*robot_state.v,np.sin(robot_state.yaw+0.5*robot_state.delta)*robot_state.v])
        score1 = v.dot(line1[0:2])+line1[2]
        score2 = v.dot(line2[0:2])+line2[2]
        score3 = v.dot(line3[0:2])+line3[2]
        score3 -= 5
        print(score1,score2,score3)
        chose = np.argmax([score1,score2,score3])
        # chose = np.argmax([score1, score2,])
        line = [line1,line2,line3][chose]
        color = ['r','r','r']
        color[chose] = 'y'
        if visual is not None:
            axes = visual
            draw_circle1 = plt.Circle(center1, (self.radius+param.radius),fill=False)
            draw_circle2 = plt.Circle(center2,(self.radius+param.radius),fill=False)
            draw_circle3 = plt.Circle(smallcenter1, (self.radius + param.radius)/horizon, fill=False)
            draw_circle4 = plt.Circle(smallcenter2, (self.radius + param.radius)/horizon, fill=False)
            axes.plot([-10,10,],[-(-10*line1[0]+line1[2])/(line1[1]),-(10*line1[0]+line1[2])/(line1[1])],color = color[0])
            axes.plot([-10,10,],[-(-10*line2[0]+line2[2])/(line2[1]),-(10*line2[0]+line2[2])/(line2[1])],color = color[1])
            axes.plot([-10,10,],[-(-10*line3[0]+line3[2])/(line3[1]),-(10*line3[0]+line3[2])/(line3[1])],color = color[2])
            axes.arrow(0, 0, v[0], v[1], color='g',head_width=0.5)
            axes.set_xlim(-20, 20)
            axes.set_ylim(-20, 20)

            axes.spines['right'].set_color('none')
            axes.spines['top'].set_color('none')
            axes.xaxis.set_ticks_position('bottom')
            axes.spines['bottom'].set_position(('data', 0))
            axes.yaxis.set_ticks_position('left')
            axes.spines['left'].set_position(('data', 0))

            axes.set_aspect(1)
            axes.add_artist(draw_circle1)
            axes.add_artist(draw_circle2)
            axes.add_artist(draw_circle3)
            axes.add_artist(draw_circle4)
            # plt.Circle(center2,self.radius)

        return line
if __name__ == '__main__':
    obstacle = Obstacle(np.array([0,7]),np.array([-1,0]),2)
    from linear_mpcc.kinematic_bicycle_model  import ROBOT_STATE
    from linear_mpcc.config import Param
    robot_state = ROBOT_STATE(0, 0, 0.1, 1, 0)
    def set_params():
        param_dict = {"dt": 0.5,
                      "N": 8,
                      "Q": np.diag([4.0, 20.0, 1]),
                      "P": 2 * np.diag([2.0, 20.0]),
                      "q": np.array([[0.02]]),
                      "R": np.diag([0.1, 1]),
                      "Rdu": np.diag([0.1, 1]),
                      "Rv": np.diag([0.1]),
                      "disc_offset": 1,
                      "radius": 1.5,
                      "C1": 0.5,
                      "C2": 2.5,
                      "max_vel": 2.0,
                      "max_acc": 2.0,
                      "max_deltadot": 0.15,
                      "max_delta": 0.3,
                      "use_terminal_cost": True,
                      "use_prev_optim_ctrl": True

                      }
        param = Param(param_dict)
        return param
    param = set_params()
    fig_, axs_ = plt.subplots(1, 1, figsize=(5, 12))
    line1,line2,line3 = obstacle.get_velobstacle(robot_state,4,param,visual = axs_)
    plt.show()
    print(line1,line2,line3)