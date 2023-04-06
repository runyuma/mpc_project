<<<<<<< Updated upstream

=======
import numpy as np
>>>>>>> Stashed changes

class DynamicObstacles():
    def __init__(self, xo=0.0, yo=0.0, phi=0.0, alpha=1., beta=1., dx=0.3, dy=0.2, dphi=2.0):
        self.xo = xo
        self.yo = yo
        self.phi = phi
        self.alpha = alpha
        self.beta = beta
        self.dx = dx
        self.dy = dy
        self.dphi = dphi

    def state_update(self):
        self.xo += self.dx
        self.yo += self.dy
        self.phi += self.dphi
<<<<<<< Updated upstream
=======

    def get_tangential_line(self,xv,yv):
        dist = 0
        R_vehicle = np.sqrt((2-dist)**2+1.2**2)
        theta = np.arctan2((yv-self.yo),(xv-self.xo))
        x1,y1 = self.xo,self.yo
        x2,y2 = xv+R_vehicle*np.sin(theta), yv+R_vehicle*np.cos(theta)
        xm,ym = (x1+x2)/2,(y1+y2)/2
        res = ((x2-self.xo)*np.cos(self.phi) + (y2-self.yo)*np.sin(self.phi))**2/self.alpha**2 + \
                 ((self.xo-x2)*np.sin(self.phi) + (y2-self.yo)*np.cos(self.phi))**2/self.beta**2
        if res < 1:
            # print("collision")
            return None,None,None
        else:
            while True:
                # res = ((np.cos(self.phi))**2/self.alpha**2+(np.sin(self.phi))**2/self.beta**2)*(xm-self.xo)**2 + \
                #         2*np.cos(self.phi)*np.sin(self.phi)*(1/self.alpha**2-1/self.beta**2)*xm*ym + \
                #         ((np.sin(self.phi))**2/self.alpha**2+(np.cos(self.phi))**2/self.beta**2)*ym**2
                res = ((xm-self.xo)*np.cos(self.phi) + (ym-self.yo)*np.sin(self.phi))**2/self.alpha**2 + \
                        ((self.xo-xm)*np.sin(self.phi) + (ym-self.yo)*np.cos(self.phi))**2/self.beta**2
                if res - 1 > 0 and res-1<1e-12:
                    break
                elif res < 1:
                    x1,y1 = xm,ym
                else:
                    x2,y2 = xm,ym
                xm,ym = (x1+x2)/2,(y1+y2)/2

            k1 = (ym-self.yo)/(xm-self.xo)
            k_tan = -self.alpha**2/self.beta**2/k1
        
            return k_tan,xm,ym
>>>>>>> Stashed changes
