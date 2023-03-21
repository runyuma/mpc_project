

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
