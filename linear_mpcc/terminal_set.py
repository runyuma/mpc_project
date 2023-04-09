import control
import numpy as np
import cvxpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

ecmax,vmax,deltamax,accmax,deltadotmax = 2,2,0.3,2,0.15
def CalInvSet(A,B,Q,R,k):
    K, S, E = control.dlqr(A, B, Q, R)
    Ak = A - B.dot(K)
    x_star = []
    # print("F",F )
    # print("A",(Ak ** (k + 1)))
    for i in range(F.shape[0]):
        x = cvxpy.Variable(6)
        cost = 0.
        constraints = []
        for j in range(k):
            Ak_ = np.eye(6)
            for t in range(j):
                Ak_ = Ak_.dot(Ak)
            constraints+=[F@Ak_@x<=f]
            # print("have a look at ",j,"\n",F @ Ak_ @ np.array([0, 0, 0, 0, 0, 0]) - f)
        cost = (F@(Ak**(k+1)))[i:i+1,:]@x
        prob = cvxpy.Problem(cvxpy.Maximize(cost), constraints)
        prob.solve()
        if prob.status == cvxpy.OPTIMAL or \
                prob.status == cvxpy.OPTIMAL_INACCURATE:
            x = x.value
            x_star.append(x)
        else:
            print((F@(Ak**(k+1)))[i:i+1,:])
            print("Error ",(k,i),": Cannot find optimal solution!")
    return x_star



A = np.eye(6)
k,dt,v0,delta0,C2,delta0dot = 0.0,0.5,1.0,0.0,2.5,1.0

A[0][1] = -k * dt
A[1][0] = k * dt
A[0][2] = v0 * dt
A[1][4] = -dt
A[2][4] = -delta0 / C2 * dt
A[2][5] = -v0 / C2 * dt
B = np.array([[0, 0, 0],
              [0, 0, dt],
              [0, 0, k/delta0dot * dt],
              [0, 0, dt],
              [dt, 0, 0],
              [0, dt, 0]
              ])
Q = np.diag([4,20,0.1,0.02,1,1])
R = np.diag([1,10,0.1])
K,S,E = control.dlqr(A,B,Q,R)

y = np.array([0,0.5,0.2,-100,v0,delta0])
u = -K.dot(y)
print(K)
print(u)
done = 0
Ak = A - B.dot(K)
F = np.zeros((8, 6))
F[0][0] = 1
F[1][0] = -1
F[2][1] = -1
F[3][3] = 1
F[4][4] = 1
F[5][4] = -1
F[6][5] = 1
F[7][5] = -1
f = np.array([ecmax, ecmax, 0, 0, vmax, 0, deltamax, deltamax])
# F = np.vstack([F, -K,K])
# f = np.array([ecmax,ecmax,0,0,vmax,0,deltamax,deltamax,accmax,deltadotmax,2,accmax,deltadotmax,0])
done = 0
for k in range(2,15):
    done = 1
    x_star = CalInvSet(A,B,Q,R,k)
    for ind,x in enumerate(x_star):
        # print("k=",k)
        # print(F@(Ak**(k+1))@x-f)
        if not ((F@(Ak**(k+1))@x-f)<0).all():
            print("Error ",(k,ind))
            done = 0
    if done:
        print("k=",k)
        break
print(x_star)
# plt ec,el
r = [x[:2] for x in x_star]
fig, ax1 = plt.subplots(3,1,figsize=(5,12))
ax1[0].scatter([x[0] for x in x_star], [x[1] for x in x_star], s=10)
ax1[0].set_xlabel('lateral error')
ax1[0].set_ylabel('lag error')
ax1[0].set_title('terminal region')
ax1[1].scatter([x[0] for x in x_star], [x[5] for x in x_star], s=10)
ax1[1].set_xlabel('lateral error')
ax1[1].set_ylabel('delta')
ax1[2].scatter([x[1] for x in x_star], [x[4] for x in x_star], s=10)
ax1[2].set_xlabel('lag error')
ax1[2].set_ylabel('v')
plt.show()



