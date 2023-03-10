# for model predictive contoure control, we need to calculate the linear and discrete time dynamic model.
import numpy as np
import cvxpy
import linear_mpcc.bicycle_model as model
import copy

def linear_mpc_control(robot_state,theta0,contour,param,prev_optim_ctrl,prev_optim_v):
    """
    linear model predictive control
    :param robot_state: robot state
    :param contour: contour
    :param param: parameters
    :return: control input
    """
    # todo
    # 1. calculate the linear and discrete time dynamic model
    # 2. calculate the linear and discrete time cost function
    # 3. solve the linear model predictive control problem
    # 4. return the control input
    T = param.N     # horizon
    Q = param.Q
    P = param.P
    q = param.q
    R = param.R
    Rv = param.Rv

    e = cvxpy.Variable((2, T + 1))
    x = cvxpy.Variable((5, T + 1))
    u = cvxpy.Variable((2, T))
    v = cvxpy.Variable((1, T ))
    theta = cvxpy.Variable((1, T + 1))

    cost = 0.
    constraints = []

    approx_optim_ctrl = prev_optim_ctrl              # length is N-1
    approx_optim_v = prev_optim_v.reshape(-1)         # length is N-1

    robot_state_cp = copy.deepcopy(robot_state)

    phi0,v0,delta0 = robot_state.yaw, robot_state.v, robot_state.delta
    A,B,C = model.calc_linear_discrete_model(v0,phi0,delta0,param)
    Ec,Jx,Jtheta = cal_error_linear(robot_state,theta0,contour)

    x0 = np.array([robot_state.x,robot_state.y,robot_state.yaw,robot_state.v,robot_state.delta])
    constraints += [x[:, 0] == x0]
    constraints += [theta[:, 0] == theta0]
    
    for k in range(T):
        #dynamic model contraints
        constraints += [e[:,k] == Ec.reshape(2,) + (Jx@x[:,k]) + (Jtheta@theta[:,k])]
        constraints += [x[:, k + 1] == A@x[:, k]+B@u[:, k]+C]
        constraints += [theta[:, k + 1] == theta[:, k]+param.dt*v[:,k]]
        #input constraints
        constraints += [v[:, k]>=0]
        constraints += [u[1, k] <= param.delta_dot_max]
        constraints += [u[1, k] >= -param.delta_dot_max]
        constraints += [u[0, k] <= param.a_max]
        constraints += [u[0, k] >= -param.a_max]
        #state constraints
        constraints += [x[3, k] <= param.v_max]
        constraints += [x[3, k] >= 0]
        constraints += [x[4, k] <= param.delta_max]
        constraints += [x[4, k] >= -param.delta_max]
        constraints += [theta[:,k]<=0]
        #cost function
        cost += cvxpy.quad_form(x[2:3,k+1]-x[2:3,k],np.diag([10]))
        cost += cvxpy.quad_form(e[:,k],Q)
        cost += -q.T@theta[:,k]
        cost += cvxpy.quad_form(u[:,k],R)
        cost += cvxpy.quad_form(v[:,k],Rv)

        if param.use_prev_optim_ctrl and k<T-1: # use previous optimal control solution, need iteration every step
            robot_state_cp.state_update(approx_optim_ctrl[0,k], approx_optim_ctrl[1,k], param)    # update robot state
            theta0 = approx_optim_v[k]                                                         # update theta state
            Ec,Jx,Jtheta = cal_error_linear(robot_state_cp,theta0,contour)

    #terminal cost
    if param.use_terminal_cost:
        cost += cvxpy.quad_form(e[:,T],P)
        cost += -q.T@theta[:,T]

    constraints += [e[:, T] == Ec.reshape(2, ) + (Jx @ x[:, T]) + (Jtheta @ theta[:, T])]
    constraints += [x[3, T] <= param.v_max]
    constraints += [x[3, T] >= 0]
    constraints += [x[4, T] <= param.delta_max]
    constraints += [x[4, T] >= -param.delta_max]
    constraints += [theta[:, T] <= 0]
    #todo:terminal cost
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.OSQP)
    if prob.status == cvxpy.OPTIMAL or \
            prob.status == cvxpy.OPTIMAL_INACCURATE:
        x = x.value
        u = u.value
        v = v.value
        e = e.value
        theta = theta.value
    else:
        print("Cannot solve linear mpc!")

    return x, u, theta

def cal_error_linear(robot_state,theta,contour):
    # E = Ec + Jx*X + Jtheta*theta = E0 + Jx*(X-X0) + Jtheta*(theta-theta0)
    x = robot_state.x
    y = robot_state.y
    xd,yd  = contour.loc(theta)
    a1,b1,c1,d1 = contour.xparam
    a2,b2,c2,d2 = contour.yparam
    dx = 3*a1*theta**2 + 2*b1*theta + c1
    dy = 3*a2*theta**2 + 2*b2*theta + c2
    phi_theta = np.arctan2(dy,dx)
    # numerical derivative
    interv = 0.1
    dxp = 3*a1*(theta+interv)**2 + 2*b1*(theta+interv) + c1
    dyp = 3*a2*(theta+interv)**2 + 2*b2*(theta+interv) + c2
    dxm = 3*a1*(theta-interv)**2 + 2*b1*(theta-interv) + c1
    dym = 3*a2*(theta-interv)**2 + 2*b2*(theta-interv) + c2
    phi_theta_p = np.arctan2(dyp,dxp)
    phi_theta_m = np.arctan2(dym,dxm)
    dphi_theta= (phi_theta_p-phi_theta_m)/(2*interv)

    Jx = np.array([[np.sin(phi_theta), -np.cos(phi_theta), 0, 0, 0],
                   [-np.cos(phi_theta), -np.sin(phi_theta), 0, 0, 0]])
    Jtheta1 = np.cos(phi_theta)*dphi_theta*(x-xd)+np.sin(phi_theta)*dphi_theta*(y-yd)+np.sin(phi_theta)*(-dx)-np.cos(phi_theta)*(-dy)
    Jtheta2 = np.sin(phi_theta)*dphi_theta*(x-xd)-np.cos(phi_theta)*dphi_theta*(y-yd)-np.cos(phi_theta)*(-dx)-np.sin(phi_theta)*(-dy)
    Jtheta = np.array([[Jtheta1,Jtheta2]]).T
    Ec = np.array([[np.sin(phi_theta), - np.cos(phi_theta)],
                   [- np.cos(phi_theta), - np.sin(phi_theta)]])\
         @np.array([[x-xd],[y-yd]])
    X = np.array([[x],[y],[robot_state.yaw],[robot_state.v],[robot_state.delta]])

    Ec -= Jx@X
    Ec -= (Jtheta@np.array([theta])).reshape((2,1))

    return Ec,Jx,Jtheta

