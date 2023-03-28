# for model predictive contoure control, we need to calculate the linear and discrete time dynamic model.
import numpy as np
import cvxpy
from linear_mpcc import kinematic_bicycle_model, bicycle_model
import control
import copy
import matplotlib.pyplot as plt
def linear_mpc_control_b(robot_state,theta0,contour,param,prev_optim_ctrl,prev_optim_theta):
    """
    linear kinematic bicycle model predictive control
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

    e = cvxpy.Variable((3, T + 1))
    x = cvxpy.Variable((3, T + 1))
    u = cvxpy.Variable((2, T))
    v = cvxpy.Variable((1, T ))
    theta = cvxpy.Variable((1, T + 1))

    cost = 0.
    constraints = []

    approx_optim_ctrl = prev_optim_ctrl              # length is N-1
    prev_optim_theta = prev_optim_theta.reshape(-1)         # length is N-1

    robot_state_cp = copy.deepcopy(robot_state)

    phi0,v0,delta0,thetadot0 = robot_state.yaw, robot_state.v, robot_state.delta,robot_state.thetadot
    A,B,C = bicycle_model.calc_linear_discrete_model(v0,phi0,delta0,param)
    Ec,Jx,Jtheta = cal_error_linear(robot_state,theta0,contour)

    x0 = np.array([robot_state.x,robot_state.y,robot_state.yaw,])
    constraints += [x[:, 0] == x0]
    constraints += [theta[:, 0] == theta0]

    for k in range(T):
        #dynamic model contraints
        constraints += [e[:,k] == Ec.reshape(3,) + (Jx@x[:3,k]) + (Jtheta@theta[:,k])]
        constraints += [x[:, k + 1] == A@x[:, k]+B@u[:, k]+C]
        constraints += [theta[:, k + 1] == theta[:, k]+param.dt*v[:,k]]

        #state constraints
        if k>0:
            constraints += [e[1,k]>=0]

        #input constraints
        constraints += [v[:, k]>=0]
        constraints += [u[0, k] <= param.v_max]
        constraints += [u[0, k] >= 0]
        constraints += [u[1, k]<= param.delta_max]
        constraints += [u[1, k] >= -param.delta_max]
        if k>0:
            constraints += [u[1, k]-u[1, k-1] <= param.delta_dot_max*param.dt]
            constraints += [u[1, k]-u[1, k-1] >= -param.delta_dot_max*param.dt]
            constraints += [u[0, k]- u[0, k-1]<= param.a_max*param.dt]
            constraints += [u[0, k]-u[0, k-1]>= -param.a_max*param.dt]
        else:
            constraints += [u[1, k] - delta0 <= param.delta_dot_max*param.dt]
            constraints += [u[1, k] - delta0>= -param.delta_dot_max*param.dt]
            constraints += [u[0, k] - v0<= param.a_max*param.dt]
            constraints += [u[0, k] -v0>= -param.a_max*param.dt]

        constraints += [theta[:,k]<=0]
        #cost function
        cost += cvxpy.quad_form(e[:,k],Q)
        cost += cvxpy.quad_form(theta[:,k],q)
        cost += cvxpy.quad_form(u[:,k]-np.array([0,delta0]),R)
        cost += cvxpy.quad_form(v[:,k],Rv)

        if param.use_prev_optim_ctrl and k<T-1: # use previous optimal control solution, need iteration every step
            robot_state_cp.state_update(approx_optim_ctrl[0,k], approx_optim_ctrl[1,k],approx_optim_ctrl[2,k], param)    # update robot state
            delta0 = approx_optim_ctrl[1,k]                                                         # update delta state
            theta0 = prev_optim_theta[k]                                                         # update theta state
            Ec,Jx,Jtheta = cal_error_linear(robot_state_cp,theta0,contour)


    constraints += [e[:, T] == Ec.reshape(3, ) + (Jx @ x[:3, T]) + (Jtheta @ theta[:, T])]
    constraints += [theta[:, T] <= 0]
    # terminal cost
    if param.use_terminal_cost:
        if param.use_prev_optim_ctrl:
            thetadot0 = approx_optim_ctrl[2,-1]
        LQR_A,LQR_B,LQR_res = calculate_LQR_b(robot_state, theta0, contour, param, thetadot=thetadot0)
        terminal_cost = calculate_terminal_cost_b(LQR_A,LQR_B,param)
        P = terminal_cost[0]
        cost += cvxpy.quad_form(cvxpy.hstack([e[:,T],theta[:, T]]), P)




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
    print(robot_state.delta,u[1][0])
    # print("x,u,e",x,u,e)
    print(e[:,0].shape)
    log = {'cost': cost.value,'error':e[:,0] }

    if param.use_terminal_cost:
        x_ter = np.vstack([e[:,-1:],theta[:,-1:]])
        log['terminal_cost'] = (x_ter.T@P@x_ter)[0][0]
        K = LQR_res[0]
        u_lqr = -K@x_ter
        x_ter_plus = LQR_A@x_ter+LQR_B@u_lqr
        print(x_ter_plus)
        Q = np.diag([param.Q[0][0], param.Q[1][1], param.Q[2][2], param.q[0][0]])
        R = np.diag([param.R[0][0], param.R[1][1], param.Rv[0][0]])
        terminal_costnext = (x_ter_plus.T@P@x_ter_plus)[0][0] + (x_ter.T@Q@x_ter+u_lqr.T@R@u_lqr )[0][0]
        log['terminal_costnext'] = terminal_costnext
        log['terminal_stage_cost'] = (x_ter.T@Q@x_ter+u_lqr.T@R@u_lqr )[0][0]

        # print((x_ter.T@P@x_ter).shape)
    return x, u, theta,v,e,log


def linear_mpc_control_kb(robot_state, theta0, contour, param, prev_optim_ctrl, prev_optim_v):
    """
    linear bicycle model predictive control
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
    T = param.N  # horizon
    Q = param.Q
    q = param.q
    R = param.R
    Rdu = param.Rdu
    Rv = param.Rv

    e = cvxpy.Variable((3, T + 1))
    x = cvxpy.Variable((5, T + 1))
    u = cvxpy.Variable((2, T))
    v = cvxpy.Variable((1, T))
    theta = cvxpy.Variable((1, T + 1))

    cost = 0.
    constraints = []

    approx_optim_ctrl = prev_optim_ctrl  # length is N-1
    approx_optim_v = prev_optim_v.reshape(-1)  # length is N-1

    robot_state_cp = copy.deepcopy(robot_state)

    phi0, v0, delta0 = robot_state.yaw, robot_state.v, robot_state.delta
    A, B, C = kinematic_bicycle_model.calc_linear_discrete_model(v0, phi0, delta0, param)
    Ec, Jx, Jtheta = cal_error_linear(robot_state, theta0, contour)

    x0 = np.array([robot_state.x, robot_state.y, robot_state.yaw, robot_state.v, robot_state.delta])
    constraints += [x[:, 0] == x0]
    constraints += [theta[:, 0] == theta0]

    for k in range(T):
        # dynamic model contraints
        constraints += [e[:, k] == Ec.reshape(3, ) + (Jx @ x[:3, k]) + (Jtheta @ theta[:, k])]
        constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k] + C]
        constraints += [theta[:, k + 1] == theta[:, k] + param.dt * v[:, k]]
        if k>0:
            constraints += [e[1,k]>=0]

        # input constraints
        constraints += [v[:, k] >= 0]
        constraints += [u[1, k] <= param.delta_dot_max]
        constraints += [u[1, k] >= -param.delta_dot_max]
        constraints += [u[0, k] <= param.a_max]
        constraints += [u[0, k] >= -param.a_max]
        # state constraints
        constraints += [x[3, k] <= param.v_max]
        constraints += [x[3, k] >= 0]
        constraints += [x[4, k] <= param.delta_max]
        constraints += [x[4, k] >= -param.delta_max]
        constraints += [theta[:, k] <= 0]
        # cost function
        cost += cvxpy.quad_form(e[:, k], Q)
        cost += cvxpy.quad_form(theta[:, k], q)
        cost += cvxpy.quad_form(x[3:, k]-np.array([0,delta0]), R)
        cost += cvxpy.quad_form(u[:, k], Rdu)
        cost += cvxpy.quad_form(v[:, k], Rv)

        if param.use_prev_optim_ctrl and k < T - 1:  # use previous optimal control solution, need iteration every step
            robot_state_cp.state_update(approx_optim_ctrl[0, k], approx_optim_ctrl[1, k], approx_optim_ctrl[2,k],param)  # update robot state
            theta0 = approx_optim_v[k]  # update theta state
            Ec, Jx, Jtheta = cal_error_linear(robot_state_cp, theta0, contour)


    # terminal cost
    if param.use_terminal_cost:
        if param.use_prev_optim_ctrl:
            thetadot0 = approx_optim_ctrl[2, -1]
        LQR_A, LQR_B, LQR_res = calculate_LQR_kb(robot_state, theta0, contour, param, thetadot=thetadot0)
        terminal_cost = calculate_terminal_cost_kb(LQR_A, LQR_B, param)
        P = terminal_cost[0]
        cost += cvxpy.quad_form(cvxpy.hstack([e[:, T], theta[:, T], x[3:, T]]), P)

    constraints += [e[:, T] == Ec.reshape(3, ) + (Jx @ x[:3, T]) + (Jtheta @ theta[:, T])]
    constraints += [x[3, T] <= param.v_max]
    constraints += [x[3, T] >= 0]
    constraints += [x[4, T] <= param.delta_max]
    constraints += [x[4, T] >= -param.delta_max]
    constraints += [theta[:, T] <= 0]
    # todo:terminal cost
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
    log = {'cost': cost.value, 'error': e[:, 0]}
    if param.use_terminal_cost:
        x_ter = np.hstack([e[:, T], theta[:, T], x[3:, T]])
        log['terminal_cost'] = (x_ter.T@P@x_ter)
        K = LQR_res[0]
        u_lqr = -K@x_ter
        x_ter_plus = LQR_A@x_ter+LQR_B@u_lqr
        print(x_ter_plus)
        Q = np.diag([param.Q[0][0], param.Q[1][1], param.Q[2][2], param.q[0][0], param.R[0][0], param.R[1][1]])
        R = np.diag([param.Rdu[0][0], param.Rdu[1][1], param.Rv[0][0]])
        terminal_costnext = (x_ter_plus.T@P@x_ter_plus) + (x_ter.T@Q@x_ter+u_lqr.T@R@u_lqr )
        log['terminal_costnext'] = terminal_costnext
        log['terminal_stage_cost'] = (x_ter.T@Q@x_ter+u_lqr.T@R@u_lqr )

    return x, u, theta,v,e,log
def cal_error_linear(robot_state,theta,contour):
    # E = Ec + Jx*X + Jtheta*theta = E0 + Jx*(X-X0) + Jtheta*(theta-theta0)
    x = robot_state.x
    y = robot_state.y
    phi = robot_state.yaw
    xd,yd  = contour.loc(theta)
    a1,b1,c1,d1 = contour.xparam
    a2,b2,c2,d2 = contour.yparam
    dx = 3*a1*theta**2 + 2*b1*theta + c1
    dy = 3*a2*theta**2 + 2*b2*theta + c2
    phi_theta = np.arctan2(dy,dx)
    print(phi,phi_theta)
    # numerical derivative
    interv = 0.1
    dxp = 3*a1*(theta+interv)**2 + 2*b1*(theta+interv) + c1
    dyp = 3*a2*(theta+interv)**2 + 2*b2*(theta+interv) + c2
    dxm = 3*a1*(theta-interv)**2 + 2*b1*(theta-interv) + c1
    dym = 3*a2*(theta-interv)**2 + 2*b2*(theta-interv) + c2
    phi_theta_p = np.arctan2(dyp,dxp)
    phi_theta_m = np.arctan2(dym,dxm)
    dphi_theta= (phi_theta_p-phi_theta_m)/(2*interv)

    Jx = np.array([[np.sin(phi_theta), -np.cos(phi_theta), 0, ],
                   [-np.cos(phi_theta), -np.sin(phi_theta), 0, ],
                   [0, 0, -1, ]])
    Jtheta1 = np.cos(phi_theta)*dphi_theta*(x-xd)+np.sin(phi_theta)*dphi_theta*(y-yd)+np.sin(phi_theta)*(-dx)-np.cos(phi_theta)*(-dy)
    Jtheta2 = np.sin(phi_theta)*dphi_theta*(x-xd)-np.cos(phi_theta)*dphi_theta*(y-yd)-np.cos(phi_theta)*(-dx)-np.sin(phi_theta)*(-dy)
    Jtheta = np.array([[Jtheta1,Jtheta2,dphi_theta]]).T
    Ec = np.array([[np.sin(phi_theta), - np.cos(phi_theta),0],
                   [- np.cos(phi_theta), - np.sin(phi_theta),0],
                   [0,0,-1]])\
         @np.array([[x-xd],[y-yd],[phi-phi_theta]])
    X = np.array([[x],[y],[robot_state.yaw],])

    Ec -= Jx@X
    Ec -= (Jtheta@np.array([theta])).reshape((3,1))

    return Ec,Jx,Jtheta
def calculate_LQR_kb(robot_state,theta0,contour,param,thetadot):
    # A, B is the transition matrix of error dynamics
    # state = [error_lateral,error_longitudinal,error_heading,theta,velocity,theta]
    # Input = [steering_angle,velocity, theta_dot]
    A = np.eye(6)

    # calculate dphi/dtheta
    x = robot_state.x
    y = robot_state.y
    xd, yd = contour.loc(theta0)
    a1, b1, c1, d1 = contour.xparam
    a2, b2, c2, d2 = contour.yparam
    dx = 3 * a1 * theta0 ** 2 + 2 * b1 * theta0 + c1
    dy = 3 * a2 * theta0 ** 2 + 2 * b2 * theta0 + c2
    phi_theta = np.arctan2(dy, dx)
    # numerical derivative
    interv = 0.1
    dxp = 3 * a1 * (theta0 + interv) ** 2 + 2 * b1 * (theta0 + interv) + c1
    dyp = 3 * a2 * (theta0 + interv) ** 2 + 2 * b2 * (theta0 + interv) + c2
    dxm = 3 * a1 * (theta0 - interv) ** 2 + 2 * b1 * (theta0 - interv) + c1
    dym = 3 * a2 * (theta0 - interv) ** 2 + 2 * b2 * (theta0 - interv) + c2
    phi_theta_p = np.arctan2(dyp, dxp)
    phi_theta_m = np.arctan2(dym, dxm)
    dphi_theta = (phi_theta_p - phi_theta_m) / (2 * interv)

    k = thetadot*dphi_theta
    phi0, v0, delta0 = robot_state.yaw, robot_state.v, robot_state.delta
    A[0][1] = -k*param.dt
    A[1][0] = k*param.dt
    A[0][2] = v0*param.dt
    A[1][4] = -param.dt
    # print(-delta0/param.C2*param.dt)
    A[2][4] = -delta0/param.C2*param.dt
    A[2][5] = -v0/param.C2*param.dt


    B = np.array([[0,0,0],
                  [0,0,param.dt],
                  [0,0,dphi_theta*param.dt],
                  [0,0,param.dt],
                  [param.dt,0,0],
                  [0,param.dt,0]
    ])
    Q = np.diag([param.Q[0][0],param.Q[1][1],param.Q[2][2],param.q[0][0],param.R[0][0],param.R[1][1]])
    R = np.diag([param.Rdu[0][0],param.Rdu[1][1],param.Rv[0][0]])
    # print("A",A)
    # print("B",B)

    C = control.ctrb(A,B)
    print(np.linalg.matrix_rank(C))
    # lqr
    K = control.dlqr(A,B,Q,R)
    # print("K",K[0])

    return A,B,K
def calculate_terminal_cost_kb(A,B,param):
    Q = np.diag([param.Q[0][0], param.Q[1][1], param.Q[2][2], param.q[0][0], param.R[0][0], param.R[1][1]])
    R = np.diag([param.Rdu[0][0], param.Rdu[1][1], param.Rv[0][0]])
    P = control.dare(A,B,Q,R)
    # print("p",P[0])
    return P



def calculate_LQR_b(robot_state,theta0,contour,param,thetadot):
    # A, B is the transition matrix of error dynamics
    A = np.eye(4)

    # calculate dphi/dtheta
    x = robot_state.x
    y = robot_state.y
    xd, yd = contour.loc(theta0)
    a1, b1, c1, d1 = contour.xparam
    a2, b2, c2, d2 = contour.yparam
    dx = 3 * a1 * theta0 ** 2 + 2 * b1 * theta0 + c1
    dy = 3 * a2 * theta0 ** 2 + 2 * b2 * theta0 + c2
    phi_theta = np.arctan2(dy, dx)
    # numerical derivative
    interv = 0.1
    dxp = 3 * a1 * (theta0 + interv) ** 2 + 2 * b1 * (theta0 + interv) + c1
    dyp = 3 * a2 * (theta0 + interv) ** 2 + 2 * b2 * (theta0 + interv) + c2
    dxm = 3 * a1 * (theta0 - interv) ** 2 + 2 * b1 * (theta0 - interv) + c1
    dym = 3 * a2 * (theta0 - interv) ** 2 + 2 * b2 * (theta0 - interv) + c2
    phi_theta_p = np.arctan2(dyp, dxp)
    phi_theta_m = np.arctan2(dym, dxm)
    dphi_theta = (phi_theta_p - phi_theta_m) / (2 * interv)

    k = thetadot*dphi_theta
    phi0, v0, delta0 = robot_state.yaw, robot_state.v, robot_state.delta
    A[0][1] = -k*param.dt
    A[1][0] = k*param.dt
    A[0][2] = v0*param.dt

    B = np.array([[0,0,0],
                  [-param.dt,0,param.dt],
                  [-delta0/param.C2*param.dt,-v0/param.C2*param.dt,dphi_theta*param.dt],
                  [0,0,param.dt]
    ])
    Q = np.diag([param.Q[0][0],param.Q[1][1],param.Q[2][2],param.q[0][0]])
    R = np.diag([param.R[0][0],param.R[1][1],param.Rv[0][0]])
    # print("A",A)
    # print("B",B)

    C = control.ctrb(A,B)
    print(np.linalg.matrix_rank(C))
    # lqr
    K = control.dlqr(A,B,Q,R)
    print("K",K[0])

    return A,B,K


def calculate_terminal_cost_b(A,B,param):
    Q = np.diag([param.Q[0][0],param.Q[1][1],param.Q[2][2],param.q[0][0]])
    R = np.diag([param.R[0][0],param.R[1][1],param.Rv[0][0]])
    P = control.dare(A,B,Q,R)
    print("p",P[0])
    return P



