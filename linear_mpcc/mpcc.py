# for model predictive contoure control, we need to calculate the linear and discrete time dynamic model.
import numpy as np
import cvxpy
import linear_mpcc.bicycle_model as model

def linear_mpc_control(robot_state,theta0,contour,param):
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
    T = param.N
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

    phi0,v0,delta0 = robot_state.yaw,robot_state.v,robot_state.delta
    A,B,C = model.calc_linear_discrete_model(phi0,v0,delta0,param)
    Ec,Jx,Jtheta = cal_error_linear(robot_state,theta0,contour,param)

    x0 = np.array([robot_state.x,robot_state.y,robot_state.yaw,robot_state.v,robot_state.delta])
    constraints += [x[:, 0] == x0]
    constraints += [theta[:, 0] == theta0]

    for k in range(T):
        #dynamic model contraints
        constraints += [e[:,k] == Ec + Jx@x[:,k] + Jtheta@theta[:,k]]
        constraints += [x[:, k + 1] == A@x[:, k]+B@u[:, k]+C]
        constraints += [theta[:, k + 1] == theta[:, k]+param.dt*v[:,k]]
        #input constraints
        #state constraints
        #cost function
        cost += cvxpy.quad_form(e[:,k],Q)
        cost += q.T@theta[:,k]
        cost += cvxpy.quad_form(u[:,k],R)
        cost += cvxpy.quad_form(v[:,k],Rv)
    #todo:terminal cost
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.OSQP)
    if prob.status == cvxpy.OPTIMAL or \
            prob.status == cvxpy.OPTIMAL_INACCURATE:
        x = x.value
        u = u.value
    else:
        print("Cannot solve linear mpc!")

    return x, u

def cal_error_linear(robot_state,contour,param):
    pass

