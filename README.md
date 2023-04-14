# Model Predictive Contour Control Repository
This repository contains code and examples for implementing a Model Predictive Contour Control (MPCC) approach for planning and control of autonomous vehicles. The goal of this approach is to track a given reference trajectory while avoiding obstacles in a dynamic environment.

*Authors:*
- Runyu, Ma (r.ma-8@student.tudelft.nl)
- Yongxi, Cao (y.cao-18@student.tudelft.nl)

## Abstract
We focus on a model predictive contouring control (MPCC) approach for planning and control of autonomous vehicles. The dynamics of vehicle and contouring systems are linearized to a linear time varying (LTV) system around the reference trajectory and current state of the robot. In this paper, a model predictive control problem is formulated to minimize a cost function which reflects the trade-off between tracking error and tracking velocity for the contouring system, subject to vehicle's state and actuator constraints. To guarantee its feasibility, we analytically show the system can yield asymptotic stability by specially designing the terminal cost and terminal constraints. To achieve dynamic obstacle avoidance, a velocity obstacle algorithm that has been specially designed is deployed. Simulations are designed to evaluate tracking and obstacle avoidance performance involving hyperparameter setting and terminal cost.

## Installation
To use this repository, first clone the repository to your local machine. You will need to have Python 3.x installed, as well as the following packages:

- numpy
- matplotlib
- control
- copy
- cvxpy
- os
- sys

You can install these packages using pip:

```
pip install numpy matplotlib control copy cvxpy
```

## Examples
This repository includes several examples in the ./examples directory. To run these examples, navigate to the root directory of the repository and execute the examples from there. For example:

```
cd [/path/to/mpcc_project]
python ./examples/[example_name].py
```
