# Gradient-Descent

This repository contains 2 files that complete different tasks, which are outlined below:

*gradient_descent.m*:
The code in this MATLAB code is responsible for analyzing nonlinear error surface
*E(u, v) = (u\*e^v - 2\*v\*e^(-u))^2* with the a learning rate eta of 0.1.

First we calculate how many iterations it takes for the error to fall below 10^(-14)
and we find the values of *u* and *v* after that many iterations.

Next, we test coordinate descent and compare it to the gradient descent. We find that the value
of error *E(u, v)* after 15 iterations of the coordinate descent is close to 0.1398.


*logistic_regression.m*
