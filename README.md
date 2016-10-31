# Gradient-Descent

This repository contains 2 files that complete different tasks, which are outlined below:

*gradient_descent.m*:
The code in this MATLAB code is responsible for analyzing nonlinear error surface
*E(u, v) = (u\*e^v - 2\*v\*e^(-u))^2* with the a learning rate eta of 0.1.

First we calculate how many iterations it takes for the error to fall below 10^(-14)
and we find the values of *u* and *v* after that many iterations.

Next, we test coordinate descent and compare it to the gradient descent. We find that the value
of error *E(u, v)* after 15 iterations of the coordinate descent is close to 0.1398.


*logistic_regression.m*:
In this file, I created a target function and evaluated outputs (+1 or -1) for each point
based on where they were with respect to a randomly generated line. We used Logistic Regression
with Stochastic Gradient Descent to find our hypothesis function *g*, and estimated the cross
entropy error *E_out*. I stopped running the SGD algorithm once the magnitude of the difference
between two different weight vectors from consecutive epochs is less than 0.01.
I repeated the experiment 100 times to take the average values and found that the average cross
entropy error was 0.1014. This took an average of 337.99 epochs to reach.
