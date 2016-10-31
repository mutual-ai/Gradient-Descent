% Tristan Née

i = 0; % Number of iterations
u_i = 1; % Starting u value
v_i = 1; % Starting v value
syms u v; % We are working in the uv space
E(u, v) = (u*exp(v) - 2*v*exp(-u))^2; % Nonlinear error surface E
eta = 0.1; % Value of the learning rate eta
E_grad(u, v) = gradient(E);

while E(u_i, v_i) > 10^(-14)
    i = i + 1;
    E_grad_val = E_grad(u_i, v_i);
    u_i = u_i - eta*double(E_grad_val(1)); % Update u_i
    v_i = v_i - eta*double(E_grad_val(2)); % Update v_i
    clear E_grad_val;
end
iterations = i % Our answer for question 5
final_u_v = [u_i, v_i] % Our answer for question 6

% Coordinate descent
iterations = 15; % Amount of iterations to run
u_i = 1; % Starting u value
v_i = 1; % Starting v value
E_u = diff(E, u); % Partial derivative of E w.r.t u
E_v = diff(E, v); % Partial derivative of E w.r.t v
for i = 1:iterations  
    u_i = double(u_i - eta*E_u(u_i, v_i));
    v_i = double(v_i - eta*E_v(u_i, v_i));
end
error =double(E(u_i, v_i)) % Our answer for question 7
    