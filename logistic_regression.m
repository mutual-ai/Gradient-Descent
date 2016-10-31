% Tristan Née

N = 100; % Amount of training points
N2 = 1000; % New, out of sample points to calculate E_out
E_out = 0; % Cross entropy error
simulations = 100; % Amount of simulations to run
iterations_sum = 0; % Keep track of total amount of iterations
eta = 0.01; % Learning rate
E_out_total = 0;
epoch_total = 0;

for s = 1:(simulations)
    % Create two random points to create target function f.
    x1 = -1+2*rand(1,1);
    y1 = -1+2*rand(1,1);
    x2 = -1+2*rand(1,1);
    y2 = -1+2*rand(1,1);
    m = (y2 - y1)/(x2-x1); % Sslope of line
    b = -m*x1 + y1; % y-intercept of line
    syms x;
    f = m*x+b; % Target function f

    % Construct matrix X.
    % First column will contain x coordinate of points
    % Second column will contain y coordinate of points
    % Third column will be the output y_n for the corresponding point
    % Fourth column will consist of all 1's for the x_0 term
    X = ones(N, 4);
    count = 0;
    sideF = zeros(N, 1); % Points above line will be +1, below are -1
    for j = 1:(N)
        count = count + 1;
        x_cord = -1+2*rand(1,1); % x coordinate of point
        y_cord = -1+2*rand(1,1);% y coordinate of point
        X(count, 1) = x_cord;
        X(count, 2) = y_cord;
        if (y_cord > m*x_cord + b)
            X(count, 3) = 1; % y_n output of +1
        else
            X(count, 3) = -1; % y_n output of -1
        end
    end
    
    w2 = zeros(1, 3); % Initialize weight vector to be all zeros
    w1 = ones(1, 3); % Weight vector for previous epoch
    epoch = 0; % t, or the amount of epochs
    
    while (norm(w1 - w2) > 0.01) % Condition to stop SGD algorithm
        w1 = w2; % Adjust previous weight vector
        epoch = epoch + 1;
        c = 0; % Counting index
        
        % Sums for Stochastic Gradient Descent algorithm
        sumX = w2(1, 1); % For x coordinate
        sumY = w2(1, 2); % For y coordinate
        sum_o = w2(1, 3); % For x_o coordinate
        
        for l = 1:(N)
            c = c + 1;
            val = exp(-X(c, 3)*dot([X(c, 1), X(c, 2), X(c, 4)], w2));
            denominator = 1 + val;
            sumX =  X(c, 3)*X(c, 1)/denominator; % Update x1
            sumY =  X(c, 3)*X(c, 2)/denominator; % Update x2
            sum_o =  X(c, 3)*X(c, 4)/denominator; % Update x_o
            w2 =  w2 - eta*[sumX, sumY, sum_o]; % Update weight vector
        end
        E_in = (-1/N)*w2; % The gradient
    end
    
    % Now generate out of sample points
    count = 0;
    misclassified_out = 0;
    
    E_out = 0;
    for j = 1:(N2)
        count = count + 1;
        x_cord2 = -1+2*rand(1,1); % x coordinate of point
        y_cord2 = -1+2*rand(1,1);% y coordinate of point
        if (y_cord2 > m*x_cord2 + b)
            correct = 1;
        else
            correct = -1;
        end
        val = log(1 + exp(correct*dot([x_cord2, y_cord2, 1], w2)));
        E_out = E_out + val;
    end
    E_out = E_out/N2; % Cross entropy error
    E_out_total = E_out_total + E_out;
    epoch_total = epoch_total + epoch;
end

% Average cross entropy error
E_out_average = E_out_total/simulations 

% Average number of epochs for each SGD algorithm
epoch_average = epoch_total/simulations