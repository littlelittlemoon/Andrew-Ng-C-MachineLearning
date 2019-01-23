function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
n = size(theta);
grad = zeros(n);

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% sigmoid function
h = sigmoid(X * theta);

% means to explicitly exclude the bias term

term = (lambda / (2 * m)) * (theta(2:n)' * theta(2:n));

% Cost function
J = -(1 / m) * (y' * log(h) + (1 - y)' * log(1 - h)) + term;

% compute gradient
grad(1) = (1 / m) * X(:, 1)' * (h - y);

grad(2:n) = (1 / m) * X(:, 2:n)' * (h - y) + (lambda / m) * theta(2:n);

end
