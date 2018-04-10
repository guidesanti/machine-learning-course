function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute cost

% First part
h = (X*theta);
h = sigmoid(h);
a = log(h);
b = (1 - y);
c = log(1 - h);
temp1 = a'*y;
temp2 = c'*b;

% Regularization part
theta_aux = theta(2:end);
aux1 = (lambda/(2*m));
aux2 = (theta_aux'*theta_aux);
reg = aux1*aux2;

% Final cost
J = (((-1)/m)*(temp1 + temp2) + reg);

% Compute grad
a = (h - y)';
grad = (1/m).*(a*X);
grad = grad';
grad1 = grad(1);
grad = grad + ((lambda/m)*theta);
grad(1) = grad1;

% =============================================================

end
