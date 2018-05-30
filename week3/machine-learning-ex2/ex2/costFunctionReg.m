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
 
h = sigmoid(X*theta); %m*1 mat y=m*1 mat
 
oneH = ones(m,1); 
 
oneMinusHlog = log(oneH - h);
H = log(h);

shift_theta = theta(2:size(theta));
theta_ = [0;shift_theta];
 
oneMinusY = oneH-y;
 
J = -((y' * log(h) + oneMinusY' * oneMinusHlog))/m + (lambda/(2*m))* (theta_')*theta_; 
 
grad = (1/m)*(((h-y)'* X)'+ (lambda)*theta_);
 
% =============================================================
 
end

