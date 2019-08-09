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
prediction = sigmoid(theta' * X');
logErr = y .* log(prediction') + (1 - y) .* log(1 - prediction');
thetaSize = size(theta);
sqrTheta = theta .^ 2;
J = ( (-1 / m) * sum(logErr) ) + ( (lambda / (2*m)) * sum(sqrTheta(2:thetaSize(1), 1) ) );


derivatives = (prediction' - y) .* X;
grad(1,1) = (1 / m) .* sum(derivatives(:,1));
gradSize = size(grad);
derivativesSize = size(derivatives);
grad(2:gradSize(1), 1) = (1 / m) .* sum(derivatives(:,2:derivativesSize(2))) + ((lambda / m) .* theta'(:,2:length(theta)));




% =============================================================

end
