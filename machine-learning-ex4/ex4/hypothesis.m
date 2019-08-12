function [p, a1, a2, z2, z3] = hypothesis(Theta1, Theta2, X)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
h2 = sigmoid(z2);

a2 = [ones(size(z2,1), 1) h2];
z3 = a2 * Theta2';
h3 = sigmoid(z3);

p = h3;

% =========================================================================


end
