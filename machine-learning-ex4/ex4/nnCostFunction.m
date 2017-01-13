function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BEGIN COLEMAN'S CODE
% ====================
%
% PART 1: FEED FORWARD
% implement neural network all at once (no 'for' loop)
% ====================================================

% add column of ones to X -- so matrix math works with Theta1
X_aug = [ones(m,1) X];

% calculate hidden layer
% NOTE: each row of hidden_layer is the activation of the 
% hidden layer for the corresponding row of X
hidden_layer_z = X_aug*Theta1';
hidden_layer_a = sigmoid(hidden_layer_z);
% add column of ones to hidden_layer -- so matrix math works with Theta2
hla_aug = [ones(m,1) hidden_layer_a];

% calculate output layer
% NOTE: each row of output_layer is the activation of the 
% output layer for the corresponding row of X
output_layer = sigmoid(hla_aug*Theta2');

% create y_expanded matrix where each row is a vector of mostly
% zeros, except there is a one in the index of the label in y
y_expanded = zeros(size(output_layer));
for i= 1:size(y_expanded,1)
  y_expanded(i,y(i)) = 1;
end

% calculate cost function w/ regularization
inside_summation = -y_expanded.*log(output_layer) - (1-y_expanded).*log(1-output_layer);
regularization_term = (lambda/(2*m)) * ...
                      (sum(sum(Theta1(1:end,2:end).^2)) +...
                       sum(sum(Theta2(1:end,2:end).^2)));
J = (1/m) * sum(sum(inside_summation)) + regularization_term;


% PART 2: BACKPROP
% Step1: calculate delta terms
%    NOTE: deltax represents error vector of activations of layerx 

% calculate delta3 directly
delta3 = output_layer - y_expanded;
% calculate delta2 using delta3 and Theta2 and hla_aug
delta2 = (Theta2' * delta3')' .* (hla_aug.*(1-hla_aug));

% calculate upper case delta
%    NOTE: This is the gradient for each scalar value in theta
Delta2 = delta3'*hla_aug;
Delta1 = delta2(1:end,2:end)'*X_aug;

% calculate D
D2 = (1/m) .* Delta2;
D1 = (1/m) .* Delta1;
% add in regularization term
D2(1:end,2:end) = D2(1:end,2:end) + (lambda/m).*Theta2(1:end,2:end);
D1(1:end,2:end) = D1(1:end,2:end) + (lambda/m).*Theta1(1:end,2:end);

Theta1_grad = D1;
Theta2_grad = D2;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
