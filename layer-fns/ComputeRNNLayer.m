% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ activations, activationsPreNL ] = ComputeRNNLayer(l, r, matrix, NL)
% Run an RNN layer as in forward propagation.

activationsPreNL = matrix * [ones(1, size(a, 2)); l; r];
activations = NL(activationsPreNL);

end
