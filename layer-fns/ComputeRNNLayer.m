% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ activations, activationsPreNL ] = ComputeRNNLayer(l, r, matrix, NL)
% Run an RNN layer as in forward propagation.

in = [ones([1, size(l, 2)], 'like', l); l; r];
activationsPreNL = matrix * in;
activations = NL(activationsPreNL);

end
