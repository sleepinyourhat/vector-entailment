% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [activations, activationsPreNL] = ComputeRNNLayer(a, b, matrix, bias, NL)
% Run an RNN layer as in forward propagation.

activationsPreNL = matrix * [a; b] + bias;

activations = NL(activationsPreNL);

end