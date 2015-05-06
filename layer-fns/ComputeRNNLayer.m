% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ activations, activationsPreNL ] = ComputeRNNLayer(l, r, matrix, NL)
% Run an RNN layer as in forward propagation.

activationsPreNL = matrix * padarray([l; r], 1, 1, 'pre');
activations = NL(activationsPreNL);

end
