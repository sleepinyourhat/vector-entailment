% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function result = ComputeRNNLayer(a, b, matrix, bias, NL)
% Run an RNN layer as in forward propagation.

result = NL(matrix * [a; b] + bias);

end