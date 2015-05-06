% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matrixGradients, deltaDown ] = ...
          ComputeEmbeddingTransformGradients(matrix, delta, in, output, nonlinearityDeriv, gpu)

inPadded = [ones(1, size(in, 2), 'like', in); in];

NLDeriv = nonlinearityDeriv([], output);
delta = NLDeriv .* delta;
matrixGradients = delta * inPadded';
deltaDown = matrix(:, 2:end)' * delta;

end
