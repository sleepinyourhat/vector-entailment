% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ matrixGradients, deltaDown ] = ...
          ComputeEmbeddingTransformGradients(matrix, delta, in, innerOutput, nonlinearityDeriv)

inPadded = [ones(1, size(in, 2)); in];

if nargin < 7
    innerOutput = matrix * inPadded;
end

NLDeriv = nonlinearityDeriv(innerOutput);
delta = NLDeriv .* delta;
matrixGradients = delta * inPadded';
deltaDown = matrix(:, 2:end)' * delta;

end
