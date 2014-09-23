% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function outVec = ReLU (inVec)
% Computes the ReLU nonlinearity.

outVec = max(inVec, 0);

end