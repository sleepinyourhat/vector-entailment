% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function outVec = Sigmoid (inVec)
% Compute the sigmoid nonlinearity.

outVec = 1.0/(1.0 + exp(-inVec))

end
