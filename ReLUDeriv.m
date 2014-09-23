% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function outVec = ReLUDeriv (inVec)
% Computes the gradient of the ReLU nonlinearity

outVec = (inVec>=0);

end