% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function outVec = TanhDeriv (inVec)
% Compute the gradient of the tanh nonlinearity

outVec = sech(inVec).^2;

end
