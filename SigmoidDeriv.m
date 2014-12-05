% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function outVec = SigmoidDeriv (inVec)
% Compute the gradient of the sigmoid (now actually tanh) nonlinearity

outVec = sech(inVec).^2;

end
