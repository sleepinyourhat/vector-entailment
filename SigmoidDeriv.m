% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function outVec = SigmoidDeriv (inVec)

outVec = sech(inVec).^2;

end
