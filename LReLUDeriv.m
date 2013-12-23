% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function outVec = LReLUDeriv (inVec)

outVec = (inVec>=0) + ((inVec<0) * 0.01);

end