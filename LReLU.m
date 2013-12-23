% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function outVec = LReLU (inVec)

outVec = max(inVec, 0) + (min(inVec, 0) * 0.01);

end