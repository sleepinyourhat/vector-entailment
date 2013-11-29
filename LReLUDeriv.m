function outVec = LReLUDeriv (inVec)

outVec = (inVec>=0) + ((inVec<0) * 0.01);

end