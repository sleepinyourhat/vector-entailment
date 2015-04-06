% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function deriv = LReLUDeriv (in, out)
% Compute the gradient of the LReLU nonlinearity

if nargin > 1
	deriv = (out >= 0) + ((out < 0) * 0.01);
else
	deriv = (in >= 0) + ((in < 0) * 0.01);
end

end
