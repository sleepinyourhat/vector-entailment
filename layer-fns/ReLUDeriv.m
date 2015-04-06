% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function deriv = ReLUDeriv(in, out)
% Computes the gradient of the ReLU nonlinearity

if nargin > 1
	deriv = in >= 0;
else
	deriv = out >= 0;
end

end
