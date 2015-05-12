% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function deriv = SigmoidDeriv (in, out)
% Compute the gradient of the sigmoid (now actually tanh) nonlinearity

if isempty(out)
	out = Sigmoid(in);
end

deriv = out .* (1.0 - out);

end
