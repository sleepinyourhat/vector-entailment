% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ activations, mask ] = Dropout(activations, probPreserve, trainingMode, gpuMode)
% Randomly zeroes out units with probability (1 - probPreserve).

if probPreserve < 0	
	% Gradient check mode: Always drop out the first unit.
	mask = activations .* 0 + 1;
	mask(1) = 0;	
elseif ~trainingMode
	% Test mode: Scale all activations by probPreserve without dropout.
	mask = (activations .* 0 + 1) .* probPreserve;
else
	if nargin > 3 && gpuMode
		mask = rand(size(activations, 1), size(activations, 2), 'gpuArray') < probPreserve;	
	else
		mask = rand(size(activations, 1), size(activations, 2)) < probPreserve;	
	end
end

activations = mask .* activations;

end
