% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ activations, mask ] = Dropout(activations, probPreserve, trainingMode)
% Randomly zeroes out units with probability (1 - probPreserve).

if probPreserve < 0	
	% Gradient check mode: Always drop out the first unit.
	mask = zeros(size(activations, 1), size(activations, 2)) + 1;
	mask(1) = 0;	
elseif ~trainingMode
	% Test mode: Scale all activations by probPreserve without dropout.
	mask = ones(size(activations, 1), size(activations, 2)) .* probPreserve;
else
	mask = rand(size(activations, 1), size(activations, 2)) < probPreserve;	
end

activations = mask .* activations;

end
