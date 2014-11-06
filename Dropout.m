function [ activations, mask ] = Dropout(activations, probPreserve)
% Randomly zeroes out units with probability (1 - probPreserve).

if probPreserve >= 0
	mask = rand(size(activations, 1), size(activations, 2)) < probPreserve;
else  
	% Gradient check mode: Always drop out the first unit.
	mask = zeros(size(activations, 1), size(activations, 2)) + 1;
	mask(1) = 0;	
end

activations = mask .* activations;

end
