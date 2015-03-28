function tensor = InitializeNTNLayer(indim, outdim, initType)

% Not currently set up for ReLU
relu = 0

if initType == 0
	if relu
		scale = sqrt(2) / sqrt(indim);
	else
		scale = 1 / sqrt(indim);
	end
	tensor = rand(indim, indim, outdim) .* (2 * scale) - scale;
elseif initType == 1
	if relu
		scale = 2 / sqrt(2 * indim);
	else
		scale = 1 / sqrt(2 * indim);
	end
	tensor = rand(indim, indim, outdim) .* (2 * scale) - scale;
elseif initType == 2
	scale = sqrt(6 / (indim + outdim));
	tensor = rand(indim, indim, outdim) .* (2 * scale) - scale;
elseif initType == 3
	scale = sqrt(6 / (indim + indim + outdim));
	tensor = rand(indim, indim, outdim) .* (2 * scale) - scale;
elseif initType > 3
	tensor = zeros(indim, indim, outdim);
	for d = 1:outdim
		for ind = 1:indim
		  indices = randi([1, indim], 1, 15);
		  tensor(ind, indices, d) = normrnd(0, 1, 1, length(indices)) .* (1 / initType);
		end
	end
end

end
