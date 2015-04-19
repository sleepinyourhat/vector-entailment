function matrix = InitializeNNLayer(indim, outdim, depth, initType, useBias)

% Not currently set up for ReLU
relu = 0;

if initType == 0
	if relu
		scale = sqrt(2) / sqrt(outdim);
	else
		scale = 1 / sqrt(outdim);
	end
	matrix = rand(outdim, indim, depth) .* (2 * scale) - scale;
elseif initType == 1
	if relu
		scale = 2 / sqrt(indim);
	else
		scale = 1 / sqrt(indim);
	end
	matrix = rand(outdim, indim, depth) .* (2 * scale) - scale;
elseif initType == 2
	scale = sqrt(6 / (outdim + indim));
	matrix = rand(outdim, indim, depth) .* (2 * scale) - scale;
elseif initType == 3
	matrix = zeros(outdim, indim, depth);
	for d = 1:depth
		for ind = 1:outdim
		  indices = randi([1, indim], 1, 15);
		  matrix(ind, indices, d) = normrnd(0, 1, 1, length(indices)) * 0.25;
		end
	end
	if indim == outdim && depth > 0
		ev = eig(matrix(:, :, 1));
		matrix = 1.2 .* matrix ./ ev(1, 1);
	elseif indim == 2 * outdim && depth > 0
		ev = eig(matrix(1:outdim, 1:outdim, 1));
		matrix = 1.2 .* matrix ./ ev(1, 1);
	end
elseif initType == 4
	% Softmax-specific mode. I haven't yet found any good strategies for this.
	scale = 0.1 / sqrt(indim);
	matrix = rand(outdim, indim, depth) .* (2 * scale) - scale;	
elseif initType > 4
	matrix = zeros(outdim, indim, depth);
	for d = 1:depth
		for ind = 1:outdim
		  indices = randi([1, indim], 1, 15);
		  matrix(ind, indices, d) = normrnd(0, 1, 1, length(indices)) .* (1 / initType);
		end
	end
end
	
% Add a bias column
if depth > 0 && (nargin < 5 || useBias)
	matrix = [ zeros(outdim, 1, depth), matrix ];
end

end
