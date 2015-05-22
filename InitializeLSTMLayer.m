function matrix = InitializeLSTMLayer(dim, depth, initType, tree, gpu)

if tree
	numOutputs = 5;
	Ir = 0 * dim + 1:1 * dim;
	Fr = 1 * dim + 1:3 * dim;
	Or = 3 * dim + 1:4 * dim;
else
	numOutputs = 4;
	Ir = 0 * dim + 1:1 * dim;
	Fr = 1 * dim + 1:2 * dim;
	Or = 2 * dim + 1:3 * dim;
end

if initType == 1
	scale = 1 / sqrt((2 * dim) + 1);
	matrix = fRand([numOutputs * dim, 2 * dim + 1, depth], gpu) .* (2 * scale) - scale;
	matrix(:, 1) = 0;
elseif initType == 2
	scale =  sqrt(6 / ((3 * dim) + 1));
	matrix = fRand([numOutputs * dim, 2 * dim + 1, depth], gpu) .* (2 * scale) - scale;
	matrix(:, 1) = 0;
elseif initType == 3
	scale =  sqrt(6 / ((3 * dim) + 1));
	matrix = fRand([numOutputs * dim, 2 * dim + 1, depth], gpu) .* (2 * scale) - scale;
	matrix(:, 1) = 0;
	matrix(Fr, 1) = 1;
elseif initType == 4
	scale =  sqrt(6 / ((3 * dim) + 1));
	matrix = fRand([numOutputs * dim, 2 * dim + 1, depth], gpu) .* (2 * scale) - scale;
	matrix(:, 1) = 0;
	matrix(Fr, 1) = 1;
	matrix([Ir, Or], 1) = -1;
elseif initType == 5
	scale =  sqrt(6 / ((3 * dim) + 1));
	matrix = fRand([numOutputs * dim, 2 * dim + 1, depth], gpu) .* (2 * scale) - scale;
	matrix(:, 1) = 0;
	matrix(Fr, 1) = 5;
else
	assert(false, 'Bad LSTM initialization type.');
end


end
