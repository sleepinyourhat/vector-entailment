function matrix = InitializeLSTMLayer(dim, depth, initType)

if initType == 1
	scale = 1 / sqrt((2 * dim) + 1);
	matrix = rand(4 * dim, 2 * dim + 1, depth) .* (2 * scale) - scale;
	matrix(:, 1) = 3 * scale;
elseif initType == 2
	scale =  sqrt(6 / ((3 * dim) + 1));
	matrix = rand(4 * dim, 2 * dim + 1, depth) .* (2 * scale) - scale;
	matrix(:, 1) = 3 * scale;
end

end
