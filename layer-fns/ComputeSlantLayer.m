% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ out ] = ComputeSlantLayer(in, temperature)

out = zeros(size(in), 'like', in);
accum = ones(1, size(in, 2), 'like', in) .* temperature;

in = ReLU(in);

for i = 1:size(in, 1)
	out(i, :) = in(i, :) ./ accum;
	accum = accum + in(i, :);
end

end
