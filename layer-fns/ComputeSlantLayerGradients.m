% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ deltas ] = ComputeSlantLayerGradients(in, deltas, temperature)

accum = ones(1, size(in, 2), 'like', in) .* temperature;
for i = 1:size(in, 1)
	if i > 1
		deltas(1:i - 1, :) = bsxfun(@plus, deltas(1:i - 1, :), -deltas(i, :) .* in(i, :) ./ (accum .^ 2));
	end
	deltas(i, :) = deltas(i, :) ./ accum;
	accum = accum + in(i, :);
end

deltas = deltas .* ReLUDeriv(in);

end
