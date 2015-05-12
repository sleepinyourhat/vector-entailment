function [ deltas ] = ComputeFirstPastGradient(scores, threshold, weights, deltasIn, labels, multipliers, hardMax)

% TODO: Store this.
scores = Sigmoid(scores);

deltas = zeros(size(deltasIn), 'like', deltasIn);

% Looping for now during development.
for b = 1:size(scores, 2)
	if labels(b) > 0
		deltas(labels(b), b) = deltas(labels(b), b) - (scores(labels(b), b) < threshold);
		deltas(1:labels(b) - 1, b) = deltas(1:labels(b) - 1, b) + (scores(1:labels(b) - 1, b) >= repmat(threshold, labels(b) - 1, 1));
	end
end

deltas = bsxfun(@times, deltas, multipliers');

if ~hardMax
	deltas = deltas + deltasIn .* (weights > 0);
end

deltas = deltas .* SigmoidDeriv([], scores);

end
