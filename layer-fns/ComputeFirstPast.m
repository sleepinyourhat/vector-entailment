function [ weights, correctWeight, loss ] = ComputeFirstPast(scores, threshold, labels, multipliers, hardMax)

% Sigmoid to get a bounded range.
scores = Sigmoid(scores);

% Lower the threshold if nothing passes it.
effectiveThreshold = min([repmat(threshold, size(scores, 2)); max(scores, [], 1)]);

% Choose the first entry to pass the threshold.
[ ~, bestIndex ] = max(scores >= repmat(effectiveThreshold, size(scores, 1), 1), [], 1);

loss = zeros(size(scores, 2), 1, 'like', scores);
paddedScores = [zeros(1, size(scores, 2)); scores];
for b = 1:size(scores, 2)
	loss(b) = max(0, threshold - paddedScores(labels(b) + 1, b));

	if labels(b) > 1
		competingScores = scores(1:labels(b) - 1, b);
		loss(b) = loss(b) + ...
			sum(competingScores(competingScores >= repmat(threshold, labels(b) - 1, 1)) - threshold);
	end
end

weights = zeros(size(scores), 'like', scores);
if hardMax
	weights(sub2ind(size(scores), bestIndex, 1:size(scores, 2))) = 1;
else
	weights(sub2ind(size(scores), bestIndex, 1:size(scores, 2))) = scores(sub2ind(size(scores), bestIndex, 1:size(scores, 2)));
end

paddedWeights = [zeros(1, size(weights, 2)); weights];
correctWeight = paddedWeights(sub2ind(size(paddedWeights), labels' + 1, 1:length(labels)));

loss = loss .* multipliers;

end
