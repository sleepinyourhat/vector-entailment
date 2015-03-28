% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function o = Objective(trueRelation, relationProbs, hyperParams)
% Compute the non-regularized objective for a single example.

assert(sum(trueRelation > 0) == 1)
for relationIndex = 1:length(trueRelation)
	if trueRelation(relationIndex) ~= 0
		o = -log(relationProbs(trueRelation(relationIndex)));
		return
	end
end

end
