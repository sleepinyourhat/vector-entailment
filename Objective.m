% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function o = Objective(trueRelation, relationProbs, hyperParams)
% Compute the non-regularized objective for a single example.

if (trueRelation > hyperParams.numRelations)
	% Assume that both classes are equiprobable
	o = min(-0.5 * log(relationProbs(2)) - 0.5 * log(relationProbs(3)));
else
	o = -log(relationProbs(trueRelation));
end

end