% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function relationRange = ComputeRelationRange(hyperParams, trueRelation)
% Figure out which columns of the classifier parameters to use, given a
% label vector.

assert(sum(trueRelation > 0) == 1)
assert(length(trueRelation) == length(hyperParams.numRelations))

startRelation = 1;
endRelation = 0;
relationSetIndex = 0;
for relationIndex = 1:length(trueRelation)
	endRelation = endRelation + hyperParams.numRelations(relationIndex);
	if trueRelation(relationIndex) == 0
		startRelation = startRelation + hyperParams.numRelations(relationIndex);
	else
		relationSet = relationIndex;
		break
	end	
end
relationRange = startRelation:endRelation;

end