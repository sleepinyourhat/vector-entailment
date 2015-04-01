% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function relationRanges = ComputeRelationRanges(relationMap)
% Figure out which columns of the classifier parameters to use, given a
% label vector.

relationRanges = cell(length(relationMap), 1);
startIndex = 1;
for outerInd = 1:length(relationMap)
	relationRanges{outerInd} = startIndex:startIndex + length(relationMap{outerInd}) - 1;
	startIndex = startIndex + length(relationMap{outerInd});
end

end
