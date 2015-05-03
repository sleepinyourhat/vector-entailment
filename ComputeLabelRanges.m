% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function labelRanges = ComputeLabelRanges(labelMap)
% Figure out which columns of the classifier parameters to use, given a
% label vector.

labelRanges = cell(length(labelMap), 1);
startIndex = 1;
for outerInd = 1:length(labelMap)
	labelRanges{outerInd} = startIndex:startIndex + length(labelMap{outerInd}) - 1;
	startIndex = startIndex + length(labelMap{outerInd});
end

end
