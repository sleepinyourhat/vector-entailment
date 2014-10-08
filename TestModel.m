% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [combined, aggConfusion] = TestModel(CostGradFunc, theta, thetaDecoder, testDatasets, constWordFeatures, hyperParams)

% Evaluate on test datasets, and show set-by-set results while aggregating
% an overall confusion matrix.

% TODO: Currently, I only aggregate across test datasets that use the no. 1 set 
% of relations.
aggConfusion = zeros(hyperParams.numRelations(1));
heldOutConfusion = zeros(hyperParams.numRelations(1));
targetConfusion = zeros(hyperParams.numRelations(1));

for i = 1:length(testDatasets{1})
    [~, ~, err, confusion] = CostGradFunc(theta, thetaDecoder, testDatasets{2}{i}, constWordFeatures, hyperParams);
    if i == 1
        targetConfusion = confusion;
    end
    if i < hyperParams.firstSplit && (~isfield(hyperParams, 'relationIndices') || hyperParams.relationIndices(i) == 1)
        heldOutConfusion = heldOutConfusion + confusion;
    end
    if hyperParams.showConfusions && err > 0
        log_msg = sprintf('%s\n%s\n%s',['For test data: ', testDatasets{1}{i}, ': ', num2str(err)], ...
            evalc('disp(confusion)'));
        Log(hyperParams.examplelog, log_msg);
    end
    if (~isfield(hyperParams, 'relationIndices') || hyperParams.relationIndices(i) == 1)
        aggConfusion = aggConfusion + confusion;
    end
end

% Compute error rate from aggregate confusion matrix
targetErr = sum(sum(eye(hyperParams.numRelations(1)) .* targetConfusion)) / sum(sum(targetConfusion));    
aggErr = sum(sum(eye(hyperParams.numRelations(1)) .* aggConfusion)) / sum(sum(aggConfusion));    
heldOutErr = sum(sum(eye(hyperParams.numRelations(1)) .* heldOutConfusion)) / sum(sum(heldOutConfusion));

MacroF1 = [GetMacroF1(targetConfusion), GetMacroF1(heldOutConfusion), GetMacroF1(aggConfusion)];
Log(hyperParams.statlog, ['MacroF1: ', evalc('disp(MacroF1)')]);

combined = [targetErr, heldOutErr, aggErr];

end