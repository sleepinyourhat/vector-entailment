% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [combined, aggConfusion] = TestModel(CostGradFunc, theta, thetaDecoder, testDatasets, hyperParams)

% Evaluate on test datasets, and show set-by-set results while aggregating
% an overall confusion matrix.
aggConfusion = zeros(hyperParams.numDataRelations);
heldOutConfusion = zeros(hyperParams.numDataRelations);
targetConfusion = zeros(hyperParams.numDataRelations);

for i = 1:length(testDatasets{1})
    [~, ~, err, confusion] = CostGradFunc(theta, thetaDecoder, testDatasets{2}{i}, hyperParams);
    if i == 1
        targetErr = err;
        targetConfusion = confusion;
    end
    if i < hyperParams.firstSplit
        heldOutConfusion = heldOutConfusion + confusion;
    end
    if hyperParams.showConfusions && err > 0
        log_msg = sprintf('%s\n%s\n%s',['For ', testDatasets{1}{i}, ': ', num2str(err)], ...
            'GT:  #     =     >     <     |     ^     v', evalc('disp(confusion)'));
        Log(hyperParams.examplelog, log_msg);
    end
    aggConfusion = aggConfusion + confusion;
end

% Compute error rate from aggregate confusion matrix
aggErr = 1 - sum(sum(eye(hyperParams.numDataRelations) .* aggConfusion)) / sum(sum(aggConfusion));    
heldOutErr = 1 - sum(sum(eye(hyperParams.numDataRelations) .* heldOutConfusion)) / sum(sum(heldOutConfusion));

MacroF1 = [GetMacroF1(targetConfusion), GetMacroF1(heldOutConfusion), GetMacroF1(aggConfusion)];
Log(hyperParams.statlog, ['MacroF1: ', evalc('disp(MacroF1)')]);

combined = [targetErr, heldOutErr, aggErr];

end