% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [combinedAcc, combinedMf1, aggConfusion, combinedConAcc] = TestModel(CostGradFunc, theta, thetaDecoder, testDatasets, separateWordFeatures, hyperParams)
% Test on a collection of test sets.

% NOTE: Currently results are reported in three accuracy figures and three MacroAvgF1 figures:
% 1. Performance on the #1 test set. Used, for instance, with SICK.
% 2. Aggregate performance on the first hyperParams.firstSplit test sets. This is not used much right now.
% 3. Aggregate performance across all of the test sets. Used, for instance, with and/or and quantification.

% TODO: Currently, I only aggregate average test statistics across test datasets that use the no. 1 set 
% of labels.

if isfield(hyperParams, 'testLabelIndices')
    targetLabelSet = hyperParams.testLabelIndices(1);
else
    targetLabelSet = 1;

end

aggConfusion = zeros(hyperParams.numLabels(targetLabelSet));
targetConfusion = zeros(hyperParams.numLabels(targetLabelSet));    
aggConAcc = [];

for i = 1:length(testDatasets{1})
    if length(testDatasets{2}{i}) == 0
        continue
    end

    [ ~, ~, ~, acc, conAcc, confusion ] = CostGradFunc(theta, thetaDecoder, testDatasets{2}{i}, separateWordFeatures, hyperParams, 0);
    if conAcc(1) ~= -1
        aggConAcc = [aggConAcc; conAcc(2:end)];
    end
    if i == 1
        targetConfusion = confusion;
    end
    if hyperParams.showDetailedStats && acc > 0
        log_msg = sprintf('%s\n%s\n%s',['For test data: ', testDatasets{1}{i}, ': ', num2str(acc), ' (', num2str(GetMacroF1(confusion)), ')'], ...
            evalc('disp(confusion)'));
        Log(hyperParams.examplelog, log_msg);
    end
    if (~isfield(hyperParams, 'testLabelIndices') || hyperParams.testLabelIndices(i) == targetLabelSet)
        aggConfusion = aggConfusion + confusion;
    end
end

% Compute Accor rate from aggregate confusion matrix
targetAcc = sum(sum(eye(hyperParams.numLabels(targetLabelSet)) .* targetConfusion)) / sum(sum(targetConfusion));    
aggAcc = sum(sum(eye(hyperParams.numLabels(targetLabelSet)) .* aggConfusion)) / sum(sum(aggConfusion));    

combinedMf1 = [GetMacroF1(targetConfusion), GetMacroF1(aggConfusion)];

combinedAcc = [targetAcc, aggAcc];

combinedConAcc = [mean(aggConAcc(isfinite(aggConAcc))); aggConAcc];

end
