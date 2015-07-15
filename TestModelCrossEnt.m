% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [acc] = TestModelCrossEnt(CostGradFunc, theta, thetaDecoder, testDatasets, separateWordFeatures, hyperParams)
% Test on a collection of test sets.

acc = [];
for i = 1:length(testDatasets{1})
    if length(testDatasets{2}{i}) == 0
        continue
    end

    [ ~, ~, ~, localAcc, ~, ~ ] = CostGradFunc(theta, thetaDecoder, testDatasets{2}{i}, separateWordFeatures, hyperParams, 0);
    acc = [acc localAcc];
    
    if hyperParams.showDetailedStats && localAcc > 0
        Log(hyperParams.examplelog, ['For test data: ', testDatasets{1}{i}, ': ', num2str(localAcc)]);
    end
end

end
