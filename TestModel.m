function [aggErr, aggConfusion] = TestModel(theta, thetaDecoder, testDatasets, hyperParams)


% Evaluate on test datasets, and show set-by-set results
aggConfusion = zeros(hyperParams.numRelations);
for i = 1:length(testDatasets{1})
    [~, ~, err, confusion] = ComputeFullCostAndGrad(theta, thetaDecoder, testDatasets{2}{i}, hyperParams);
    if hyperParams.showConfusions && err > 0
        disp(['For ', testDatasets{1}{i}, ': ', num2str(err)])
        disp('GT:  #     =     >     <     |     ^     v')
        disp(confusion)
    end
    aggConfusion = aggConfusion + confusion;
end

% Compute error rate from summed confusion matrix
aggErr = 1 - sum(sum(eye(hyperParams.numRelations) .* aggConfusion)) / sum(sum(aggConfusion));    

end