% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta ] = AdaGradSGD(CostGradFunc, modelState, options, trainingData, ...
    hyperParams, testDatasets)
% Home-baked implementation of SGD with AdaGrad.

if modelState.step == 0
    modelState.prevCost = intmax;
    modelState.bestTestAcc = 0;
    modelState.lr = options.lr;
    modelState.sumSqGrad = zeros(size(modelState.theta));
    modelState.pass = 0;
    modelState.lastHundredCosts = zeros(100, 1);
end 

while modelState.pass < options.numPasses
    TestAndLog(CostGradFunc, modelState, options, trainingData, ...
        hyperParams, testDatasets);

    % Check the stopping criterion.
    %if abs(modelState.prevCost - cost(1)) < 10e-7
    %    Log(hyperParams.statlog, 'Stopped improving.');
    %    break;
    %end

    %modelState.prevCost = cost(1);

    if hyperParams.fragmentData
        modelState = TrainOnFragmentedData(CostGradFunc, trainingData, testDatasets, modelState, hyperParams, options);
    else
        modelState = TrainOnDataset(CostGradFunc, trainingData, testDatasets, modelState, hyperParams, options);
    end
        
    % Reset the AdaGrad stored weights.
    if mod(modelState.step + 1, options.resetSumSqFreq) == 0
        modelState.sumSqGrad = zeros(size(modelState.theta));
    end

    modelState.pass = modelState.pass + 1;
end

theta = modelState.theta;

end
